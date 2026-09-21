import argparse
import asyncio
import csv
import math
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.40

KP_YAW = 20.0
MAX_YAW_RATE = 15.0
DEADBAND = 0.08

LOST_TIMEOUT = 2.0
SAVE_COOLDOWN = 5.0

SAVE_FOLDER = "tracked_humans"
CSV_FILE = os.path.join(
    SAVE_FOLDER,
    "tracked_detections.csv"
)

WINDOW_NAME = "PX4 Human Tracking"


class TrackingState:

    def __init__(self):
        self.target_id = None
        self.target_visible = False
        self.last_seen = 0.0
        self.horizontal_error = 0.0
        self.yaw_rate = 0.0
        self.control_enabled = False
        self.quit_requested = False
        self.status = "PREVIEW"

    def clear_target(self):
        self.target_id = None
        self.target_visible = False
        self.horizontal_error = 0.0
        self.yaw_rate = 0.0
        self.control_enabled = False


def calculate_yaw_rate(horizontal_error):

    if not math.isfinite(horizontal_error):
        return 0.0

    if abs(horizontal_error) < DEADBAND:
        return 0.0

    yaw_rate = KP_YAW * horizontal_error

    return max(
        -MAX_YAW_RATE,
        min(MAX_YAW_RATE, yaw_rate)
    )


async def px4_control_loop(state, connection_address):

    from mavsdk import System
    from mavsdk.offboard import (
        OffboardError,
        VelocityBodyYawspeed
    )

    drone = System()

    state.status = "CONNECTING TO PX4"

    await drone.connect(
        system_address=connection_address
    )

    async for connection_state in (
        drone.core.connection_state()
    ):
        if connection_state.is_connected:
            break

    print("Connected to PX4.")

    state.status = (
        "PX4 READY: TAKE OFF, ENTER HOLD, "
        "SELECT TARGET, PRESS E"
    )

    offboard_active = False

    async def send_yaw_rate(yaw_rate):

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                0.0,
                0.0,
                0.0,
                yaw_rate
            )
        )

    try:
        while not state.quit_requested:

            target_lost = (
                state.target_id is not None
                and time.monotonic() - state.last_seen
                > LOST_TIMEOUT
            )

            if target_lost:
                state.clear_target()
                state.status = "TARGET LOST"

            if state.control_enabled:

                if not offboard_active:

                    state.status = "STARTING OFFBOARD"

                    for _ in range(25):
                        await send_yaw_rate(0.0)
                        await asyncio.sleep(0.05)

                    try:
                        await drone.offboard.start()
                        offboard_active = True
                        state.status = "TRACKING"

                    except OffboardError as error:
                        state.control_enabled = False
                        state.status = (
                            "OFFBOARD ERROR: "
                            f"{error._result.result}"
                        )

                if offboard_active:

                    if state.target_visible:
                        state.yaw_rate = (
                            calculate_yaw_rate(
                                state.horizontal_error
                            )
                        )
                    else:
                        state.yaw_rate = 0.0

                    await send_yaw_rate(
                        state.yaw_rate
                    )

            elif offboard_active:

                await send_yaw_rate(0.0)

                try:
                    await drone.offboard.stop()
                except OffboardError:
                    pass

                offboard_active = False
                state.yaw_rate = 0.0
                state.status = "CONTROL STOPPED"

            await asyncio.sleep(0.05)

    finally:

        if offboard_active:

            try:
                await send_yaw_rate(0.0)
                await drone.offboard.stop()

            except OffboardError:
                pass


async def main(arguments):

    os.makedirs(
        SAVE_FOLDER,
        exist_ok=True
    )

    if not os.path.exists(CSV_FILE):

        with open(
            CSV_FILE,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "track_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "image_filename"
            ])

    model = YOLO(MODEL_PATH)

    camera_source = (
        int(arguments.camera)
        if arguments.camera.isdigit()
        else arguments.camera
    )

    camera = cv2.VideoCapture(
        camera_source
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    state = TrackingState()

    detections = []
    pending_click = []
    last_save_time = {}

    def mouse_callback(
        event,
        x_coordinate,
        y_coordinate,
        flags,
        parameter
    ):

        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click.append(
                (
                    x_coordinate,
                    y_coordinate
                )
            )

    cv2.namedWindow(
        WINDOW_NAME
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    px4_task = None

    if arguments.px4:

        px4_task = asyncio.create_task(
            px4_control_loop(
                state,
                arguments.connection
            )
        )

    print(
        "Click target | E enable | "
        "S stop | R reset | Q quit"
    )

    try:
        while not state.quit_requested:

            success, frame = camera.read()

            if not success:
                break

            results = await asyncio.to_thread(
                model.track,
                frame,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=CONFIDENCE_THRESHOLD
            )

            detections = []

            boxes = results[0].boxes

            if (
                boxes is not None
                and boxes.id is not None
            ):

                for box in boxes:

                    track_id = int(
                        box.id[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    detections.append({
                        "track_id": track_id,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

            if pending_click:

                click_x, click_y = (
                    pending_click.pop(0)
                )

                for detection in detections:

                    inside_box = (
                        detection["x1"]
                        <= click_x
                        <= detection["x2"]
                        and detection["y1"]
                        <= click_y
                        <= detection["y2"]
                    )

                    if inside_box:

                        state.clear_target()

                        state.target_id = (
                            detection["track_id"]
                        )

                        state.last_seen = (
                            time.monotonic()
                        )

                        print(
                            "Selected Person "
                            f"#{state.target_id}"
                        )

                        break

            state.target_visible = False
            selected_detection = None

            frame_height, frame_width = (
                frame.shape[:2]
            )

            image_center_x = (
                frame_width / 2.0
            )

            for detection in detections:

                track_id = (
                    detection["track_id"]
                )

                x1 = detection["x1"]
                y1 = detection["y1"]
                x2 = detection["x2"]
                y2 = detection["y2"]

                is_target = (
                    track_id
                    == state.target_id
                )

                color = (
                    (0, 255, 255)
                    if is_target
                    else (0, 255, 0)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    f"Person #{track_id} "
                    f"{detection['confidence']:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                if is_target:

                    state.target_visible = True
                    state.last_seen = (
                        time.monotonic()
                    )

                    target_center_x = (
                        x1 + x2
                    ) / 2.0

                    state.horizontal_error = (
                        (
                            target_center_x
                            - image_center_x
                        )
                        / image_center_x
                    )

                    selected_detection = (
                        detection
                    )

            if not arguments.px4:

                if (
                    state.control_enabled
                    and state.target_visible
                ):
                    state.yaw_rate = (
                        calculate_yaw_rate(
                            state.horizontal_error
                        )
                    )
                else:
                    state.yaw_rate = 0.0

                state.status = (
                    "PREVIEW: NO PX4 COMMANDS"
                )

            if selected_detection is not None:

                track_id = (
                    selected_detection["track_id"]
                )

                current_time = (
                    time.monotonic()
                )

                elapsed = (
                    current_time
                    - last_save_time.get(
                        track_id,
                        -math.inf
                    )
                )

                if elapsed >= SAVE_COOLDOWN:

                    timestamp = datetime.now()

                    image_filename = (
                        f"person_{track_id}_"
                        f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
                        ".jpg"
                    )

                    image_path = os.path.join(
                        SAVE_FOLDER,
                        image_filename
                    )

                    cv2.imwrite(
                        image_path,
                        frame
                    )

                    with open(
                        CSV_FILE,
                        mode="a",
                        newline="",
                        encoding="utf-8"
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            timestamp.isoformat(),
                            track_id,
                            round(
                                selected_detection[
                                    "confidence"
                                ],
                                3
                            ),
                            selected_detection["x1"],
                            selected_detection["y1"],
                            selected_detection["x2"],
                            selected_detection["y2"],
                            image_filename
                        ])

                    last_save_time[
                        track_id
                    ] = current_time

            cv2.line(
                frame,
                (
                    int(image_center_x),
                    0
                ),
                (
                    int(image_center_x),
                    frame_height
                ),
                (255, 255, 0),
                1
            )

            information = [
                state.status,
                (
                    f"Target: {state.target_id} | "
                    f"Yaw: {state.yaw_rate:+.1f} deg/s"
                ),
                (
                    "Click target | E enable | "
                    "S stop | R reset | Q quit"
                )
            ]

            for index, text in enumerate(
                information
            ):

                cv2.putText(
                    frame,
                    text,
                    (
                        10,
                        25 + index * 25
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                state.quit_requested = True

            elif key == ord("e"):

                if state.target_id is not None:
                    state.control_enabled = True

            elif key == ord("s"):
                state.control_enabled = False

            elif key == ord("r"):
                state.clear_target()

            await asyncio.sleep(0)

    finally:

        state.control_enabled = False
        state.quit_requested = True

        if px4_task is not None:
            await px4_task

        camera.release()
        cv2.destroyAllWindows()

        print("Program closed.")


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--px4",
        action="store_true"
    )

    parser.add_argument(
        "--connection",
        default="udpin://0.0.0.0:14540"
    )

    parser.add_argument(
        "--camera",
        default="0"
    )

    return parser.parse_args()


if __name__ == "__main__":

    try:
        asyncio.run(
            main(
                parse_arguments()
            )
        )

    except KeyboardInterrupt:
        print("Program interrupted.")
