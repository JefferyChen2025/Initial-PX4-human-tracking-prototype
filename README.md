# PX4 Human Tracking

A human-tracking prototype that uses YOLO, OpenCV, ByteTrack, MAVSDK, and PX4 Offboard control.

The program detects people, allows the user to select one person, and rotates the drone to keep that person near the center of a forward-facing camera.

## Features

- YOLO person detection
- ByteTrack target tracking
- Click-to-select target
- PX4 yaw-rate control
- Automatic stop when the target is lost
- Detection image and CSV saving
- Preview mode without PX4

## Installation

```bash
python -m pip install -r requirements.txt
```

## Preview Mode

```bash
python human_tracking_px4.py
```

## PX4 Mode

```bash
python human_tracking_px4.py --px4
```

The default MAVSDK connection is:

```text
udpin://0.0.0.0:14540
```

Before enabling control:

1. Start PX4 SITL.
2. Take off using QGroundControl.
3. Enter Hold mode.
4. Click a detected person.
5. Press `E`.

## Controls

| Input | Action |
|---|---|
| Mouse click | Select a person |
| `E` | Enable tracking |
| `S` | Stop tracking |
| `R` | Clear target |
| `Q` | Quit |

## PX4 SITL

Start PX4 and Gazebo:

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
```

If PX4 runs in WSL2 and Python runs in Windows, find the Windows host address:

```bash
ip route show default
```

Then enter this command in the PX4 console:

```text
mavlink start -u 14600 -o 14540 -t WINDOWS_IP -m onboard -r 1000000
```

Example:

```text
mavlink start -u 14600 -o 14540 -t 172.31.64.1 -m onboard -r 1000000
```

Run the program in Windows:

```powershell
python human_tracking_px4.py --px4
```

## Notes

- The program only controls drone yaw.
- It does not automatically arm, take off, land, or disarm.
- ByteTrack IDs are temporary tracking IDs.
- Test the system in PX4 SITL before using real hardware.
