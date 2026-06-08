# Indoor Positioning System (GoPro Max 2 + Raspberry Pi 4)

A self-contained indoor positioning system for a mobile robot. A **GoPro Max 2**
in USB webcam mode feeds frames to a **Raspberry Pi 4** onboard the robot. The
Pi detects **ArUco fiducial markers** placed at known positions in the room for
absolute pose fixes, runs **visual odometry** for smooth motion between markers,
and **fuses** the two with an Extended Kalman Filter. The fused pose
`(x, y, heading)` is **published over the network** (UDP/TCP JSON) and **logged**
to disk.

This package is independent of the ODrive firmware in the rest of this repo — it
runs as a normal Python program on the Pi and can optionally feed the ODrive a
target velocity for closed-loop navigation (see *Driving the ODrive* below).

```
GoPro Max 2 ──USB──► Raspberry Pi 4
                       │
   ┌───────────────────┴───────────────────────────────┐
   │ frame ─► undistort ─► ArUco detect ─► map localiser │  absolute fixes
   │              └─────► visual odometry                │  relative motion
   │                           └────► EKF fusion ────────┼─► UDP/TCP + JSONL log
   └────────────────────────────────────────────────────┘
```

## Why this design

* **360 camera + markers** — the Max 2's huge field of view means a wall/ceiling
  marker is almost always visible, so absolute fixes are frequent and drift is
  bounded. Markers are cheap, robust, and light enough to run on a Pi 4.
* **Odometry fusion** — between marker sightings, sparse optical-flow odometry
  keeps the pose smooth and fills momentary gaps. The EKF reconciles the two and
  rejects bad marker fixes with a Mahalanobis gate.
* **Decoupled output** — pose is published as plain JSON so *anything* can
  consume it (a navigation node, a plotter, or the ODrive control layer).

## Layout

```
indoor_positioning/
├── ips/                     # the package
│   ├── geometry.py          # SE(2)/SE(3) helpers, Pose2D  (pure numpy)
│   ├── config.py            # typed config + YAML loaders   (pure numpy)
│   ├── camera/              # pluggable camera sources + lens undistort
│   │   ├── gopro_usb.py     #   GoPro Max 2 USB webcam backend
│   │   ├── opencv_camera.py #   generic V4L2 backend
│   │   └── file_camera.py   #   replay video/images for offline testing
│   ├── markers/             # ArUco detection + map-based localisation
│   ├── odometry/            # sparse optical-flow visual odometry
│   ├── fusion/ekf.py        # the EKF                       (pure numpy)
│   ├── output/              # UDP/TCP publisher + JSONL logger
│   └── system.py            # the orchestrator that wires it together
├── scripts/
│   ├── run_positioning.py   # live entry point
│   ├── calibrate_camera.py  # produce camera_calibration.yaml
│   └── generate_markers.py  # printable ArUco markers
├── config/                  # *.example.yaml — copy and edit
└── tests/                   # pure-math unit tests (no camera needed)
```

The heavy OpenCV-dependent pieces import `cv2` lazily, so the pure-math core
(`geometry`, `config`, `fusion`, transform chain) is fully unit-tested without a
camera or even OpenCV installed.

## Setup on the Raspberry Pi 4

```bash
sudo apt install -y python3-pip
cd indoor_positioning
pip install -r requirements.txt          # numpy, opencv-contrib-python, pyyaml
```

### 1. Bring up the GoPro as a USB webcam

1. Put the Max 2 into **USB / webcam mode** (long-press the mode button, or use
   the GoPro Quik app) and connect it to the Pi with a USB-C cable.
2. Start the webcam endpoint so it appears as `/dev/video*`. The open-source
   [`gopro_as_webcam_on_linux`](https://github.com/jschmid1/gopro_as_webcam_on_linux)
   helper does this; if its CLI is on `PATH` as `gopro-webcam`, this package will
   auto-start it. Otherwise start it yourself, then confirm the device:
   ```bash
   v4l2-ctl --list-devices
   ```
3. **Choose a reframed *linear* output** in the webcam settings so frames are
   rectilinear and match a pinhole calibration. (If you must use the raw
   equirectangular feed, calibrate with `--model fisheye` and the undistorter
   will dewarp it.)

Point `camera.device` in `config.yaml` at the resulting `/dev/videoN`.

### 2. Calibrate the camera

Capture ~20 photos of a printed chessboard through the exact webcam view you'll
run with, then:

```bash
python scripts/calibrate_camera.py --images calib_images/ \
    --rows 6 --cols 9 --square 0.025 --out config/camera_calibration.yaml
```

Aim for an RMS reprojection error well under 1 px.

### 3. Print and place markers

```bash
python scripts/generate_markers.py --ids 0-9 --size 600 --out markers/
```

Print each at a known physical size, set that size as `detector.marker_length_m`,
mount them on walls/ceiling, measure their world positions, and record them in
`config/marker_map.yaml` (see `marker_map.example.yaml` for the format). The
accuracy of your map *is* the accuracy of your positioning — measure carefully.

### 4. Configure and run

```bash
cp config/config.example.yaml          config/config.yaml
cp config/marker_map.example.yaml      config/marker_map.yaml
# edit both for your robot + room, then:

python scripts/run_positioning.py \
    --config config/config.yaml \
    --calibration config/camera_calibration.yaml \
    --map config/marker_map.yaml
```

Listen to the pose stream from any machine on the LAN:

```bash
nc -ul 9870        # line-delimited JSON, one pose per line
```

To start automatically on boot, install a `systemd` unit that runs the command
above (set `WorkingDirectory` to this folder and `Restart=on-failure`).

## Output schema

One JSON object per UDP datagram / log line:

```json
{"type":"pose","t":1234.567,"x":1.23,"y":-0.45,"theta":0.78,
 "cov":[...9 floats...],"n_markers":2,"fixed":true}
```

* `x`, `y` — world position in metres; `theta` — heading in radians (CCW from
  world +x).
* `cov` — row-major 3×3 covariance of `(x, y, theta)`; use it to gauge trust.
* `n_markers` — markers accepted in the latest fix. `fixed` — `true` once at
  least one marker fix has initialised the filter.

## Coordinate conventions

* **World/map frame**: building-fixed, robot drives on `z = 0`. `theta` is yaw
  CCW from world +x.
* **Body frame**: +x forward, +y left, +z up; origin at the drive centre. The
  camera mount offset/orientation relative to the body is set in `config.camera`.
* **Camera optical frame**: OpenCV convention (+x right, +y down, +z forward).
* **Marker frame**: OpenCV convention — origin at the marker centre, +x right,
  +y up, +z out of the printed face.

## Tuning notes

* `odometry.flow_scale_m` converts image flow to metres; drive a known distance
  and scale it until the dead-reckoned distance matches.
* `fusion.process_std_*` controls how fast the estimate is allowed to move on
  odometry alone; `fusion.marker_std_*` controls how strongly markers pull it.
* `fusion.mahalanobis_gate` rejects inconsistent marker fixes — lower it if a
  mislabelled/duplicated marker is corrupting the pose.

## Driving the ODrive (optional)

This package only *produces* pose. To close a navigation loop with the ODrive in
this repo, write a small consumer that reads the UDP pose stream and commands a
body velocity toward a waypoint, e.g. using `tools/odrive`:

```python
import odrive
from odrive.enums import AXIS_STATE_CLOSED_LOOP_CONTROL
# read pose from UDP 9870, compute (v_left, v_right) toward the target, then:
# odrv.axis0.controller.input_vel = v_left ; odrv.axis1.controller.input_vel = v_right
```

Keeping it as a separate consumer means the positioning system never blocks on
motor I/O and can be reused for non-ODrive robots.

## Testing

```bash
pip install numpy pyyaml pytest
python -m pytest          # 33 tests, no camera/OpenCV required
```

The tests cover the geometry helpers, the EKF (prediction, update, angle
wrapping, outlier gating, covariance symmetry, dead-reckon-then-correct), the
full marker→world transform chain, config/marker-map loading, and the UDP
publisher round-trip.
