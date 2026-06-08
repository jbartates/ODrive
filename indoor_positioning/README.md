# Indoor Positioning System (GoPro Max 2 + Raspberry Pi 4)

A self-contained indoor positioning system for a mobile robot. A **GoPro Max 2**
in USB webcam mode feeds frames to a **Raspberry Pi 4** onboard the robot. The
Pi detects **ArUco fiducial markers** placed at known positions in the room for
absolute pose fixes, runs **visual odometry** for smooth motion between markers,
optionally folds in the GoPro's **IMU telemetry** (gyro/accel/gravity from the
GPMF metadata track) for accurate heading, and **fuses** everything with an
Extended Kalman Filter. The fused pose `(x, y, heading)` is **published over the
network** (UDP/TCP JSON) and **logged** to disk.

This package is independent of the ODrive firmware in the rest of this repo — it
runs as a normal Python program on the Pi and can optionally feed the ODrive a
target velocity for closed-loop navigation (see *Driving the ODrive* below).

```
GoPro Max 2 ──USB──► Raspberry Pi 4
                       │
   ┌───────────────────┴───────────────────────────────┐
   │ frame ─► undistort ─► ArUco detect ─► map localiser │  absolute fixes
   │              └─────► visual odometry (translation)  │  relative motion
   │   GPMF IMU ─► preintegrate gyro (heading) ──────────┤  relative heading
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
* **IMU heading** — the GoPro's gyro gives a far better short-term heading than
  vision. We project it onto the gravity vector so the estimate is independent
  of how the camera is mounted, remove bias during stationary periods, and feed
  it to the EKF's prediction step (see *IMU fusion* below).
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
│   ├── imu/                 # GoPro GPMF/IMU telemetry + yaw preintegration
│   │   ├── mp4.py           #   pure-python MP4 reader (finds the gpmd track)
│   │   ├── gpmf.py          #   GPMF KLV parser (GYRO/ACCL/GRAV + SCAL)
│   │   ├── source.py        #   GPMF + in-memory IMU sources
│   │   └── preintegrator.py #   gravity-projected gyro -> yaw increment
│   ├── fusion/ekf.py        # the EKF                       (pure numpy)
│   ├── output/              # UDP/TCP publisher + JSONL logger
│   └── system.py            # the orchestrator that wires it together
├── scripts/
│   ├── run_positioning.py   # live entry point
│   ├── calibrate_camera.py  # produce camera_calibration.yaml
│   ├── generate_markers.py  # printable ArUco markers
│   └── dump_telemetry.py    # inspect a clip's IMU + pick imu.yaw_sign
├── config/                  # *.example.yaml — copy and edit
└── tests/                   # pure-math unit tests (no camera needed)
```

The heavy OpenCV-dependent pieces import `cv2` lazily, so the pure-math core
(`geometry`, `config`, `fusion`, IMU/GPMF parsing, transform chain) is fully
unit-tested without a camera or even OpenCV installed.

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

## IMU fusion (GoPro GPMF telemetry)

The GoPro Max 2 records inertial telemetry — `GYRO`, `ACCL`, `GRAV` — into the
**GPMF metadata track** of the MP4 it writes to the SD card. This package parses
that track (a pure-Python MP4 reader + GPMF KLV parser, no ffmpeg required),
converts the gyro into a heading-rate signal, and feeds it to the EKF's
prediction step. Heading from a gyro is dramatically better over short
timescales than vision-derived heading, so the fused pose is smoother and the
between-marker drift is much smaller. Visual odometry still supplies the
translation; the IMU takes over yaw.

How the yaw is extracted (`ips/imu/preintegrator.py`):

* **Mount-agnostic** — the gyro vector is projected onto the vertical axis
  defined by gravity (`GRAV`, or low-passed `ACCL` as a fallback), so it doesn't
  matter how the camera is tilted or rotated on the robot.
* **Bias removal** — while the robot is detected stationary (gyro magnitude
  below `imu.stationary_gyro_thresh`) the gyro bias is learned online and a
  rotational zero-velocity update (ZUPT) stops the heading from creeping.
* **Honest covariance** — each yaw increment carries a variance from the gyro
  noise model, so the EKF weighs it correctly against the marker fixes.

> **Important:** GPMF telemetry is **not** present in the live USB-webcam (UVC)
> stream — only in recorded clips. So there are two ways to use it:
>
> 1. **Offline replay** of a recording (full IMU fusion). Point both the camera
>    and the IMU at the same MP4:
>    ```yaml
>    camera: { source: file, device: /path/clip.MP4 }
>    imu:    { source: gpmf, video_path: /path/clip.MP4 }
>    ```
>    The file camera emits presentation timestamps so video and IMU stay aligned.
> 2. **Live** runs carry no GoPro telemetry. Either record + post-process, or
>    wire a separate IMU to the Pi and push samples in:
>    ```python
>    from ips.imu import ListImuSource, ImuSample
>    system.set_imu_source(ListImuSource([...]))  # or a custom ImuSource
>    ```

Inspect a clip's telemetry and choose the heading sign:

```bash
python scripts/dump_telemetry.py my_clip.MP4 --plot heading.csv
```

Turn the robot a known direction (say, left/CCW) while recording; if the
reported net heading change has the wrong sign, set `imu.yaw_sign: -1.0`.

## Tuning notes

* `odometry.flow_scale_m` converts image flow to metres; drive a known distance
  and scale it until the dead-reckoned distance matches.
* `imu.gyro_noise_std` sets how much the EKF trusts the gyro heading; lower it
  if heading is noisy-but-unbiased, raise it if the gyro is drifting.
* `imu.yaw_sign` flips heading direction; `imu.stationary_gyro_thresh` controls
  when bias learning / ZUPT kicks in.
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
python -m pytest          # 50 tests, no camera/OpenCV required
```

The tests cover the geometry helpers, the EKF (prediction, update, angle
wrapping, outlier gating, covariance symmetry, dead-reckon-then-correct), the
full marker→world transform chain, config/marker-map loading, the UDP publisher
round-trip, the GPMF KLV parser + MP4 sample-table maths, and IMU preintegration
(constant-rate integration, gravity-projection tilt invariance, bias/ZUPT,
cross-batch bridging, and the IMU→EKF heading update).
