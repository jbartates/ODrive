"""Configuration loading and typed config objects.

Configuration is split across three YAML files so that the parts which
change at different rates live separately:

* ``config.yaml``            -- runtime/system wiring (camera, output, fusion)
* ``camera_calibration.yaml`` -- intrinsics, produced by calibrate_camera.py
* ``marker_map.yaml``         -- world poses of every fiducial marker

Only ``pyyaml`` and ``numpy`` are required here, so config can be validated
in CI without a camera attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:  # pragma: no cover - trivial import guard
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyyaml is required to load configuration. Install with "
        "`pip install pyyaml`."
    ) from exc

from .geometry import Pose2D, make_transform, rot_z


def _load_yaml(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}")
    return data


@dataclass
class CameraConfig:
    """How to open the camera and how it is mounted on the robot."""

    source: str = "gopro_usb"  # "gopro_usb" | "opencv" | "file"
    device: str = "/dev/video0"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    # Extrinsic: pose of the camera relative to the robot body origin.
    # Heights/offsets are in metres; mount yaw/pitch/roll in degrees.
    mount_x: float = 0.0
    mount_y: float = 0.0
    mount_z: float = 0.30
    mount_yaw_deg: float = 0.0
    mount_pitch_deg: float = 0.0
    mount_roll_deg: float = 0.0

    def body_from_camera(self) -> np.ndarray:
        """4x4 transform mapping camera-frame points into the robot body frame.

        The camera optical frame uses the OpenCV convention (+x right, +y
        down, +z forward).  The robot body frame is +x forward, +y left,
        +z up.  We first rotate the optical frame into the body axis
        convention, then apply the configured mount orientation and offset.
        """
        # optical (x right, y down, z fwd) -> body axes (x fwd, y left, z up)
        R_opt_to_body = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        yaw = math.radians(self.mount_yaw_deg)
        pitch = math.radians(self.mount_pitch_deg)
        roll = math.radians(self.mount_roll_deg)
        # Rz(yaw) Ry(pitch) Rx(roll)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_mount = Rz @ Ry @ Rx
        t = np.array([self.mount_x, self.mount_y, self.mount_z])
        return make_transform(R_mount @ R_opt_to_body, t)


@dataclass
class MarkerDetectorConfig:
    dictionary: str = "DICT_4X4_50"
    marker_length_m: float = 0.15  # printed side length of the markers
    max_reprojection_error_px: float = 4.0
    refine_corners: bool = True


@dataclass
class OdometryConfig:
    enabled: bool = True
    max_features: int = 200
    quality_level: float = 0.01
    min_distance_px: float = 12.0
    # Metres of robot translation per unit of normalised image flow. This is
    # a coarse scale; the EKF's marker fixes correct the residual drift.
    flow_scale_m: float = 0.5
    min_tracked_features: int = 12


@dataclass
class FusionConfig:
    # Process noise (per prediction step) std-devs: x, y [m], theta [rad].
    process_std_xy: float = 0.05
    process_std_theta: float = 0.02
    # Measurement noise for a marker fix: position [m], heading [rad].
    marker_std_xy: float = 0.04
    marker_std_theta: float = 0.03
    # Mahalanobis gate; marker fixes beyond this are rejected as outliers.
    mahalanobis_gate: float = 9.21  # chi-square 0.99 for 2 dof
    initial_std_xy: float = 2.0
    initial_std_theta: float = math.pi


@dataclass
class ImuConfig:
    """How to obtain and integrate IMU telemetry.

    The GoPro logs IMU data in the GPMF metadata track of recorded MP4s, so
    ``source: gpmf`` is for offline / replay runs.  ``source: none`` disables
    IMU fusion (the default for a live USB-webcam run, which carries no
    telemetry).  Supply ``video_path`` for the GPMF source.
    """

    source: str = "none"  # "none" | "gpmf"
    video_path: str = ""  # MP4 to read GPMF telemetry from
    time_offset_s: float = 0.0  # add to IMU timestamps to align with frames
    # Yaw extraction
    use_gravity: bool = True  # project gyro onto the gravity vector
    yaw_axis: int = 2  # gyro axis used when gravity is unavailable
    yaw_sign: float = 1.0  # flip if heading turns the wrong way
    # Bias / stationary handling
    stationary_gyro_thresh: float = 0.04  # rad/s; below this we treat as still
    bias_learn_rate: float = 0.02  # EMA rate for gyro-bias estimation
    # Noise model
    gyro_noise_std: float = 0.01  # rad/s, per-axis white noise
    min_variance: float = 1e-6  # rad^2 floor on a yaw increment
    stationary_variance: float = 1e-8  # rad^2 for a ZUPT-clamped increment
    fallback_variance: float = 1.0  # rad^2 when no IMU samples are available
    max_gap_s: float = 0.5  # ignore integration steps with a larger gap


@dataclass
class OutputConfig:
    publish: bool = True
    protocol: str = "udp"  # "udp" | "tcp"
    host: str = "255.255.255.255"
    port: int = 9870
    rate_hz: float = 30.0
    log_enabled: bool = True
    log_path: str = "pose_log.jsonl"


@dataclass
class SystemConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: MarkerDetectorConfig = field(default_factory=MarkerDetectorConfig)
    odometry: OdometryConfig = field(default_factory=OdometryConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "SystemConfig":
        def sub(name, klass):
            return klass(**(data.get(name) or {}))

        return cls(
            camera=sub("camera", CameraConfig),
            detector=sub("detector", MarkerDetectorConfig),
            odometry=sub("odometry", OdometryConfig),
            imu=sub("imu", ImuConfig),
            fusion=sub("fusion", FusionConfig),
            output=sub("output", OutputConfig),
        )

    @classmethod
    def load(cls, path) -> "SystemConfig":
        return cls.from_dict(_load_yaml(path))


@dataclass
class CameraCalibration:
    """Pinhole intrinsics + distortion for the (reframed) camera view."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    width: int
    height: int
    model: str = "pinhole"  # "pinhole" | "fisheye"

    @classmethod
    def load(cls, path) -> "CameraCalibration":
        data = _load_yaml(path)
        K = np.array(data["camera_matrix"], dtype=float).reshape(3, 3)
        dist = np.array(data["dist_coeffs"], dtype=float).reshape(-1)
        return cls(
            camera_matrix=K,
            dist_coeffs=dist,
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            model=str(data.get("model", "pinhole")),
        )

    def save(self, path) -> None:
        data = {
            "model": self.model,
            "width": int(self.width),
            "height": int(self.height),
            "camera_matrix": self.camera_matrix.reshape(3, 3).tolist(),
            "dist_coeffs": self.dist_coeffs.reshape(-1).tolist(),
        }
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)


@dataclass
class MarkerMap:
    """World poses of all known fiducial markers.

    Each marker is stored as its 4x4 world<-marker transform.  The marker's
    local frame follows OpenCV's ``estimatePoseSingleMarkers`` convention:
    origin at the marker centre, +x to the right, +y up, +z out of the
    marker face toward the viewer.
    """

    poses: Dict[int, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.poses)

    def get(self, marker_id: int) -> Optional[np.ndarray]:
        return self.poses.get(int(marker_id))

    @classmethod
    def from_dict(cls, data: dict) -> "MarkerMap":
        poses: Dict[int, np.ndarray] = {}
        for entry in data.get("markers", []):
            marker_id = int(entry["id"])
            pos = np.array(entry["position"], dtype=float).reshape(3)
            if "rotation_matrix" in entry:
                R = np.array(entry["rotation_matrix"], dtype=float).reshape(3, 3)
            else:
                # Convenience: a wall marker facing into the room described by
                # the yaw of its outward normal about the world +z axis.
                yaw = math.radians(float(entry.get("yaw_deg", 0.0)))
                R = rot_z(yaw)
            poses[marker_id] = make_transform(R, pos)
        return cls(poses=poses)

    @classmethod
    def load(cls, path) -> "MarkerMap":
        return cls.from_dict(_load_yaml(path))

    def world_pose_of(self, marker_id: int) -> Optional[Pose2D]:
        T = self.get(marker_id)
        if T is None:
            return None
        return Pose2D.from_matrix(T)
