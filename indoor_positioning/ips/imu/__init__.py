"""IMU telemetry: parse GoPro GPMF metadata and feed it into the fusion.

The GoPro Max 2 logs inertial telemetry in the **GPMF** metadata track of the
MP4 it records to the SD card (streams ``GYRO``, ``ACCL``, ``GRAV``, ...).
This subpackage extracts those streams and turns them into a clean yaw-rate
signal that the EKF uses for its prediction step -- far more accurate than the
visual-odometry heading estimate.

Important: GPMF telemetry is **not** present in the live USB-webcam (UVC)
stream.  Use it for offline processing of recorded clips, or supply samples
from any other IMU through :class:`ListImuSource`.
"""

from .sample import ImuSample
from .source import GpmfImuSource, ImuSource, ListImuSource
from .preintegrator import ImuPreintegrator, YawIncrement

__all__ = [
    "ImuSample",
    "ImuSource",
    "GpmfImuSource",
    "ListImuSource",
    "ImuPreintegrator",
    "YawIncrement",
]
