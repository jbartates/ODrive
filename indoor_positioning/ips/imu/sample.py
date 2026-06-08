"""A single timestamped IMU reading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ImuSample:
    """One inertial sample, expressed in the IMU/camera sensor frame.

    * ``t``     -- timestamp in seconds (same clock as the camera frames).
    * ``gyro``  -- angular velocity (rad/s), length-3.
    * ``accel`` -- specific force (m/s^2), length-3, optional.
    * ``grav``  -- unit gravity direction in the sensor frame, optional.
                   GoPro provides this directly as the ``GRAV`` stream; when
                   absent it can be approximated by low-pass filtering accel.
    """

    t: float
    gyro: np.ndarray
    accel: Optional[np.ndarray] = None
    grav: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.gyro = np.asarray(self.gyro, dtype=float).reshape(3)
        if self.accel is not None:
            self.accel = np.asarray(self.accel, dtype=float).reshape(3)
        if self.grav is not None:
            self.grav = np.asarray(self.grav, dtype=float).reshape(3)
