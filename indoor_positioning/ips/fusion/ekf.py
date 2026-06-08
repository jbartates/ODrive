"""A small extended Kalman filter that fuses odometry and marker fixes.

State vector (world frame):  ``[x, y, theta]``

* **predict(delta)** consumes a body-frame motion increment supplied by the
  visual odometry (forward, left, yaw).  The motion is a control input, so
  its uncertainty enters the filter as additional process noise.
* **update(measurement)** consumes an absolute pose fix derived from one or
  more ArUco markers.  Heading innovation is wrapped to ``(-pi, pi]`` and an
  optional Mahalanobis gate rejects outliers.

The implementation is deliberately pure ``numpy`` so it is fully unit
tested without any camera hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import FusionConfig
from ..geometry import Pose2D, wrap_angle


@dataclass
class EKFState:
    mean: np.ndarray  # shape (3,)
    cov: np.ndarray  # shape (3, 3)

    def pose(self) -> Pose2D:
        return Pose2D.from_array(self.mean)


class PoseEKF:
    """Extended Kalman filter over a planar robot pose."""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self._initialised = False
        self.mean = np.zeros(3)
        self.cov = np.diag(
            [
                self.config.initial_std_xy ** 2,
                self.config.initial_std_xy ** 2,
                self.config.initial_std_theta ** 2,
            ]
        )

    @property
    def initialised(self) -> bool:
        return self._initialised

    def reset(self, pose: Pose2D, std_xy: float, std_theta: float) -> None:
        self.mean = pose.as_array()
        self.cov = np.diag([std_xy ** 2, std_xy ** 2, std_theta ** 2])
        self._initialised = True

    def state(self) -> EKFState:
        return EKFState(self.mean.copy(), self.cov.copy())

    def pose(self) -> Pose2D:
        return Pose2D.from_array(self.mean)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, delta: Pose2D, control_cov: Optional[np.ndarray] = None) -> None:
        """Propagate the state using a body-frame motion increment.

        ``delta`` = (forward, left, yaw) measured in the *current* body
        frame.  ``control_cov`` is an optional 3x3 covariance describing how
        noisy that increment is; when omitted the static process noise from
        the config is used.
        """
        if not self._initialised:
            # Without an absolute reference, integrating odometry alone is
            # meaningless, but we still propagate so relative motion is kept.
            self._initialised = True

        theta = self.mean[2]
        c, s = math.cos(theta), math.sin(theta)
        dx, dy, dth = float(delta.x), float(delta.y), float(delta.theta)

        # Motion model g(x, u)
        self.mean = np.array(
            [
                self.mean[0] + c * dx - s * dy,
                self.mean[1] + s * dx + c * dy,
                wrap_angle(theta + dth),
            ]
        )

        # Jacobian of g wrt state (G)
        G = np.array(
            [
                [1.0, 0.0, -s * dx - c * dy],
                [0.0, 1.0, c * dx - s * dy],
                [0.0, 0.0, 1.0],
            ]
        )

        # Process noise
        if control_cov is None:
            Q = np.diag(
                [
                    self.config.process_std_xy ** 2,
                    self.config.process_std_xy ** 2,
                    self.config.process_std_theta ** 2,
                ]
            )
        else:
            # Map the body-frame control covariance into the world frame.
            V = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            Q = V @ np.asarray(control_cov, dtype=float) @ V.T

        self.cov = G @ self.cov @ G.T + Q

    # ------------------------------------------------------------------
    # Measurement update
    # ------------------------------------------------------------------
    def update(
        self,
        measurement: Pose2D,
        meas_cov: Optional[np.ndarray] = None,
        gate: bool = True,
    ) -> bool:
        """Fuse an absolute pose measurement (e.g. from markers).

        Returns ``True`` if the measurement was accepted, ``False`` if it was
        rejected by the Mahalanobis gate.  If the filter has never been
        initialised the first measurement is used to initialise it directly.
        """
        z = measurement.as_array()

        if meas_cov is None:
            R = np.diag(
                [
                    self.config.marker_std_xy ** 2,
                    self.config.marker_std_xy ** 2,
                    self.config.marker_std_theta ** 2,
                ]
            )
        else:
            R = np.asarray(meas_cov, dtype=float)

        if not self._initialised:
            self.mean = z.copy()
            self.cov = R.copy()
            self._initialised = True
            return True

        # H is identity: we directly measure the state.
        H = np.eye(3)
        innovation = z - self.mean
        innovation[2] = wrap_angle(innovation[2])

        S = H @ self.cov @ H.T + R

        if gate:
            try:
                maha = float(innovation.T @ np.linalg.solve(S, innovation))
            except np.linalg.LinAlgError:  # pragma: no cover - singular S
                maha = float("inf")
            if maha > self.config.mahalanobis_gate:
                return False

        K = self.cov @ H.T @ np.linalg.inv(S)
        self.mean = self.mean + K @ innovation
        self.mean[2] = wrap_angle(self.mean[2])
        I = np.eye(3)
        # Joseph form keeps the covariance symmetric positive-definite.
        self.cov = (I - K @ H) @ self.cov @ (I - K @ H).T + K @ R @ K.T
        return True
