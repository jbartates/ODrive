"""Turn raw IMU samples into a yaw increment for the EKF prediction step.

For a robot driving on a flat floor, the single most valuable thing the IMU
provides is an accurate **heading rate**.  We obtain a mount-agnostic yaw
rate by projecting the gyro vector onto the vertical axis defined by gravity
(the GoPro ``GRAV`` stream, or low-pass accelerometer as a fallback):

    yaw_rate = sign * (omega . up_hat)

This works regardless of how the camera is tilted/oriented on the robot, as
long as the gyro and gravity readings share a sensor frame (they do in GPMF).

The integrator also:

* estimates and removes the gyro bias during detected stationary periods, and
* applies a rotational zero-velocity update (ZUPT) when stationary, which
  stops heading from creeping while the robot is parked.

It bridges state across calls, so integrating frame-by-frame yields the same
result as integrating the whole sequence at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import ImuConfig
from .sample import ImuSample


@dataclass
class YawIncrement:
    dtheta: float  # integrated heading change over the interval (rad)
    variance: float  # uncertainty of dtheta (rad^2)
    dt: float  # time span integrated (s)
    n: int  # number of samples used
    stationary: bool  # whether the robot was detected as stationary
    valid: bool  # False when no usable IMU data was available


class ImuPreintegrator:
    def __init__(self, config: Optional[ImuConfig] = None):
        self.config = config or ImuConfig()
        self.gyro_bias = np.zeros(3)
        self._prev: Optional[ImuSample] = None

    def reset(self) -> None:
        self._prev = None
        self.gyro_bias = np.zeros(3)

    # ------------------------------------------------------------------
    def _vertical_axis(self, sample: ImuSample) -> Optional[np.ndarray]:
        if not self.config.use_gravity:
            return None
        grav = sample.grav
        if grav is None and sample.accel is not None:
            # At low acceleration, specific force ~ reaction to gravity.
            grav = sample.accel
        if grav is None:
            return None
        norm = np.linalg.norm(grav)
        if norm < 1e-6:
            return None
        return grav / norm

    def _yaw_rate(self, sample: ImuSample) -> float:
        omega = sample.gyro - self.gyro_bias
        axis = self._vertical_axis(sample)
        if axis is not None:
            rate = float(np.dot(omega, axis))
        else:
            rate = float(omega[self.config.yaw_axis])
        return self.config.yaw_sign * rate

    def _maybe_update_bias(self, samples: List[ImuSample]) -> bool:
        mags = [float(np.linalg.norm(s.gyro)) for s in samples]
        if not mags or max(mags) > self.config.stationary_gyro_thresh:
            return False
        mean_gyro = np.mean([s.gyro for s in samples], axis=0)
        a = self.config.bias_learn_rate
        self.gyro_bias = (1.0 - a) * self.gyro_bias + a * mean_gyro
        return True

    # ------------------------------------------------------------------
    def integrate(self, samples: List[ImuSample]) -> YawIncrement:
        """Integrate a batch of samples into a single yaw increment."""
        if not samples:
            return YawIncrement(0.0, self.config.fallback_variance, 0.0, 0, False, False)

        stationary = self._maybe_update_bias(samples)

        dtheta = 0.0
        total_dt = 0.0
        prev = self._prev
        prev_rate = self._yaw_rate(prev) if prev is not None else None

        for sample in samples:
            rate = self._yaw_rate(sample)
            if prev is not None:
                dt = sample.t - prev.t
                if 0.0 < dt < self.config.max_gap_s:
                    # trapezoidal integration of the yaw rate
                    dtheta += 0.5 * (prev_rate + rate) * dt
                    total_dt += dt
            prev = sample
            prev_rate = rate

        self._prev = prev

        if stationary:
            dtheta = 0.0
            variance = self.config.stationary_variance
        else:
            noise = self.config.gyro_noise_std
            variance = noise * noise * max(total_dt, 1e-3)
            variance = max(variance, self.config.min_variance)

        return YawIncrement(
            dtheta=dtheta,
            variance=variance,
            dt=total_dt,
            n=len(samples),
            stationary=stationary,
            valid=total_dt > 0.0,
        )
