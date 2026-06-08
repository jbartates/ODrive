"""Tests for IMU preintegration, source streaming, and EKF integration."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.config import ImuConfig  # noqa: E402
from ips.fusion import PoseEKF  # noqa: E402
from ips.geometry import Pose2D, wrap_angle  # noqa: E402
from ips.imu import ImuPreintegrator, ImuSample, ListImuSource  # noqa: E402


def _spin_samples(rate, duration=1.0, dt=0.1, grav=(0.0, 0.0, -9.8), t0=0.0):
    """Constant yaw-rate samples about the vertical defined by ``grav``."""
    samples = []
    t = t0
    n = int(round(duration / dt)) + 1
    for _ in range(n):
        samples.append(
            ImuSample(t=t, gyro=np.array([0.0, 0.0, rate]), grav=np.array(grav))
        )
        t += dt
    return samples


def test_empty_batch_is_invalid():
    pre = ImuPreintegrator()
    inc = pre.integrate([])
    assert inc.valid is False
    assert inc.variance == pytest.approx(ImuConfig().fallback_variance)


def test_constant_rate_integrates_to_angle():
    pre = ImuPreintegrator(ImuConfig(yaw_sign=1.0))
    inc = pre.integrate(_spin_samples(rate=0.5, duration=1.0, dt=0.1))
    # 11 samples => 10 intervals of 0.1 s at 0.5 rad/s = 0.5 rad total.
    assert abs(inc.dtheta) == pytest.approx(0.5, abs=1e-6)
    assert inc.valid
    assert not inc.stationary


def test_yaw_sign_flips_direction():
    pos = ImuPreintegrator(ImuConfig(yaw_sign=1.0)).integrate(_spin_samples(0.5))
    neg = ImuPreintegrator(ImuConfig(yaw_sign=-1.0)).integrate(_spin_samples(0.5))
    assert pos.dtheta == pytest.approx(-neg.dtheta, abs=1e-9)


def test_gravity_projection_is_tilt_invariant():
    """Rotating gyro and gravity together must not change the yaw estimate."""
    base = _spin_samples(0.5)
    flat = ImuPreintegrator().integrate(base)

    # Rotate every sample's gyro and gravity by the same rotation.
    theta = math.radians(35)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # tilt about x
    tilted = [
        ImuSample(t=smp.t, gyro=R @ smp.gyro, grav=R @ smp.grav) for smp in base
    ]
    out = ImuPreintegrator().integrate(tilted)
    assert out.dtheta == pytest.approx(flat.dtheta, abs=1e-9)


def test_stationary_triggers_zupt_and_learns_bias():
    cfg = ImuConfig(stationary_gyro_thresh=0.05, bias_learn_rate=0.5)
    pre = ImuPreintegrator(cfg)
    # A small constant offset on a "still" robot -> bias, not real rotation.
    offset = np.array([0.0, 0.0, 0.01])
    samples = [
        ImuSample(t=0.1 * i, gyro=offset, grav=np.array([0.0, 0.0, -9.8]))
        for i in range(10)
    ]
    inc = pre.integrate(samples)
    assert inc.stationary is True
    assert inc.dtheta == pytest.approx(0.0)  # ZUPT clamps rotation
    # Bias estimate should move toward the observed offset.
    assert pre.gyro_bias[2] > 0.0


def test_bridging_across_batches_matches_single_batch():
    samples = _spin_samples(0.4, duration=1.0, dt=0.1)
    whole = ImuPreintegrator().integrate(samples)

    pre = ImuPreintegrator()
    first = pre.integrate(samples[:5])
    second = pre.integrate(samples[5:])
    assert first.dtheta + second.dtheta == pytest.approx(whole.dtheta, abs=1e-9)


def test_list_source_streams_in_order():
    samples = [ImuSample(t=float(i), gyro=np.zeros(3)) for i in range(5)]
    src = ListImuSource(samples)
    first = src.until(2.0)
    assert [s.t for s in first] == [0.0, 1.0, 2.0]
    rest = src.until(10.0)
    assert [s.t for s in rest] == [3.0, 4.0]
    assert src.until(10.0) == []  # nothing left


def test_imu_yaw_drives_ekf_prediction():
    """An EKF predict fed the IMU yaw increment should rotate the estimate."""
    ekf = PoseEKF()
    ekf.reset(Pose2D(0.0, 0.0, 0.0), std_xy=0.1, std_theta=0.1)
    pre = ImuPreintegrator(ImuConfig(yaw_sign=-1.0))  # so +gyro_z -> +yaw
    inc = pre.integrate(_spin_samples(rate=1.0, duration=1.0, dt=0.1))
    ekf.predict(Pose2D(0.0, 0.0, inc.dtheta), np.diag([0.01, 0.01, inc.variance]))
    # 10 intervals at 1.0 rad/s = 1.0 rad of heading change.
    assert wrap_angle(ekf.pose().theta) == pytest.approx(1.0, abs=1e-6)
