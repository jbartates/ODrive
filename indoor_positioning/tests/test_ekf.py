import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.config import FusionConfig  # noqa: E402
from ips.fusion import PoseEKF  # noqa: E402
from ips.geometry import Pose2D, wrap_angle  # noqa: E402


def test_first_measurement_initialises_filter():
    ekf = PoseEKF()
    assert not ekf.initialised
    accepted = ekf.update(Pose2D(1.0, 2.0, 0.5))
    assert accepted
    assert ekf.initialised
    pose = ekf.pose()
    assert pose.x == pytest.approx(1.0)
    assert pose.y == pytest.approx(2.0)
    assert pose.theta == pytest.approx(0.5)


def test_prediction_moves_state_forward():
    ekf = PoseEKF()
    ekf.reset(Pose2D(0.0, 0.0, 0.0), std_xy=0.1, std_theta=0.1)
    ekf.predict(Pose2D(1.0, 0.0, 0.0))
    pose = ekf.pose()
    assert pose.x == pytest.approx(1.0, abs=1e-9)
    assert pose.y == pytest.approx(0.0, abs=1e-9)


def test_prediction_grows_covariance():
    ekf = PoseEKF()
    ekf.reset(Pose2D(), std_xy=0.1, std_theta=0.1)
    before = np.trace(ekf.cov)
    ekf.predict(Pose2D(0.5, 0.0, 0.1))
    assert np.trace(ekf.cov) > before


def test_update_shrinks_covariance():
    ekf = PoseEKF()
    ekf.reset(Pose2D(0.0, 0.0, 0.0), std_xy=1.0, std_theta=1.0)
    before = np.trace(ekf.cov)
    ekf.update(Pose2D(0.2, 0.1, 0.05))
    assert np.trace(ekf.cov) < before


def test_update_pulls_estimate_toward_measurement():
    ekf = PoseEKF()
    ekf.reset(Pose2D(0.0, 0.0, 0.0), std_xy=1.0, std_theta=1.0)
    ekf.update(Pose2D(2.0, 0.0, 0.0), meas_cov=np.diag([0.01, 0.01, 0.01]))
    # With a confident measurement the estimate should move most of the way.
    assert ekf.pose().x > 1.5


def test_angle_innovation_wraps():
    ekf = PoseEKF()
    ekf.reset(Pose2D(0.0, 0.0, math.radians(170)), std_xy=0.1, std_theta=1.0)
    # Measurement just across the +/-180 boundary.
    ekf.update(
        Pose2D(0.0, 0.0, math.radians(-170)),
        meas_cov=np.diag([0.01, 0.01, 0.01]),
    )
    # Result should sit near +/-180, NOT swing all the way back through 0.
    theta = ekf.pose().theta
    assert abs(wrap_angle(theta - math.radians(180))) < math.radians(15)


def test_mahalanobis_gate_rejects_outlier():
    cfg = FusionConfig(mahalanobis_gate=9.21)
    ekf = PoseEKF(cfg)
    ekf.reset(Pose2D(0.0, 0.0, 0.0), std_xy=0.05, std_theta=0.05)
    # A wildly inconsistent fix relative to a confident state.
    accepted = ekf.update(
        Pose2D(50.0, 50.0, 0.0), meas_cov=np.diag([0.01, 0.01, 0.01])
    )
    assert accepted is False
    # State should be unchanged.
    assert ekf.pose().x == pytest.approx(0.0, abs=1e-9)


def test_covariance_stays_symmetric():
    ekf = PoseEKF()
    ekf.reset(Pose2D(), std_xy=0.5, std_theta=0.5)
    for _ in range(20):
        ekf.predict(Pose2D(0.1, 0.0, 0.02))
        ekf.update(Pose2D(0.1, 0.0, 0.02))
    assert np.allclose(ekf.cov, ekf.cov.T, atol=1e-9)
    # Positive semidefinite.
    assert np.all(np.linalg.eigvalsh(ekf.cov) > -1e-9)


def test_dead_reckoning_then_correction():
    """Drive a square open-loop, then a marker fix snaps us back."""
    ekf = PoseEKF()
    ekf.reset(Pose2D(0.0, 0.0, 0.0), std_xy=0.05, std_theta=0.05)
    for _ in range(10):
        ekf.predict(Pose2D(0.1, 0.0, 0.0))  # 1 m forward total
    assert ekf.pose().x == pytest.approx(1.0, abs=1e-6)
    # Ground-truth marker says we are actually at x=0.9.
    ekf.update(Pose2D(0.9, 0.0, 0.0), meas_cov=np.diag([0.001, 0.001, 0.001]))
    assert ekf.pose().x == pytest.approx(0.9, abs=0.05)
