"""Validate the marker -> world-pose transform chain without OpenCV.

We synthesise a marker observation by *placing* a robot, camera and marker
in the world, computing the marker pose the camera would see, and then
checking that the localiser maths recovers the robot's true pose.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.config import CameraConfig  # noqa: E402
from ips.geometry import Pose2D, invert_transform  # noqa: E402
from ips.markers.localizer import world_body_from_marker  # noqa: E402


def _recover(robot: Pose2D, camera_cfg: CameraConfig, T_world_marker: np.ndarray):
    T_world_body = robot.to_matrix()
    T_body_camera = camera_cfg.body_from_camera()
    T_world_camera = T_world_body @ T_body_camera
    # What the camera actually observes:
    T_camera_marker = invert_transform(T_world_camera) @ T_world_marker
    recovered = world_body_from_marker(
        T_world_marker, T_camera_marker, T_body_camera
    )
    return Pose2D.from_matrix(recovered)


def test_recovers_robot_pose_forward_camera():
    cam = CameraConfig(mount_z=0.3, mount_pitch_deg=0.0)  # looks forward
    marker = np.eye(4)
    marker[:3, 3] = [5.0, 0.0, 0.3]
    robot = Pose2D(1.0, -0.5, math.radians(20))
    out = _recover(robot, cam, marker)
    assert out.x == pytest.approx(robot.x, abs=1e-6)
    assert out.y == pytest.approx(robot.y, abs=1e-6)
    assert out.theta == pytest.approx(robot.theta, abs=1e-6)


def test_recovers_robot_pose_ceiling_camera():
    # Camera pitched up 90 deg to stare at the ceiling.
    cam = CameraConfig(mount_z=0.3, mount_pitch_deg=90.0)
    marker = np.eye(4)
    marker[:3, 3] = [2.0, 1.0, 2.6]
    robot = Pose2D(2.0, 1.0, math.radians(-40))
    out = _recover(robot, cam, marker)
    assert out.x == pytest.approx(robot.x, abs=1e-6)
    assert out.y == pytest.approx(robot.y, abs=1e-6)
    assert out.theta == pytest.approx(robot.theta, abs=1e-6)


def test_offset_mounted_camera():
    cam = CameraConfig(mount_x=0.2, mount_y=0.1, mount_z=0.5, mount_yaw_deg=30.0)
    marker = np.eye(4)
    marker[:3, 3] = [3.0, 3.0, 1.5]
    robot = Pose2D(0.4, 0.7, math.radians(15))
    out = _recover(robot, cam, marker)
    assert out.x == pytest.approx(robot.x, abs=1e-6)
    assert out.y == pytest.approx(robot.y, abs=1e-6)
    assert out.theta == pytest.approx(robot.theta, abs=1e-6)


def test_body_from_camera_is_rigid():
    cam = CameraConfig(mount_pitch_deg=37.0, mount_yaw_deg=12.0, mount_roll_deg=-5.0)
    T = cam.body_from_camera()
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
