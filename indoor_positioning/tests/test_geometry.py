import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.geometry import (  # noqa: E402
    Pose2D,
    angle_diff,
    invert_transform,
    make_transform,
    rot_z,
    wrap_angle,
    yaw_from_matrix,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),
        (3 * math.pi, math.pi),
        (math.pi + 0.1, -math.pi + 0.1),
        (-3 * math.pi / 2, math.pi / 2),
    ],
)
def test_wrap_angle(raw, expected):
    assert wrap_angle(raw) == pytest.approx(expected, abs=1e-9)


def test_angle_diff_wraps_shortest_way():
    assert angle_diff(math.radians(170), math.radians(-170)) == pytest.approx(
        math.radians(-20), abs=1e-9
    )


def test_invert_transform_roundtrip():
    T = make_transform(rot_z(0.7), np.array([1.0, -2.0, 0.5]))
    assert np.allclose(T @ invert_transform(T), np.eye(4), atol=1e-12)


def test_yaw_from_matrix():
    assert yaw_from_matrix(rot_z(1.2)) == pytest.approx(1.2, abs=1e-9)


def test_pose_matrix_roundtrip():
    pose = Pose2D(1.5, -0.3, 2.0)
    recovered = Pose2D.from_matrix(pose.to_matrix())
    assert recovered.x == pytest.approx(pose.x)
    assert recovered.y == pytest.approx(pose.y)
    assert recovered.theta == pytest.approx(wrap_angle(pose.theta))


def test_compose_pure_forward():
    pose = Pose2D(0.0, 0.0, math.pi / 2)  # facing +y
    moved = pose.compose(Pose2D(1.0, 0.0, 0.0))  # 1 m forward
    assert moved.x == pytest.approx(0.0, abs=1e-9)
    assert moved.y == pytest.approx(1.0, abs=1e-9)


def test_compose_left_translation():
    pose = Pose2D(0.0, 0.0, 0.0)  # facing +x
    moved = pose.compose(Pose2D(0.0, 1.0, 0.0))  # 1 m to the left
    assert moved.x == pytest.approx(0.0, abs=1e-9)
    assert moved.y == pytest.approx(1.0, abs=1e-9)


def test_compose_accumulates_rotation():
    pose = Pose2D(0.0, 0.0, 0.0)
    for _ in range(4):
        pose = pose.compose(Pose2D(0.0, 0.0, math.pi / 2))
    assert wrap_angle(pose.theta) == pytest.approx(0.0, abs=1e-9)
