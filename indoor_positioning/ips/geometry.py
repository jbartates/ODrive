"""Small, dependency-light geometry helpers.

Everything in here is pure ``numpy`` so it can be unit tested on a laptop
without a camera or OpenCV.  We use two conventions consistently:

* **World / map frame**: a right-handed frame fixed to the building.  The
  robot drives on the ``z = 0`` floor plane.  A robot pose is the triple
  ``(x, y, theta)`` where ``theta`` is the heading (yaw) measured CCW from
  the world +x axis.
* **3D poses** are represented as 4x4 homogeneous transforms ``T`` that map
  a point expressed in the child frame into the parent frame:
  ``p_parent = T @ p_child``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_angle(theta: float) -> float:
    """Wrap an angle in radians into the half-open interval ``(-pi, pi]``."""
    wrapped = (theta + math.pi) % TWO_PI - math.pi
    # ``% TWO_PI`` maps -pi to +pi which we keep; guard the -pi edge.
    if wrapped <= -math.pi:
        wrapped += TWO_PI
    return wrapped


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference ``a - b`` wrapped to ``(-pi, pi]``."""
    return wrap_angle(a - b)


def rot_z(theta: float) -> np.ndarray:
    """3x3 rotation matrix about the world +z axis."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 R and a length-3 t."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=float)
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform efficiently (no general inverse)."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def yaw_from_matrix(R: np.ndarray) -> float:
    """Extract the yaw (rotation about +z) from a 3x3 rotation matrix."""
    return math.atan2(R[1, 0], R[0, 0])


@dataclass
class Pose2D:
    """A planar pose: position ``(x, y)`` in metres and heading in radians."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, wrap_angle(self.theta)])

    @classmethod
    def from_array(cls, arr) -> "Pose2D":
        arr = np.asarray(arr, dtype=float).reshape(3)
        return cls(float(arr[0]), float(arr[1]), wrap_angle(float(arr[2])))

    def to_matrix(self) -> np.ndarray:
        """Return the 4x4 world<-robot transform for this planar pose."""
        return make_transform(rot_z(self.theta), np.array([self.x, self.y, 0.0]))

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "Pose2D":
        return cls(float(T[0, 3]), float(T[1, 3]), yaw_from_matrix(T[:3, :3]))

    def compose(self, delta: "Pose2D") -> "Pose2D":
        """Apply a body-frame motion ``delta`` to this pose.

        ``delta`` is interpreted in the robot's current body frame: a forward
        translation of ``delta.x``, a left translation of ``delta.y`` and a
        rotation of ``delta.theta``.
        """
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose2D(
            x=self.x + c * delta.x - s * delta.y,
            y=self.y + s * delta.x + c * delta.y,
            theta=wrap_angle(self.theta + delta.theta),
        )
