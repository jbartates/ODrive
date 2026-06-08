"""Sparse optical-flow visual odometry.

Between two consecutive frames we track Shi-Tomasi corners with the
pyramidal Lucas-Kanade tracker and fit a partial-affine (rotation +
translation + scale) transform to the matched points.  From that transform
we read off:

* a yaw increment (the rotation component), and
* a body-frame translation increment (the image translation mapped to metres
  through ``flow_scale_m``).

This is a *relative* motion source only.  It drifts over time, which is
exactly what the marker fixes correct inside the EKF.  The estimator is
written so that when too few features survive the match it reports zero
motion with a large covariance rather than injecting garbage.

Assumption: the dominant tracked surface (e.g. the ceiling for an upward
GoPro view, or the floor for a downward one) is roughly parallel to the
robot's plane of motion, so image rotation maps to robot yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import OdometryConfig
from ..geometry import Pose2D


@dataclass
class OdometryEstimate:
    delta: Pose2D  # body-frame increment (forward, left, yaw)
    cov: np.ndarray  # 3x3 covariance of the increment
    num_tracked: int  # features successfully tracked
    valid: bool  # False when the estimate is unreliable


class VisualOdometry:
    def __init__(self, config: OdometryConfig):
        self.config = config
        self._prev_gray = None
        self._prev_points = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_points = None

    def _detect_features(self, gray):
        import cv2

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_features,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance_px,
        )

    def _unreliable(self, num_tracked: int) -> OdometryEstimate:
        # Large covariance => the EKF will essentially ignore this step.
        big = np.diag([1.0, 1.0, 1.0])
        return OdometryEstimate(Pose2D(), big, num_tracked, valid=False)

    def process(self, image: np.ndarray) -> OdometryEstimate:
        """Consume the next frame and return the motion since the previous one."""
        import cv2

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        )

        if self._prev_gray is None or self._prev_points is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            return self._unreliable(0)

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None
        )

        if next_points is None or status is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            return self._unreliable(0)

        status = status.reshape(-1).astype(bool)
        good_prev = self._prev_points.reshape(-1, 2)[status]
        good_next = next_points.reshape(-1, 2)[status]
        num_tracked = int(good_prev.shape[0])

        # Always re-seed features for the next iteration so we don't bleed out.
        self._prev_gray = gray
        self._prev_points = self._detect_features(gray)

        if num_tracked < self.config.min_tracked_features:
            return self._unreliable(num_tracked)

        transform, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_next, method=cv2.RANSAC
        )
        if transform is None:
            return self._unreliable(num_tracked)

        estimate = self._decompose(transform, num_tracked, inliers)
        return estimate

    def _decompose(self, transform, num_tracked, inliers) -> OdometryEstimate:
        """Convert a 2x3 partial-affine matrix into a body-frame increment."""
        a, b = transform[0, 0], transform[0, 1]
        tx, ty = transform[0, 2], transform[1, 2]

        # Image rotation angle (CCW positive in image coords).
        image_dtheta = math.atan2(b, a)

        scale = self.config.flow_scale_m
        # Normalise image translation by frame size isn't available here, so
        # the caller-tuned ``flow_scale_m`` already folds in the px->m factor.
        # Image +x is to the right and +y is down; map to the robot body
        # frame where +x is forward and +y is left.  We treat upward image
        # motion of features (scene moving up) as the robot moving forward.
        forward = -ty * scale
        left = -tx * scale
        # A CCW image rotation of the *scene* corresponds to a CW robot yaw.
        yaw = -image_dtheta

        n_inliers = int(np.sum(inliers)) if inliers is not None else num_tracked
        confidence = max(n_inliers, 1)
        # More inliers => tighter covariance.
        sigma_xy = scale * 0.5 / math.sqrt(confidence)
        sigma_theta = 0.05 / math.sqrt(confidence)
        cov = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_theta ** 2])

        return OdometryEstimate(
            delta=Pose2D(forward, left, yaw),
            cov=cov,
            num_tracked=n_inliers,
            valid=True,
        )
