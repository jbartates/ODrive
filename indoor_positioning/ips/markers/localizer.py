"""Turn camera-frame marker observations into world-frame robot fixes.

Given a marker observation (marker pose in the camera frame) and the
marker's known pose in the world, we chain the transforms:

    T_world_body = T_world_marker @ inv(T_camera_marker) @ inv(T_body_camera)

and read the planar pose off the result.  Each visible marker yields an
independent :class:`PoseFix`; the caller (the EKF) fuses them sequentially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..config import CameraConfig, FusionConfig, MarkerMap
from ..geometry import Pose2D, invert_transform
from .aruco_detector import MarkerObservation


def world_body_from_marker(
    T_world_marker: np.ndarray,
    T_camera_marker: np.ndarray,
    T_body_camera: np.ndarray,
) -> np.ndarray:
    """Chain the transforms to get the robot body pose in the world frame.

    Pure ``numpy`` (no OpenCV) so the localisation maths is unit testable::

        T_world_body = T_world_marker @ inv(T_camera_marker) @ inv(T_body_camera)
    """
    T_world_camera = T_world_marker @ invert_transform(T_camera_marker)
    return T_world_camera @ invert_transform(T_body_camera)


@dataclass
class PoseFix:
    """An absolute world-frame pose estimate derived from one marker."""

    marker_id: int
    pose: Pose2D
    range_m: float  # distance from camera to the marker
    reprojection_error: float
    cov: np.ndarray  # 3x3 measurement covariance (x, y, theta)


class MarkerLocalizer:
    def __init__(
        self,
        marker_map: MarkerMap,
        camera_config: CameraConfig,
        fusion_config: FusionConfig,
    ):
        self.marker_map = marker_map
        self.camera_config = camera_config
        self.fusion_config = fusion_config
        self._body_from_camera = camera_config.body_from_camera()

    def fixes_from(self, observations: List[MarkerObservation]) -> List[PoseFix]:
        fixes: List[PoseFix] = []
        for obs in observations:
            T_world_marker = self.marker_map.get(obs.marker_id)
            if T_world_marker is None:
                # Marker seen but not in the map: ignore (could be a foreign tag).
                continue
            T_camera_marker = obs.transform_camera_marker()
            T_world_body = world_body_from_marker(
                T_world_marker, T_camera_marker, self._body_from_camera
            )
            pose = Pose2D.from_matrix(T_world_body)
            range_m = float(np.linalg.norm(obs.tvec))
            fixes.append(
                PoseFix(
                    marker_id=obs.marker_id,
                    pose=pose,
                    range_m=range_m,
                    reprojection_error=obs.reprojection_error,
                    cov=self._covariance(range_m, obs.reprojection_error),
                )
            )
        return fixes

    def _covariance(self, range_m: float, reproj_err: float) -> np.ndarray:
        """Scale the nominal marker noise by range and reprojection error.

        Pose error from a square fiducial grows roughly with distance, so we
        inflate the base std-devs by ``(1 + range)`` and by the reprojection
        residual.  This makes far / poorly-fit markers contribute less.
        """
        base_xy = self.fusion_config.marker_std_xy
        base_theta = self.fusion_config.marker_std_theta
        err_factor = 1.0 + reproj_err
        std_xy = base_xy * (1.0 + range_m) * err_factor
        std_theta = base_theta * (1.0 + 0.5 * range_m) * err_factor
        return np.diag([std_xy ** 2, std_xy ** 2, std_theta ** 2])
