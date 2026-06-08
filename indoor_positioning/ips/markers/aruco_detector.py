"""ArUco marker detection + per-marker pose estimation.

OpenCV's ArUco API changed shape across 4.x releases (the free functions
``cv2.aruco.detectMarkers`` / ``estimatePoseSingleMarkers`` were superseded
by the ``ArucoDetector`` class and, in 4.7+, ``estimatePoseSingleMarkers``
was deprecated in favour of ``solvePnP``).  This module hides those
differences behind one stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import CameraCalibration, MarkerDetectorConfig


@dataclass
class MarkerObservation:
    """A single detected marker and its pose in the camera frame."""

    marker_id: int
    corners: np.ndarray  # shape (4, 2) image coordinates
    rvec: np.ndarray  # Rodrigues rotation, marker -> camera
    tvec: np.ndarray  # translation, marker -> camera (metres)
    reprojection_error: float

    def transform_camera_marker(self) -> np.ndarray:
        """4x4 transform mapping marker-frame points into the camera frame."""
        import cv2

        T = np.eye(4)
        R, _ = cv2.Rodrigues(self.rvec.reshape(3, 1))
        T[:3, :3] = R
        T[:3, 3] = self.tvec.reshape(3)
        return T


def _resolve_dictionary(name: str):
    import cv2

    attr = getattr(cv2.aruco, name, None)
    if attr is None:
        raise ValueError(f"Unknown ArUco dictionary: {name!r}")
    # Newer OpenCV: getPredefinedDictionary; older: Dictionary_get
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(attr)
    return cv2.aruco.Dictionary_get(attr)  # pragma: no cover - old OpenCV


class ArucoDetector:
    """Detect ArUco markers and estimate each marker's pose in metres."""

    def __init__(
        self,
        config: MarkerDetectorConfig,
        calibration: CameraCalibration,
    ):
        self.config = config
        self.calibration = calibration
        self._dictionary = None
        self._detector = None

    # -- lazy OpenCV setup ---------------------------------------------
    def _ensure_detector(self) -> None:
        if self._dictionary is not None:
            return
        import cv2

        self._dictionary = _resolve_dictionary(self.config.dictionary)
        if hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            if self.config.refine_corners:
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
        else:  # pragma: no cover - legacy OpenCV
            self._detector = None

    def _detect_corners(self, gray):
        import cv2

        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:  # pragma: no cover - legacy OpenCV
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=params
            )
        return corners, ids

    # -- main entry point ----------------------------------------------
    def detect(self, image: np.ndarray) -> List[MarkerObservation]:
        """Return all markers visible in ``image`` with camera-frame poses."""
        import cv2

        self._ensure_detector()
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        corners, ids = self._detect_corners(gray)
        if ids is None or len(ids) == 0:
            return []

        K = self.calibration.camera_matrix
        D = self.calibration.dist_coeffs
        half = self.config.marker_length_m / 2.0
        # Marker corner layout used by OpenCV: order is top-left, top-right,
        # bottom-right, bottom-left in the marker's own frame (+x right,
        # +y up, z out of the face).
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

        observations: List[MarkerObservation] = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            image_points = marker_corners.reshape(4, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue
            err = self._reprojection_error(
                object_points, image_points, rvec, tvec, K, D
            )
            if err > self.config.max_reprojection_error_px:
                continue
            observations.append(
                MarkerObservation(
                    marker_id=int(marker_id),
                    corners=image_points,
                    rvec=rvec.reshape(3),
                    tvec=tvec.reshape(3),
                    reprojection_error=float(err),
                )
            )
        return observations

    @staticmethod
    def _reprojection_error(object_points, image_points, rvec, tvec, K, D) -> float:
        import cv2

        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
        projected = projected.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
