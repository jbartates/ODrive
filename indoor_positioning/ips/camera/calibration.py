"""Lens un-distortion / dewarping helpers.

The marker detector and PnP solve assume a pinhole camera.  These helpers
turn raw frames into rectilinear frames:

* ``pinhole``  -- standard radial/tangential undistort (Brown-Conrady).
* ``fisheye``  -- OpenCV fisheye model, suitable for a single GoPro lens.

For the Max 2's full 360 (equirectangular) feed, prefer configuring the
GoPro webcam to output a reframed *linear* view; that yields near-pinhole
frames that the ``pinhole`` path handles directly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import CameraCalibration


class Undistorter:
    """Pre-computes remap tables so per-frame undistort is cheap on a Pi."""

    def __init__(self, calib: CameraCalibration, alpha: float = 0.0):
        self.calib = calib
        self.alpha = alpha
        self._map1 = None
        self._map2 = None
        self._new_camera_matrix = calib.camera_matrix.copy()

    def _ensure_maps(self, size) -> None:
        if self._map1 is not None:
            return
        import cv2

        w, h = size
        K = self.calib.camera_matrix
        D = self.calib.dist_coeffs
        if self.calib.model == "fisheye":
            new_K = K.copy()
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D.reshape(4, 1), np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            self._new_camera_matrix = new_K
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                K, D, (w, h), self.alpha, (w, h)
            )
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            self._new_camera_matrix = new_K

    @property
    def new_camera_matrix(self) -> np.ndarray:
        return self._new_camera_matrix

    def undistort(self, image: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        self._ensure_maps((w, h))
        return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR)
