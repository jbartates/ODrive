"""Replay a video file or a folder of images as a camera source.

Invaluable for offline testing and for tuning the pipeline on recorded
footage without the robot or GoPro present.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

from .base import CameraSource, Frame


class FileCamera(CameraSource):
    def __init__(self, config):
        super().__init__(config)
        self._cap = None
        self._image_paths: List[str] = []
        self._is_video = False

    def open(self) -> None:
        import cv2

        path = self.config.device
        if os.path.isdir(path):
            patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            paths: List[str] = []
            for pat in patterns:
                paths.extend(glob.glob(os.path.join(path, pat)))
            self._image_paths = sorted(paths)
            self._is_video = False
            if not self._image_paths:
                raise RuntimeError(f"No images found in {path!r}")
        else:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file {path!r}")
            self._cap = cap
            self._is_video = True

    def read(self) -> Optional[Frame]:
        import cv2

        if self._is_video:
            # Presentation time BEFORE grabbing this frame == its start time,
            # so it shares the "seconds from start" clock that GPMF telemetry
            # uses, letting the IMU and video be aligned in offline replay.
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, image = self._cap.read()
            if not ok or image is None:
                return None
            idx = self._next_index()
            timestamp = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else float(idx)
            return Frame(image=image, timestamp=timestamp, index=idx)

        if self._index >= len(self._image_paths):
            return None
        image = cv2.imread(self._image_paths[self._index])
        idx = self._next_index()
        if image is None:  # pragma: no cover - corrupt image
            return None
        fps = self.config.fps or 30
        return Frame(image=image, timestamp=idx / float(fps), index=idx)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
