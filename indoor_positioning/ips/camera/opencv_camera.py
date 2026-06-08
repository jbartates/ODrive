"""Generic V4L2 / OpenCV ``VideoCapture`` camera backend.

This is the workhorse used by the GoPro USB backend once the camera is
presenting itself as a ``/dev/videoN`` device.  It is also useful for any
ordinary USB webcam during bring-up and testing.
"""

from __future__ import annotations

from typing import Optional

from .base import CameraSource, Frame


class OpenCVCamera(CameraSource):
    def __init__(self, config):
        super().__init__(config)
        self._cap = None

    def open(self) -> None:
        import cv2  # deferred: not needed for the pure-python test suite

        device = self.config.device
        # Accept either an integer index ("0") or a device path.
        try:
            device_arg = int(device)
        except (TypeError, ValueError):
            device_arg = device

        cap = cv2.VideoCapture(device_arg)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device!r}")

        # Prefer MJPG so the Pi 4's USB bus can sustain high resolution/fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        # Keep the buffer shallow so we always process the freshest frame.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - backend dependent
            pass
        self._cap = cap

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            self.open()
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        return Frame(image=image, index=self._next_index())

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
