"""Abstract camera interface shared by every backend."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """A single captured frame plus the wall-clock time it was grabbed."""

    image: np.ndarray  # BGR uint8, shape (H, W, 3)
    timestamp: float = field(default_factory=time.monotonic)
    index: int = 0


class CameraSource(abc.ABC):
    """Common contract for all camera backends.

    Subclasses must implement :meth:`open`, :meth:`read` and :meth:`close`.
    The class supports the context-manager protocol so callers can write::

        with open_camera(cfg) as cam:
            for frame in cam.frames():
                ...
    """

    def __init__(self, config):
        self.config = config
        self._index = 0
        self._opened = False

    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the underlying device. Idempotent where possible."""

    @abc.abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame, or ``None`` if no frame is available."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the device."""

    # -- convenience ----------------------------------------------------
    def frames(self):
        """Yield frames until the source is exhausted (or forever for live)."""
        if not self._opened:
            self.open()
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def _next_index(self) -> int:
        idx = self._index
        self._index += 1
        return idx

    def __enter__(self) -> "CameraSource":
        self.open()
        self._opened = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        self._opened = False
