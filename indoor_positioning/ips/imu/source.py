"""IMU sources: where timestamped samples come from.

* :class:`GpmfImuSource` reads the inertial streams out of a recorded GoPro
  MP4 (offline / replay use).
* :class:`ListImuSource` wraps an in-memory list -- handy for tests and for
  feeding samples from any other IMU (e.g. one wired to the Pi) in a live run.

Both expose :meth:`until`, which streams samples in timestamp order aligned to
the camera-frame clock the caller drives the pipeline with.
"""

from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np

from . import gpmf, mp4
from .sample import ImuSample


class ImuSource(abc.ABC):
    """Time-ordered IMU samples consumed monotonically via :meth:`until`."""

    def __init__(self, samples: Optional[List[ImuSample]] = None):
        self._samples: List[ImuSample] = list(samples or [])
        self._samples.sort(key=lambda s: s.t)
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> List[ImuSample]:
        return self._samples

    def reset(self) -> None:
        self._cursor = 0

    def start(self) -> None:
        """Begin acquiring samples. No-op for offline/in-memory sources."""

    def close(self) -> None:
        """Release any resources. No-op for offline/in-memory sources."""

    def until(self, t: float) -> List[ImuSample]:
        """Return (and consume) all unread samples with ``sample.t <= t``."""
        out: List[ImuSample] = []
        n = len(self._samples)
        while self._cursor < n and self._samples[self._cursor].t <= t:
            out.append(self._samples[self._cursor])
            self._cursor += 1
        return out


class ListImuSource(ImuSource):
    """An IMU source backed by an explicit list of samples."""


class GpmfImuSource(ImuSource):
    """Build IMU samples from the GPMF telemetry of a GoPro MP4."""

    def __init__(self, path: str, time_offset: float = 0.0):
        samples = self._load(path, time_offset)
        super().__init__(samples)
        self.path = path

    @staticmethod
    def _load(path: str, time_offset: float) -> List[ImuSample]:
        payloads = mp4.extract_gpmd_payloads(path)
        streams = gpmf.extract_streams(payloads, wanted=("GYRO", "ACCL", "GRAV"))
        if "GYRO" not in streams:
            raise ValueError(f"No GYRO stream found in GPMF telemetry of {path!r}")

        gyro_t = streams["GYRO"]["t"] + time_offset
        gyro = streams["GYRO"]["data"]

        accel = _resample(streams.get("ACCL"), gyro_t, time_offset)
        grav = _resample(streams.get("GRAV"), gyro_t, time_offset)

        samples: List[ImuSample] = []
        for i, t in enumerate(gyro_t):
            samples.append(
                ImuSample(
                    t=float(t),
                    gyro=gyro[i],
                    accel=None if accel is None else accel[i],
                    grav=None if grav is None else grav[i],
                )
            )
        return samples


def _resample(stream, target_t, time_offset) -> Optional[np.ndarray]:
    """Linearly resample a (t, data) stream onto ``target_t``."""
    if stream is None or stream["data"].size == 0:
        return None
    src_t = stream["t"] + time_offset
    data = stream["data"]
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    cols = data.shape[1]
    out = np.empty((len(target_t), cols))
    for c in range(cols):
        out[:, c] = np.interp(target_t, src_t, data[:, c])
    return out
