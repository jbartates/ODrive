"""WitMotion SINDT IMU over USB (serial), for live runs.

The WitMotion SINDT enumerates as a USB-serial device (CH340) and emits the
standard WitMotion binary protocol: a stream of fixed 11-byte frames

    [ 0x55 ][ kind ][ d0 d1 d2 d3 d4 d5 d6 d7 ][ checksum ]

where the 8 data bytes are four little-endian ``int16`` values and the
checksum is ``sum(first 10 bytes) & 0xFF``.  We care about three kinds:

* ``0x51`` acceleration  -> ax, ay, az  (and temperature)
* ``0x52`` angular rate  -> wx, wy, wz  (gyro, deg/s; and temperature)
* ``0x53`` angle         -> roll, pitch, yaw (deg; and version)

The SINDT is a 6-axis unit (no magnetometer), so its reported *yaw angle*
drifts; we therefore use the **gyro** (0x52) for heading rate and the
**accelerometer** (0x51) as the gravity reference, exactly matching the
:class:`~ips.imu.preintegrator.ImuPreintegrator` model (which projects gyro
onto gravity, so the result is independent of how the SINDT is mounted).

The :class:`WitmotionDecoder` is pure and unit-tested; the
:class:`WitmotionImuSource` wraps it with a background serial-reader thread.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from .sample import ImuSample
from .source import ImuSource

_HEADER = 0x55
_ACCEL = 0x51
_GYRO = 0x52
_ANGLE = 0x53
_FRAME_LEN = 11
_GRAVITY = 9.80665


class WitmotionDecoder:
    """Stateful byte-stream decoder that emits an ImuSample per gyro frame.

    Feed it whatever bytes arrive from the serial port; it resynchronises on
    the 0x55 header, validates the checksum, and pairs the most recent
    acceleration reading with each gyro reading.
    """

    def __init__(self, gyro_range_dps: float = 2000.0, accel_range_g: float = 16.0):
        self.gyro_scale = math.radians(gyro_range_dps) / 32768.0
        self.accel_scale = accel_range_g * _GRAVITY / 32768.0
        self._buf = bytearray()
        self._latest_accel: Optional[np.ndarray] = None

    def feed(self, data: bytes, timestamp: float) -> List[ImuSample]:
        """Consume bytes and return any IMU samples completed by them."""
        self._buf.extend(data)
        samples: List[ImuSample] = []

        while len(self._buf) >= _FRAME_LEN:
            if self._buf[0] != _HEADER:
                # Drop bytes until the next plausible header.
                idx = self._buf.find(_HEADER, 1)
                if idx == -1:
                    self._buf.clear()
                else:
                    del self._buf[:idx]
                continue

            frame = bytes(self._buf[:_FRAME_LEN])
            if (sum(frame[:10]) & 0xFF) != frame[10]:
                # Bad checksum: this wasn't a real frame start; skip one byte.
                del self._buf[:1]
                continue

            del self._buf[:_FRAME_LEN]
            sample = self._handle_frame(frame, timestamp)
            if sample is not None:
                samples.append(sample)
        return samples

    def _values(self, frame: bytes) -> np.ndarray:
        return np.array(struct.unpack("<4h", frame[2:10]), dtype=float)

    def _handle_frame(self, frame: bytes, timestamp: float) -> Optional[ImuSample]:
        kind = frame[1]
        if kind == _ACCEL:
            self._latest_accel = self._values(frame)[:3] * self.accel_scale
            return None
        if kind == _GYRO:
            gyro = self._values(frame)[:3] * self.gyro_scale
            return ImuSample(t=timestamp, gyro=gyro, accel=self._latest_accel)
        # Angle (0x53) and other frame kinds are ignored: yaw drifts on a
        # 6-axis unit, and we get gravity from the accelerometer instead.
        return None


class WitmotionImuSource(ImuSource):
    """Live IMU source backed by a WitMotion SINDT on a serial port."""

    def __init__(self, config):
        super().__init__([])
        self.config = config
        self._serial = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._queue: Deque[ImuSample] = deque(maxlen=20000)
        self._decoder = WitmotionDecoder(
            gyro_range_dps=config.gyro_range_dps,
            accel_range_g=config.accel_range_g,
        )

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        import serial  # pyserial; deferred so the pure-math tests don't need it

        self._serial = serial.Serial(
            self.config.port, self.config.baud, timeout=0.05
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._serial.read(64)
            except Exception:  # pragma: no cover - serial hiccup
                continue
            if not data:
                continue
            now = time.monotonic() + self.config.time_offset_s
            samples = self._decoder.feed(data, now)
            if samples:
                with self._lock:
                    self._queue.extend(samples)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    # -- consumption ----------------------------------------------------
    def until(self, t: float) -> List[ImuSample]:
        out: List[ImuSample] = []
        with self._lock:
            while self._queue and self._queue[0].t <= t:
                out.append(self._queue.popleft())
        return out
