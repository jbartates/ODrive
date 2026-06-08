"""Unit tests for the WitMotion SINDT serial protocol decoder."""

import math
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.imu.witmotion import WitmotionDecoder  # noqa: E402

_G = 9.80665


def frame(kind: int, values) -> bytes:
    """Build a valid 11-byte WitMotion frame (4 little-endian int16 + cksum)."""
    body = bytearray([0x55, kind])
    for v in values:
        body += struct.pack("<h", int(v))
    body.append(sum(body) & 0xFF)
    return bytes(body)


def test_accel_then_gyro_emits_paired_sample():
    dec = WitmotionDecoder(gyro_range_dps=2000.0, accel_range_g=16.0)
    # accel: raw 2048 on x -> 2048/32768 * 16 g = 1 g = 9.80665 m/s^2
    out = dec.feed(frame(0x51, [2048, 0, 0, 25]), timestamp=1.0)
    assert out == []  # accel alone does not emit
    # gyro: raw 16384 on z -> 16384/32768 * 2000 = 1000 deg/s
    out = dec.feed(frame(0x52, [0, 0, 16384, 25]), timestamp=1.1)
    assert len(out) == 1
    sample = out[0]
    assert sample.t == pytest.approx(1.1)
    assert sample.gyro[2] == pytest.approx(math.radians(1000.0), rel=1e-6)
    assert sample.accel[0] == pytest.approx(_G, rel=1e-6)


def test_gyro_without_accel_has_none_accel():
    dec = WitmotionDecoder()
    out = dec.feed(frame(0x52, [0, 0, 0, 0]), timestamp=2.0)
    assert len(out) == 1
    assert out[0].accel is None
    np.testing.assert_allclose(out[0].gyro, [0.0, 0.0, 0.0])


def test_negative_values_are_signed():
    dec = WitmotionDecoder(gyro_range_dps=2000.0)
    out = dec.feed(frame(0x52, [-16384, 0, 0, 0]), timestamp=0.0)
    assert out[0].gyro[0] == pytest.approx(math.radians(-1000.0), rel=1e-6)


def test_resync_after_garbage_bytes():
    dec = WitmotionDecoder()
    stream = bytes([0x00, 0xAB, 0x55]) + frame(0x52, [0, 0, 8192, 0])
    out = dec.feed(stream, timestamp=0.0)
    assert len(out) == 1
    assert out[0].gyro[2] == pytest.approx(math.radians(500.0), rel=1e-6)


def test_bad_checksum_frame_is_skipped():
    dec = WitmotionDecoder()
    bad = bytearray(frame(0x52, [0, 0, 16384, 0]))
    bad[-1] ^= 0xFF  # corrupt the checksum
    good = frame(0x52, [0, 0, 8192, 0])
    out = dec.feed(bytes(bad) + good, timestamp=0.0)
    # The corrupted frame is discarded; the following valid one still decodes.
    assert len(out) == 1
    assert out[0].gyro[2] == pytest.approx(math.radians(500.0), rel=1e-6)


def test_frame_split_across_feeds():
    dec = WitmotionDecoder()
    full = frame(0x52, [0, 0, 16384, 0])
    assert dec.feed(full[:5], timestamp=0.0) == []  # partial: nothing yet
    out = dec.feed(full[5:], timestamp=0.5)
    assert len(out) == 1
    assert out[0].t == pytest.approx(0.5)


def test_multiple_frames_in_one_chunk():
    dec = WitmotionDecoder()
    chunk = (
        frame(0x51, [2048, 0, 0, 0])
        + frame(0x52, [0, 0, 0, 0])
        + frame(0x52, [0, 0, 16384, 0])
    )
    out = dec.feed(chunk, timestamp=3.0)
    assert len(out) == 2  # two gyro frames -> two samples
    assert out[1].gyro[2] == pytest.approx(math.radians(1000.0), rel=1e-6)
    # Both gyro samples carry the accel that preceded them in the chunk.
    assert out[0].accel[0] == pytest.approx(_G, rel=1e-6)
