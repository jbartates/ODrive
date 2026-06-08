"""Unit tests for the GPMF KLV parser and stream extraction."""

import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.imu import gpmf  # noqa: E402
from ips.imu import mp4  # noqa: E402


def klv(key: str, type_char: str, sample_size: int, count: int, payload: bytes) -> bytes:
    type_byte = ord(type_char) if type_char else 0
    header = key.encode("latin-1") + bytes([type_byte, sample_size])
    header += struct.pack(">H", count)
    pad = (-len(payload)) % 4
    return header + payload + b"\x00" * pad


def _build_devc():
    # GYRO: 3x int16 per sample, 2 samples; SCAL divisor 100.
    gyro_payload = struct.pack(">6h", 100, 200, 300, 110, 210, 310)
    gyro = klv("GYRO", "s", 6, 2, gyro_payload)
    scal = klv("SCAL", "s", 2, 1, struct.pack(">h", 100))
    strm_payload = scal + gyro
    strm = klv("STRM", "\x00", 1, len(strm_payload), strm_payload)
    return klv("DEVC", "\x00", 1, len(strm), strm)


def test_parse_nested_structure():
    entries = gpmf.parse(_build_devc())
    assert len(entries) == 1
    devc = entries[0]
    assert devc.key == "DEVC"
    assert devc.is_nested
    strm = devc.children[0]
    assert strm.key == "STRM"
    keys = {c.key for c in strm.children}
    assert {"SCAL", "GYRO"} <= keys


def test_numeric_decode_with_columns():
    entries = gpmf.parse(_build_devc())
    gyro = entries[0].children[0].children[1]
    assert gyro.key == "GYRO"
    arr = gyro.numeric()
    assert arr.shape == (2, 3)
    # raw values, before SCAL is applied
    assert arr[0].tolist() == [100.0, 200.0, 300.0]


def test_extract_streams_applies_scaling_and_time():
    devc = _build_devc()
    streams = gpmf.extract_streams([(0.0, 1.0, devc)], wanted=("GYRO",))
    assert "GYRO" in streams
    data = streams["GYRO"]["data"]
    times = streams["GYRO"]["t"]
    assert data.shape == (2, 3)
    # SCAL divisor of 100 applied
    np.testing.assert_allclose(data[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(data[1], [1.1, 2.1, 3.1])
    # 2 samples spread evenly across [0, 1)
    np.testing.assert_allclose(times, [0.0, 0.5])


def test_extract_streams_concatenates_payloads():
    devc = _build_devc()
    streams = gpmf.extract_streams(
        [(0.0, 1.0, devc), (1.0, 2.0, devc)], wanted=("GYRO",)
    )
    assert streams["GYRO"]["data"].shape == (4, 3)
    np.testing.assert_allclose(streams["GYRO"]["t"], [0.0, 0.5, 1.0, 1.5])


def test_fixed_point_q1516_decoding():
    # type 'q' = Q15.16: value 1.5 -> 1.5 * 65536
    payload = struct.pack(">i", int(1.5 * (1 << 16)))
    entry = gpmf.parse(klv("TEST", "q", 4, 1, payload))[0]
    np.testing.assert_allclose(entry.numeric(), [1.5])


def test_padding_is_respected():
    # A 2-byte payload must be padded to 4 bytes; a following entry must parse.
    first = klv("AAAA", "s", 2, 1, struct.pack(">h", 7))
    second = klv("BBBB", "s", 2, 1, struct.pack(">h", 9))
    entries = gpmf.parse(first + second)
    assert [e.key for e in entries] == ["AAAA", "BBBB"]


# -- MP4 sample-table helpers -------------------------------------------------


def test_sample_offsets_single_chunk():
    # one chunk at offset 1000 holding 3 samples of sizes 10, 20, 30
    stsc = [(1, 3, 1)]
    offsets = mp4._sample_offsets(stsc, [1000], [10, 20, 30])
    assert offsets == [1000, 1010, 1030]


def test_sample_offsets_multiple_chunks():
    # two chunks, 2 samples each, sizes all 5
    stsc = [(1, 2, 1)]
    offsets = mp4._sample_offsets(stsc, [100, 200], [5, 5, 5, 5])
    assert offsets == [100, 105, 200, 205]


def test_sample_times_uses_timescale():
    # timescale 1000, two samples of 500 ticks each -> 0.5 s apart
    times = mp4._sample_times([(2, 500)], timescale=1000)
    assert times == [(0.0, 0.5), (0.5, 1.0)]
