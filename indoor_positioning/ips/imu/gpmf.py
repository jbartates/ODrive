"""A self-contained parser for GoPro's GPMF metadata format.

GPMF is a KLV (key-length-value) container.  Each entry is::

    [ 4-byte FourCC key ][ 1-byte type ][ 1-byte sample size ][ 2-byte count ]
    [ payload: size*count bytes, zero-padded up to a 4-byte boundary ]

A ``type`` of ``0`` marks a *nested* entry whose payload is itself a run of
KLV entries (this is how ``DEVC`` devices contain ``STRM`` streams, which in
turn contain the sensor data plus its scaling).

We implement just enough of the spec to pull the inertial streams we care
about -- ``GYRO``, ``ACCL`` and ``GRAV`` -- and apply their ``SCAL`` scaling.
No third-party dependencies; this is pure ``struct`` + ``numpy``.

Reference: https://github.com/gopro/gpmf-parser
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# type char -> (struct format, byte size)
_NUMERIC: Dict[str, Tuple[str, int]] = {
    "b": ("b", 1),
    "B": ("B", 1),
    "s": ("h", 2),
    "S": ("H", 2),
    "l": ("i", 4),
    "L": ("I", 4),
    "f": ("f", 4),
    "d": ("d", 8),
    "j": ("q", 8),
    "J": ("Q", 8),
}
# fixed-point types -> (byte size, struct format, divisor)
_FIXED: Dict[str, Tuple[int, str, float]] = {
    "q": (4, "i", float(1 << 16)),  # Q15.16
    "Q": (8, "q", float(1 << 32)),  # Q31.32
}


@dataclass
class GpmfEntry:
    key: str
    type: str
    sample_size: int
    count: int
    raw: bytes
    children: List["GpmfEntry"] = field(default_factory=list)

    @property
    def is_nested(self) -> bool:
        return self.type == "\x00"

    def numeric(self) -> Optional[np.ndarray]:
        """Decode the payload as a ``(count, cols)`` (or ``(count,)``) array.

        Returns ``None`` for non-numeric entries (strings, nested, ...).
        """
        return _decode_numeric(self.type, self.sample_size, self.count, self.raw)


def _decode_numeric(type_char, sample_size, count, raw):
    if type_char in _NUMERIC:
        fmt, size = _NUMERIC[type_char]
        divisor = 1.0
    elif type_char in _FIXED:
        size, fmt, divisor = _FIXED[type_char]
    else:
        return None
    if size == 0 or sample_size % size != 0:
        return None
    cols = sample_size // size
    total = cols * count
    needed = total * size
    if needed == 0 or len(raw) < needed:
        return None
    values = struct.unpack(f">{total}{fmt}", raw[:needed])
    arr = np.asarray(values, dtype=float)
    if divisor != 1.0:
        arr = arr / divisor
    if cols == 1:
        return arr.reshape(count)
    return arr.reshape(count, cols)


def parse(data: bytes, recurse: bool = True) -> List[GpmfEntry]:
    """Parse a buffer of GPMF KLV entries into a flat list (with children)."""
    entries: List[GpmfEntry] = []
    pos = 0
    n = len(data)
    while pos + 8 <= n:
        key = data[pos : pos + 4].decode("latin-1")
        type_char = chr(data[pos + 4])
        sample_size = data[pos + 5]
        count = struct.unpack(">H", data[pos + 6 : pos + 8])[0]
        pos += 8
        payload_len = sample_size * count
        payload = data[pos : pos + payload_len]
        entry = GpmfEntry(key, type_char, sample_size, count, payload)
        if recurse and type_char == "\x00":
            entry.children = parse(payload, recurse=True)
        entries.append(entry)
        # advance, padding the payload up to a 4-byte boundary
        pos += (payload_len + 3) & ~3
    return entries


def _find(entries: List[GpmfEntry], key: str) -> Optional[GpmfEntry]:
    for entry in entries:
        if entry.key == key:
            return entry
    return None


def _apply_scal(data: np.ndarray, scal: Optional[GpmfEntry]) -> np.ndarray:
    if scal is None:
        return data
    divisors = scal.numeric()
    if divisors is None:
        return data
    divisors = np.atleast_1d(divisors).astype(float)
    if data.ndim == 1:
        return data / divisors[0]
    if divisors.size == 1:
        return data / divisors[0]
    if divisors.size == data.shape[1]:
        return data / divisors.reshape(1, -1)
    return data / divisors[0]


def extract_streams(
    devc_payloads: List[Tuple[float, float, bytes]],
    wanted: Tuple[str, ...] = ("GYRO", "ACCL", "GRAV"),
) -> Dict[str, Dict[str, np.ndarray]]:
    """Collect sensor streams across all ``DEVC`` payloads.

    ``devc_payloads`` is a list of ``(t_start, t_end, raw_bytes)`` where the
    raw bytes are one DEVC container's KLV (typically one per ~second of
    video) and the times bound that container on the shared clock.

    Returns ``{stream_key: {"t": (N,), "data": (N, cols)}}`` with SCAL
    applied and per-sample timestamps interpolated across each payload.
    """
    collected: Dict[str, Dict[str, List[np.ndarray]]] = {
        k: {"t": [], "data": []} for k in wanted
    }

    for t_start, t_end, raw in devc_payloads:
        devc = parse(raw)
        # A DEVC payload is one top-level nested entry; tolerate either the
        # container or its already-unwrapped children being passed in.
        top = devc[0].children if len(devc) == 1 and devc[0].is_nested else devc
        for strm in (e for e in top if e.key == "STRM"):
            scal = _find(strm.children, "SCAL")
            for key in wanted:
                sensor = _find(strm.children, key)
                if sensor is None:
                    continue
                values = sensor.numeric()
                if values is None or sensor.count == 0:
                    continue
                values = _apply_scal(values, scal)
                n = sensor.count
                # Evenly space the n samples across [t_start, t_end).
                if n == 1 or t_end <= t_start:
                    times = np.full(n, t_start)
                else:
                    times = t_start + (t_end - t_start) * np.arange(n) / n
                collected[key]["t"].append(times)
                collected[key]["data"].append(np.atleast_2d(values.reshape(n, -1)))

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for key, parts in collected.items():
        if not parts["t"]:
            continue
        result[key] = {
            "t": np.concatenate(parts["t"]),
            "data": np.concatenate(parts["data"], axis=0),
        }
    return result
