"""Minimal ISO base-media (MP4) reader for GoPro telemetry extraction.

We only implement what is needed to locate the GoPro metadata track (the
``stsd`` entry whose format is ``gpmd``) and pull out each of its samples
with the correct presentation time.  Boxes are read through an ``mmap`` so a
multi-gigabyte clip does not have to be loaded into RAM.

No third-party dependencies.
"""

from __future__ import annotations

import mmap
import struct
from typing import List, Optional, Tuple

# Boxes that contain child boxes and so must be descended into.
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta"}


def _iter_boxes(buf, start: int, end: int):
    """Yield ``(type, payload_start, payload_end, box_end)`` within a range."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos : pos + 4])[0]
        typ = bytes(buf[pos + 4 : pos + 8])
        header = 8
        if size == 1:  # 64-bit extended size
            size = struct.unpack(">Q", buf[pos + 8 : pos + 16])[0]
            header = 16
        elif size == 0:  # box extends to the end of the range
            size = end - pos
        if size < header:
            break
        yield typ, pos + header, pos + size, pos + size
        pos += size


def _find(buf, start: int, end: int, typ: bytes):
    for box_type, p_start, p_end, _ in _iter_boxes(buf, start, end):
        if box_type == typ:
            return p_start, p_end
    return None


def _find_path(buf, start, end, path: Tuple[bytes, ...]):
    cur = (start, end)
    for typ in path:
        found = _find(buf, cur[0], cur[1], typ)
        if found is None:
            return None
        cur = found
    return cur


def _parse_stsz(buf, start, end) -> List[int]:
    sample_size = struct.unpack(">I", buf[start + 4 : start + 8])[0]
    count = struct.unpack(">I", buf[start + 8 : start + 12])[0]
    if sample_size != 0:
        return [sample_size] * count
    base = start + 12
    return [
        struct.unpack(">I", buf[base + 4 * i : base + 4 * i + 4])[0]
        for i in range(count)
    ]


def _parse_stsc(buf, start, end) -> List[Tuple[int, int, int]]:
    count = struct.unpack(">I", buf[start + 4 : start + 8])[0]
    base = start + 8
    out = []
    for i in range(count):
        off = base + 12 * i
        first_chunk = struct.unpack(">I", buf[off : off + 4])[0]
        per_chunk = struct.unpack(">I", buf[off + 4 : off + 8])[0]
        desc = struct.unpack(">I", buf[off + 8 : off + 12])[0]
        out.append((first_chunk, per_chunk, desc))
    return out


def _parse_chunk_offsets(buf, start, end, wide: bool) -> List[int]:
    count = struct.unpack(">I", buf[start + 4 : start + 8])[0]
    base = start + 8
    step = 8 if wide else 4
    fmt = ">Q" if wide else ">I"
    return [
        struct.unpack(fmt, buf[base + step * i : base + step * i + step])[0]
        for i in range(count)
    ]


def _parse_stts(buf, start, end) -> List[Tuple[int, int]]:
    count = struct.unpack(">I", buf[start + 4 : start + 8])[0]
    base = start + 8
    out = []
    for i in range(count):
        off = base + 8 * i
        sample_count = struct.unpack(">I", buf[off : off + 4])[0]
        delta = struct.unpack(">I", buf[off + 4 : off + 8])[0]
        out.append((sample_count, delta))
    return out


def _stsd_format(buf, start, end) -> Optional[bytes]:
    # payload: [ver/flags 4][entry_count 4][entry: size 4][format 4]...
    if end - start < 16:
        return None
    return bytes(buf[start + 12 : start + 16])


def _sample_offsets(stsc, chunk_offsets, sample_sizes) -> List[int]:
    num_chunks = len(chunk_offsets)
    per_chunk = [0] * num_chunks
    for i, (first_chunk, samples, _) in enumerate(stsc):
        last = (stsc[i + 1][0] - 1) if i + 1 < len(stsc) else num_chunks
        for c in range(first_chunk, last + 1):
            if 1 <= c <= num_chunks:
                per_chunk[c - 1] = samples

    offsets: List[int] = []
    idx = 0
    for c in range(num_chunks):
        off = chunk_offsets[c]
        for _ in range(per_chunk[c]):
            if idx >= len(sample_sizes):
                break
            offsets.append(off)
            off += sample_sizes[idx]
            idx += 1
    return offsets


def _sample_times(stts, timescale) -> List[Tuple[float, float]]:
    """Return ``(t_start, t_end)`` seconds for every sample."""
    times: List[Tuple[float, float]] = []
    decode = 0
    ts = float(timescale) if timescale else 1.0
    for sample_count, delta in stts:
        for _ in range(sample_count):
            times.append((decode / ts, (decode + delta) / ts))
            decode += delta
    return times


def extract_gpmd_payloads(path) -> List[Tuple[float, float, bytes]]:
    """Return ``(t_start, t_end, raw_bytes)`` for each GPMF (gpmd) sample.

    Raises ``ValueError`` if the file has no GoPro metadata track.
    """
    with open(path, "rb") as handle:
        buf = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return _extract(buf)
        finally:
            buf.close()


def _extract(buf) -> List[Tuple[float, float, bytes]]:
    end = len(buf)
    moov = _find(buf, 0, end, b"moov")
    if moov is None:
        raise ValueError("No moov box: not a valid MP4 file")

    for typ, p_start, p_end, _ in _iter_boxes(buf, moov[0], moov[1]):
        if typ != b"trak":
            continue
        stbl = _find_path(buf, p_start, p_end, (b"mdia", b"minf", b"stbl"))
        if stbl is None:
            continue
        stsd = _find(buf, stbl[0], stbl[1], b"stsd")
        if stsd is None or _stsd_format(buf, stsd[0], stsd[1]) != b"gpmd":
            continue

        mdhd = _find_path(buf, p_start, p_end, (b"mdia", b"mdhd"))
        timescale = 1
        if mdhd is not None:
            version = buf[mdhd[0]]
            ts_off = mdhd[0] + (20 if version == 1 else 12)
            timescale = struct.unpack(">I", buf[ts_off : ts_off + 4])[0]

        stsz = _find(buf, stbl[0], stbl[1], b"stsz")
        stsc = _find(buf, stbl[0], stbl[1], b"stsc")
        stco = _find(buf, stbl[0], stbl[1], b"stco")
        co64 = _find(buf, stbl[0], stbl[1], b"co64")
        stts = _find(buf, stbl[0], stbl[1], b"stts")
        if stsz is None or stsc is None or stts is None or (stco is None and co64 is None):
            continue

        sizes = _parse_stsz(buf, *stsz)
        chunks = (
            _parse_chunk_offsets(buf, stco[0], stco[1], wide=False)
            if stco is not None
            else _parse_chunk_offsets(buf, co64[0], co64[1], wide=True)
        )
        offsets = _sample_offsets(_parse_stsc(buf, *stsc), chunks, sizes)
        times = _sample_times(_parse_stts(buf, *stts), timescale)

        payloads: List[Tuple[float, float, bytes]] = []
        for i, offset in enumerate(offsets):
            size = sizes[i]
            t_start, t_end = times[i] if i < len(times) else (0.0, 0.0)
            payloads.append((t_start, t_end, bytes(buf[offset : offset + size])))
        return payloads

    raise ValueError("No GoPro metadata (gpmd) track found in file")
