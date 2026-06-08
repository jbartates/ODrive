#!/usr/bin/env python3
"""Generate printable ArUco marker images.

Example
-------
    # Make markers 0..9 from the 4x4_50 dictionary as 600px PNGs.
    python scripts/generate_markers.py --ids 0-9 --size 600 --out markers/

Print each at a known physical size and record that side length (in metres)
as ``detector.marker_length_m`` in config.yaml, then place them in the room
and record their world poses in marker_map.yaml.
"""

from __future__ import annotations

import argparse
import os
from typing import List


def parse_ids(spec: str) -> List[int]:
    ids: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            ids.extend(range(int(lo), int(hi) + 1))
        elif part:
            ids.append(int(part))
    return ids


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default="0-9", help="e.g. '0-9' or '0,3,7'")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--size", type=int, default=600, help="image px")
    parser.add_argument("--border", type=int, default=1, help="border bits")
    parser.add_argument("--out", default="markers")
    args = parser.parse_args()

    attr = getattr(cv2.aruco, args.dictionary)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(attr)
        draw = getattr(cv2.aruco, "generateImageMarker", None)
    else:  # pragma: no cover - legacy OpenCV
        dictionary = cv2.aruco.Dictionary_get(attr)
        draw = None

    os.makedirs(args.out, exist_ok=True)
    for marker_id in parse_ids(args.ids):
        if draw is not None:
            img = draw(dictionary, marker_id, args.size, borderBits=args.border)
        else:  # pragma: no cover - legacy OpenCV
            img = cv2.aruco.drawMarker(dictionary, marker_id, args.size)
        path = os.path.join(args.out, f"marker_{marker_id:03d}.png")
        cv2.imwrite(path, img)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
