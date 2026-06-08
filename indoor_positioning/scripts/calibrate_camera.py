#!/usr/bin/env python3
"""Calibrate the (reframed) GoPro view from a set of chessboard images.

Capture 15-30 photos of a printed chessboard at varied angles/distances with
the exact camera configuration you will run with, then::

    python scripts/calibrate_camera.py \
        --images calib_images/ \
        --rows 6 --cols 9 --square 0.025 \
        --out config/camera_calibration.yaml

For a single GoPro fisheye lens you can pass ``--model fisheye``.  If you run
the camera in a reframed *linear* mode the default pinhole model is correct.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.config import CameraCalibration  # noqa: E402


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="folder of chessboard images")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--square", type=float, default=0.025, help="square size (m)")
    parser.add_argument("--model", choices=["pinhole", "fisheye"], default="pinhole")
    parser.add_argument("--out", default="config/camera_calibration.yaml")
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    objp *= args.square

    obj_points, img_points = [], []
    image_size = None
    paths = sorted(
        p
        for ext in ("png", "jpg", "jpeg", "bmp")
        for p in glob.glob(os.path.join(args.images, f"*.{ext}"))
    )
    if not paths:
        print(f"No images found in {args.images!r}", file=sys.stderr)
        return 1

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not found:
            print(f"  chessboard NOT found in {os.path.basename(path)}")
            continue
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        obj_points.append(objp)
        img_points.append(corners)
        print(f"  found corners in {os.path.basename(path)}")

    if len(obj_points) < 5:
        print("Need at least 5 good views to calibrate.", file=sys.stderr)
        return 1

    if args.model == "fisheye":
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            [o.reshape(-1, 1, 3) for o in obj_points],
            img_points,
            image_size,
            K,
            D,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_FIX_SKEW,
        )
        dist = D.reshape(-1)
    else:
        rms, K, dist, _, _ = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
        dist = dist.reshape(-1)

    calib = CameraCalibration(
        camera_matrix=K,
        dist_coeffs=dist,
        width=image_size[0],
        height=image_size[1],
        model=args.model,
    )
    calib.save(args.out)
    print(f"\nRMS reprojection error: {rms:.4f} px")
    print(f"Wrote calibration to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
