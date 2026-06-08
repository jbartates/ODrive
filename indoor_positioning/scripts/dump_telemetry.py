#!/usr/bin/env python3
"""Inspect the IMU telemetry inside a recorded GoPro MP4.

Useful for confirming the GPMF track parses and for choosing ``imu.yaw_sign``:
turn the robot a known direction while recording, then check the sign of the
integrated heading reported here.

Example
-------
    python scripts/dump_telemetry.py my_clip.MP4
    python scripts/dump_telemetry.py my_clip.MP4 --plot heading.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.config import ImuConfig  # noqa: E402
from ips.imu import GpmfImuSource, ImuPreintegrator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="recorded GoPro MP4")
    parser.add_argument("--yaw-sign", type=float, default=1.0)
    parser.add_argument(
        "--plot", default=None, help="write t,heading CSV to this path"
    )
    args = parser.parse_args()

    source = GpmfImuSource(args.video)
    samples = source.samples
    if not samples:
        print("No IMU samples found.", file=sys.stderr)
        return 1

    duration = samples[-1].t - samples[0].t
    rate = len(samples) / duration if duration > 0 else 0.0
    has_grav = samples[0].grav is not None
    has_accel = samples[0].accel is not None
    print(f"samples:    {len(samples)}")
    print(f"duration:   {duration:.2f} s  (~{rate:.0f} Hz)")
    print(f"gravity:    {'yes' if has_grav else 'no'}")
    print(f"accel:      {'yes' if has_accel else 'no'}")

    pre = ImuPreintegrator(ImuConfig(yaw_sign=args.yaw_sign))
    heading = 0.0
    rows = []
    for smp in samples:
        inc = pre.integrate([smp])
        heading += inc.dtheta
        rows.append((smp.t, heading))
    print(f"net heading change: {np.degrees(heading):.1f} deg")
    print(f"estimated gyro bias: {pre.gyro_bias}")

    if args.plot:
        with open(args.plot, "w", encoding="utf-8") as handle:
            handle.write("t,heading_rad\n")
            for t, h in rows:
                handle.write(f"{t:.6f},{h:.6f}\n")
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
