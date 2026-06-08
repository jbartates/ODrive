#!/usr/bin/env python3
"""Live sanity-check for a WitMotion SINDT before running the full pipeline.

Opens the serial port, decodes the WitMotion stream, and prints the gyro,
accelerometer, sample rate and an integrated heading so you can confirm:

* the **port** and **baud** are right (samples arrive at a sensible rate),
* the SINDT's **output rate** is high enough (aim for >= 100 Hz),
* the **yaw_sign** is correct -- turn the robot left/CCW and watch that the
  reported heading INCREASES; if it decreases, pass ``--yaw-sign -1``.

Example
-------
    python scripts/imu_monitor.py --port /dev/ttyUSB0 --baud 9600
    python scripts/imu_monitor.py --port /dev/ttyUSB0 --baud 115200 --yaw-sign -1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.config import ImuConfig  # noqa: E402
from ips.imu import ImuPreintegrator, WitmotionImuSource  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--gyro-range", type=float, default=2000.0, help="deg/s")
    parser.add_argument("--accel-range", type=float, default=16.0, help="g")
    parser.add_argument("--yaw-sign", type=float, default=1.0, choices=[1.0, -1.0])
    parser.add_argument(
        "--duration", type=float, default=0.0, help="seconds (0 = until Ctrl-C)"
    )
    args = parser.parse_args()

    config = ImuConfig(
        source="witmotion",
        port=args.port,
        baud=args.baud,
        gyro_range_dps=args.gyro_range,
        accel_range_g=args.accel_range,
        yaw_sign=args.yaw_sign,
    )
    source = WitmotionImuSource(config)
    pre = ImuPreintegrator(config)

    print(f"Opening {args.port} @ {args.baud} baud ... (Ctrl-C to stop)")
    try:
        source.start()
    except Exception as exc:  # pragma: no cover - hardware/permission errors
        print(f"\nFailed to open serial port: {exc}", file=sys.stderr)
        print(
            "Check the port path, that no other program holds it, and that you "
            "are in the 'dialout' group (sudo usermod -aG dialout $USER).",
            file=sys.stderr,
        )
        return 1

    heading = 0.0
    total = 0
    start = time.monotonic()
    last_print = start
    window_count = 0
    last_gyro = np.zeros(3)
    last_accel = None
    stationary = False

    try:
        while True:
            now = time.monotonic()
            samples = source.until(now)
            if samples:
                inc = pre.integrate(samples)
                heading += inc.dtheta
                stationary = inc.stationary
                total += len(samples)
                window_count += len(samples)
                last_gyro = samples[-1].gyro
                last_accel = samples[-1].accel

            if now - last_print >= 0.2:
                rate = window_count / (now - last_print) if now > last_print else 0.0
                window_count = 0
                last_print = now
                gyro_dps = np.degrees(last_gyro)
                accel_txt = (
                    "  n/a"
                    if last_accel is None
                    else f"[{last_accel[0]:+5.2f} {last_accel[1]:+5.2f} {last_accel[2]:+5.2f}] m/s^2"
                )
                sys.stdout.write(
                    f"\rrate {rate:5.0f} Hz | gyro "
                    f"[{gyro_dps[0]:+7.2f} {gyro_dps[1]:+7.2f} {gyro_dps[2]:+7.2f}] dps"
                    f" | accel {accel_txt}"
                    f" | heading {math.degrees(heading):+7.1f} deg"
                    f" | {'STILL' if stationary else 'moving'}   "
                )
                sys.stdout.flush()

            if args.duration and (now - start) >= args.duration:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()

    print()
    if total == 0:
        print(
            "No samples received. Wrong baud, wrong port, or the SINDT is not "
            "streaming. Try --baud 115200.",
            file=sys.stderr,
        )
        return 1
    elapsed = max(time.monotonic() - start, 1e-6)
    print(f"Received {total} samples in {elapsed:.1f}s (~{total / elapsed:.0f} Hz).")
    print(f"Net heading change: {math.degrees(heading):+.1f} deg")
    print(f"Estimated gyro bias: {pre.gyro_bias} rad/s")
    print(
        "If turning the robot left/CCW made heading DEcrease, re-run with "
        "--yaw-sign -1 (and set imu.yaw_sign accordingly in config.yaml)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
