#!/usr/bin/env python3
"""Run the indoor positioning system live on the robot.

Example
-------
    python scripts/run_positioning.py \
        --config config/config.yaml \
        --calibration config/camera_calibration.yaml \
        --map config/marker_map.yaml

Listen to the published pose from any machine on the LAN with::

    nc -ul 9870        # UDP, line-delimited JSON
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.system import PositioningSystem  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--calibration", default="config/camera_calibration.yaml")
    parser.add_argument("--map", default="config/marker_map.yaml")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (default: run until interrupted).",
    )
    args = parser.parse_args()

    system = PositioningSystem.from_paths(args.config, args.calibration, args.map)
    print(
        f"[ips] starting: camera={system.config.camera.source} "
        f"markers={len(system.marker_map)} "
        f"publish={system.config.output.protocol}://"
        f"{system.config.output.host}:{system.config.output.port}"
    )
    try:
        system.run(max_frames=args.max_frames)
    except KeyboardInterrupt:
        print("\n[ips] stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
