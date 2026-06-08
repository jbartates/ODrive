"""Append pose estimates to a JSON-lines file for offline analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..geometry import Pose2D


class PoseLogger:
    """One JSON object per line; safe to tail while the robot runs."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._handle = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(
        self,
        pose: Pose2D,
        timestamp: float,
        cov: Optional[np.ndarray] = None,
        n_markers: int = 0,
        fixed: bool = False,
    ) -> None:
        if self._handle is None:
            self.open()
        record = {
            "t": float(timestamp),
            "x": float(pose.x),
            "y": float(pose.y),
            "theta": float(pose.theta),
            "n_markers": int(n_markers),
            "fixed": bool(fixed),
        }
        if cov is not None:
            record["cov"] = [float(v) for v in np.asarray(cov).reshape(-1)]
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "PoseLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
