"""Publish the fused pose as JSON over UDP (broadcast) or TCP.

The message schema is intentionally small and stable so any consumer (a
navigation node, a logger, the ODrive control layer) can parse it::

    {
      "type": "pose",
      "t": 1234.567,        # monotonic timestamp (s)
      "x": 1.23,            # world position (m)
      "y": -0.45,
      "theta": 0.78,        # heading (rad, CCW from world +x)
      "cov": [...9 floats...],  # row-major 3x3 covariance
      "n_markers": 2,       # markers used in the last fix
      "fixed": true         # true once at least one marker fix has landed
    }
"""

from __future__ import annotations

import json
import socket
from typing import Optional

import numpy as np

from ..config import OutputConfig
from ..geometry import Pose2D


def encode_pose_message(
    pose: Pose2D,
    timestamp: float,
    cov: Optional[np.ndarray] = None,
    n_markers: int = 0,
    fixed: bool = False,
) -> bytes:
    payload = {
        "type": "pose",
        "t": round(float(timestamp), 6),
        "x": round(float(pose.x), 5),
        "y": round(float(pose.y), 5),
        "theta": round(float(pose.theta), 6),
        "n_markers": int(n_markers),
        "fixed": bool(fixed),
    }
    if cov is not None:
        payload["cov"] = [round(float(v), 8) for v in np.asarray(cov).reshape(-1)]
    return (json.dumps(payload) + "\n").encode("utf-8")


class PosePublisher:
    """Send pose messages over UDP or TCP.

    UDP is the default: it is connectionless and supports broadcast, which
    suits a robot pushing state to whoever is listening.  TCP is offered for
    a single reliable consumer.
    """

    def __init__(self, config: OutputConfig):
        self.config = config
        self._sock: Optional[socket.socket] = None

    def open(self) -> None:
        if self.config.protocol == "udp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._sock = sock
        elif self.config.protocol == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.config.host, self.config.port))
            self._sock = sock
        else:
            raise ValueError(f"Unknown protocol: {self.config.protocol!r}")

    def publish(self, message: bytes) -> None:
        if self._sock is None:
            self.open()
        try:
            if self.config.protocol == "udp":
                self._sock.sendto(message, (self.config.host, self.config.port))
            else:
                self._sock.sendall(message)
        except OSError:
            # A transient network error should never crash the estimator.
            # Drop this message; the next one carries the latest state anyway.
            pass

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "PosePublisher":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
