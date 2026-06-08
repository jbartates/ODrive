"""Round-trip the UDP publisher over the loopback interface."""

import json
import os
import socket
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.config import OutputConfig  # noqa: E402
from ips.geometry import Pose2D  # noqa: E402
from ips.output import PosePublisher, encode_pose_message  # noqa: E402


def test_udp_publish_round_trip():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    port = receiver.getsockname()[1]

    cfg = OutputConfig(protocol="udp", host="127.0.0.1", port=port)
    with PosePublisher(cfg) as pub:
        pub.publish(
            encode_pose_message(Pose2D(3.0, 4.0, 1.0), timestamp=5.0, cov=np.eye(3))
        )
        data, _ = receiver.recvfrom(4096)

    receiver.close()
    obj = json.loads(data)
    assert obj["x"] == pytest.approx(3.0)
    assert obj["y"] == pytest.approx(4.0)


class _BoomSocket:
    """A stand-in socket whose sends always fail."""

    def sendto(self, *args, **kwargs):
        raise OSError("network down")

    def sendall(self, *args, **kwargs):
        raise OSError("network down")

    def close(self):
        pass


def test_publish_survives_send_error():
    """A transient socket error must not propagate out of publish()."""
    cfg = OutputConfig(protocol="udp", host="127.0.0.1", port=9)
    pub = PosePublisher(cfg)
    pub._sock = _BoomSocket()  # inject a failing socket
    # Should swallow the error rather than raise.
    pub.publish(b"{}\n")
    pub.close()
