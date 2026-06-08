import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ips.config import MarkerMap, SystemConfig  # noqa: E402
from ips.geometry import Pose2D  # noqa: E402
from ips.output.publisher import encode_pose_message  # noqa: E402

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config")
)


def test_example_config_loads():
    cfg = SystemConfig.load(os.path.join(CONFIG_DIR, "config.example.yaml"))
    assert cfg.camera.source == "gopro_usb"
    assert cfg.detector.marker_length_m == pytest.approx(0.15)
    assert cfg.output.port == 9870


def test_example_marker_map_loads_and_resolves_poses():
    mm = MarkerMap.load(os.path.join(CONFIG_DIR, "marker_map.example.yaml"))
    assert len(mm) == 5
    # Marker 0 on the west wall faces +x (yaw 0) at (0, 2).
    pose0 = mm.world_pose_of(0)
    assert pose0.x == pytest.approx(0.0)
    assert pose0.y == pytest.approx(2.0)
    assert pose0.theta == pytest.approx(0.0)
    # Marker 1 faces -x (yaw 180).
    assert abs(mm.world_pose_of(1).theta) == pytest.approx(math.pi)


def test_marker_map_yaw_to_rotation():
    mm = MarkerMap.from_dict(
        {"markers": [{"id": 7, "position": [1, 2, 3], "yaw_deg": 90}]}
    )
    T = mm.get(7)
    assert T is not None
    # +z column should point along world +y after a 90 deg yaw.
    assert np.allclose(T[:3, 2], [0, 1, 0], atol=1e-9) or np.allclose(
        T[:3, 0], [0, 1, 0], atol=1e-9
    )


def test_unknown_marker_returns_none():
    mm = MarkerMap()
    assert mm.get(123) is None
    assert mm.world_pose_of(123) is None


def test_encode_pose_message_is_valid_json_line():
    import json

    msg = encode_pose_message(
        Pose2D(1.0, 2.0, 0.5),
        timestamp=10.0,
        cov=np.eye(3),
        n_markers=2,
        fixed=True,
    )
    assert msg.endswith(b"\n")
    obj = json.loads(msg)
    assert obj["type"] == "pose"
    assert obj["x"] == pytest.approx(1.0)
    assert obj["n_markers"] == 2
    assert obj["fixed"] is True
    assert len(obj["cov"]) == 9
