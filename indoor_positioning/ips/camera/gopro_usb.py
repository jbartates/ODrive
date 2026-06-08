"""GoPro Max 2 USB-webcam backend for the Raspberry Pi 4.

The GoPro Max 2 in **USB webcam mode** enumerates as a UVC device once the
GoPro webcam handshake has been performed.  On a Pi 4 the most reliable way
to drive it is:

1. Put the camera into USB/webcam mode (a long press of the mode button, or
   the GoPro app).  Connect it to the Pi with a USB-C cable.
2. Start the webcam endpoint.  GoPro streams MJPEG/H264 over a UVC gadget;
   the open-source ``gopro_webcam`` helper (or the official desktop utility)
   exposes it as ``/dev/video*``.  See the package README for the exact
   one-liner and a systemd unit.
3. From there it is an ordinary V4L2 device, so we reuse
   :class:`OpenCVCamera` for the actual capture and only add GoPro-specific
   bring-up / health checks here.

Because the Max 2 is a 360 camera, choose a *reframed* (linear/HERO) output
when configuring the webcam so the frames are rectilinear and match a
pinhole calibration.  If you must work with the raw equirectangular feed,
set ``model: fisheye``/use the dewarp helper in :mod:`ips.camera.calibration`.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

from .base import Frame
from .opencv_camera import OpenCVCamera


class GoProUsbCamera(OpenCVCamera):
    """A V4L2 camera with GoPro-specific bring-up convenience."""

    def open(self) -> None:
        self._ensure_webcam_endpoint()
        super().open()

    def _ensure_webcam_endpoint(self) -> None:
        """Best-effort nudge of the GoPro webcam helper if it is installed.

        We never hard-fail here: if the helper is missing we assume the user
        started the stream themselves and fall through to opening the V4L2
        device, which will raise a clear error if it is absent.
        """
        helper = shutil.which("gopro-webcam") or shutil.which("gopro_webcam")
        if helper is None:
            return
        try:
            # ``--auto-start`` is a no-op if the endpoint is already up.
            subprocess.run(
                [helper, "--auto-start"],
                timeout=15,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:  # pragma: no cover - helper is optional
            # The device may already be streaming; let open() decide.
            pass

    def read(self) -> Optional[Frame]:
        return super().read()
