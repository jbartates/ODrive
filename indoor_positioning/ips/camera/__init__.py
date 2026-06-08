"""Camera sources and lens calibration helpers."""

from .base import CameraSource, Frame

__all__ = ["CameraSource", "Frame", "open_camera"]


def open_camera(config):
    """Factory that builds the camera source named in ``config.source``.

    Imports are deferred so that selecting one backend does not require the
    dependencies of the others (e.g. the file backend works without a GoPro).
    """
    source = config.source
    if source == "gopro_usb":
        from .gopro_usb import GoProUsbCamera

        return GoProUsbCamera(config)
    if source == "opencv":
        from .opencv_camera import OpenCVCamera

        return OpenCVCamera(config)
    if source == "file":
        from .file_camera import FileCamera

        return FileCamera(config)
    raise ValueError(f"Unknown camera source: {source!r}")
