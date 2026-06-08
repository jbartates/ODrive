"""Indoor Positioning System (IPS) for a mobile robot.

A self-contained package that estimates the 2D pose (x, y, heading) of a
mobile robot on a flat indoor floor using a GoPro Max 2 (USB webcam mode)
attached to a Raspberry Pi 4.

The pipeline is:

    camera frame  ->  ArUco marker detection (absolute fixes)
                  ->  visual odometry (smooth relative motion)
                  ->  EKF fusion  ->  network publish + log

The package is organised so that the heavy hardware / OpenCV dependent
pieces (``camera``, ``markers``, ``odometry``) are isolated from the pure
math (``geometry``, ``fusion``) which is fully unit tested without a
camera attached.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
