"""Packaging for the indoor positioning system (ips)."""

from setuptools import find_packages, setup

setup(
    name="ips-indoor-positioning",
    version="0.1.0",
    description=(
        "Indoor positioning for a mobile robot using a GoPro Max 2 on a "
        "Raspberry Pi 4 (ArUco markers + visual odometry fused with an EKF)."
    ),
    packages=find_packages(include=["ips", "ips.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "PyYAML>=5.4",
        # opencv-contrib-python is required at runtime for cv2.aruco but is
        # left out of install_requires so CI can run the pure-math tests
        # without pulling the large wheel. Install it from requirements.txt.
    ],
    extras_require={
        "camera": ["opencv-contrib-python>=4.6"],
        "test": ["pytest>=6.0"],
    },
)
