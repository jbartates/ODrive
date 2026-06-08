"""The top-level orchestrator that wires the whole pipeline together.

``PositioningSystem`` owns every component and exposes two things:

* :meth:`process_frame` -- run one frame end to end (odometry predict, marker
  updates, publish/log) and return the current fused pose.  This is pure
  enough to drive from a video file in tests.
* :meth:`run` -- the live loop: pull frames from the camera until interrupted.

It is deliberately tolerant: a frame with no markers simply propagates
odometry; a frame the camera fails to deliver is skipped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .camera import open_camera
from .camera.calibration import Undistorter
from .config import CameraCalibration, MarkerMap, SystemConfig
from .fusion import PoseEKF
from .geometry import Pose2D
from .imu import ImuPreintegrator, ImuSource
from .markers import ArucoDetector, MarkerLocalizer
from .odometry import VisualOdometry
from .output import PoseLogger, PosePublisher, encode_pose_message


@dataclass
class FrameResult:
    pose: Pose2D
    n_markers: int
    fixed: bool
    timestamp: float
    imu_samples: int = 0
    stationary: bool = False


class PositioningSystem:
    def __init__(
        self,
        config: SystemConfig,
        calibration: CameraCalibration,
        marker_map: MarkerMap,
    ):
        self.config = config
        self.calibration = calibration
        self.marker_map = marker_map

        self.detector = ArucoDetector(config.detector, calibration)
        self.localizer = MarkerLocalizer(
            marker_map, config.camera, config.fusion
        )
        self.odometry = (
            VisualOdometry(config.odometry) if config.odometry.enabled else None
        )
        # IMU fusion: a preintegrator is always available so callers can inject
        # a live IMU source; the GPMF source is built here for offline replay.
        self.preintegrator = ImuPreintegrator(config.imu)
        self.imu_source: Optional[ImuSource] = None
        if config.imu.source == "gpmf":
            from .imu import GpmfImuSource

            self.imu_source = GpmfImuSource(
                config.imu.video_path, config.imu.time_offset_s
            )
        self.ekf = PoseEKF(config.fusion)
        self.undistorter = (
            Undistorter(calibration) if _needs_undistort(calibration) else None
        )

        self.publisher = (
            PosePublisher(config.output) if config.output.publish else None
        )
        self.logger = (
            PoseLogger(config.output.log_path)
            if config.output.log_enabled
            else None
        )

        self._last_publish = 0.0
        self._publish_interval = (
            1.0 / config.output.rate_hz if config.output.rate_hz > 0 else 0.0
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_paths(
        cls,
        config_path,
        calibration_path,
        marker_map_path,
    ) -> "PositioningSystem":
        return cls(
            config=SystemConfig.load(config_path),
            calibration=CameraCalibration.load(calibration_path),
            marker_map=MarkerMap.load(marker_map_path),
        )

    # ------------------------------------------------------------------
    def process_frame(self, image, timestamp: Optional[float] = None) -> FrameResult:
        timestamp = time.monotonic() if timestamp is None else timestamp

        if self.undistorter is not None:
            image = self.undistorter.undistort(image)

        # 1. Gather relative motion: visual odometry for translation, IMU for
        #    heading.  The IMU samples are always drained (even before the
        #    filter is initialised) so bias estimation and the integrator's
        #    bridging state stay current.
        vo_est = self.odometry.process(image) if self.odometry is not None else None

        imu_inc = None
        n_imu = 0
        if self.imu_source is not None:
            samples = self.imu_source.until(timestamp)
            n_imu = len(samples)
            imu_inc = self.preintegrator.integrate(samples)

        # 2. Predict (only meaningful once an absolute fix has initialised us).
        if self.ekf.initialised:
            delta, cov = self._build_motion(vo_est, imu_inc)
            self.ekf.predict(delta, cov)

        # 3. Update with every marker that is in the map (absolute fixes).
        observations = self.detector.detect(image)
        fixes = self.localizer.fixes_from(observations)
        accepted = 0
        for fix in fixes:
            if self.ekf.update(fix.pose, fix.cov):
                accepted += 1

        pose = self.ekf.pose()
        result = FrameResult(
            pose=pose,
            n_markers=accepted,
            fixed=self.ekf.initialised,
            timestamp=timestamp,
            imu_samples=n_imu,
            stationary=bool(imu_inc.stationary) if imu_inc is not None else False,
        )
        self._emit(result)
        return result

    def _build_motion(self, vo_est, imu_inc):
        """Combine VO (translation) and IMU (heading) into one EKF control.

        Returns ``(delta, cov)`` where ``cov`` may be ``None`` to fall back on
        the static process noise (used when no relative-motion source fired).
        """
        dx = dy = dth = 0.0
        var_x = var_y = var_th = None

        if vo_est is not None and vo_est.valid:
            dx, dy, dth = vo_est.delta.x, vo_est.delta.y, vo_est.delta.theta
            var_x = float(vo_est.cov[0, 0])
            var_y = float(vo_est.cov[1, 1])
            var_th = float(vo_est.cov[2, 2])

        if imu_inc is not None and imu_inc.valid:
            # The gyro heading is far better than VO's, so it takes over yaw.
            dth = imu_inc.dtheta
            var_th = float(imu_inc.variance)

        if var_x is None and var_th is None:
            # Nothing fired this frame: zero motion, default process noise so
            # the covariance still grows to reflect the elapsed time.
            return Pose2D(), None

        f = self.config.fusion
        cov = np.diag(
            [
                var_x if var_x is not None else f.process_std_xy ** 2,
                var_y if var_y is not None else f.process_std_xy ** 2,
                var_th if var_th is not None else f.process_std_theta ** 2,
            ]
        )
        return Pose2D(dx, dy, dth), cov

    def set_imu_source(self, source: ImuSource) -> None:
        """Attach a live IMU source (e.g. an IMU wired to the Pi)."""
        self.imu_source = source

    def _emit(self, result: FrameResult) -> None:
        # Always log; rate-limit the network publish.
        if self.logger is not None:
            self.logger.log(
                result.pose,
                result.timestamp,
                cov=self.ekf.cov,
                n_markers=result.n_markers,
                fixed=result.fixed,
            )
        if self.publisher is not None:
            if (
                self._publish_interval <= 0.0
                or result.timestamp - self._last_publish >= self._publish_interval
            ):
                message = encode_pose_message(
                    result.pose,
                    result.timestamp,
                    cov=self.ekf.cov,
                    n_markers=result.n_markers,
                    fixed=result.fixed,
                )
                self.publisher.publish(message)
                self._last_publish = result.timestamp

    # ------------------------------------------------------------------
    def run(self, max_frames: Optional[int] = None) -> None:
        """Live loop: stream frames from the configured camera until stopped."""
        if self.publisher is not None:
            self.publisher.open()
        if self.logger is not None:
            self.logger.open()

        count = 0
        try:
            with open_camera(self.config.camera) as camera:
                for frame in camera.frames():
                    self.process_frame(frame.image, frame.timestamp)
                    count += 1
                    if max_frames is not None and count >= max_frames:
                        break
        finally:
            self.close()

    def close(self) -> None:
        if self.publisher is not None:
            self.publisher.close()
        if self.logger is not None:
            self.logger.close()


def _needs_undistort(calibration: CameraCalibration) -> bool:
    """Undistort only when there are non-trivial distortion coefficients."""
    import numpy as np

    return bool(np.any(np.abs(calibration.dist_coeffs) > 1e-9))
