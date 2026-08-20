# ============================================================
# CAMERA MANAGER
# ============================================================

import threading
import time
import logging

from detection_multicam2 import (
    SafetyDetectorEngine,
)

from camera_config import (
    CAMERA_CONFIG,
    MAX_CAMERAS,
)


logger = logging.getLogger(__name__)


class CameraManager:

    def __init__(
        self,
        camera_config=None,
    ):

        self.config = (
            camera_config
            if camera_config is not None
            else CAMERA_CONFIG
        )

        self.engines = {}

        self.lock = (
            threading.RLock()
        )

        self.running = False

    # ========================================================
    # START ALL
    # ========================================================

    def start_all(self):

        if self.running:

            logger.warning(
                "CameraManager already running."
            )

            return

        self.running = True

        logger.info(
            "Starting CameraManager..."
        )

        for camera_id in range(
            1,
            MAX_CAMERAS + 1,
        ):

            self.start_camera(
                camera_id
            )

        logger.info(
            "CameraManager startup completed."
        )

    # ========================================================
    # START CAMERA
    # ========================================================

    def start_camera(
        self,
        camera_id,
    ):

        camera_id = int(
            camera_id
        )

        config = self.config.get(
            camera_id
        )

        # ----------------------------------------------------
        # No configuration
        # ----------------------------------------------------

        if config is None:

            logger.warning(
                "Camera %s has no configuration.",
                camera_id,
            )

            return False

        # ----------------------------------------------------
        # Disabled
        # ----------------------------------------------------

        if not config.get(
            "enabled",
            False,
        ):

            logger.info(
                "Camera %s disabled. "
                "No pipeline will be created.",
                camera_id,
            )

            return False

        # ----------------------------------------------------
        # RTSP missing
        # ----------------------------------------------------

        rtsp_url = config.get(
            "rtsp_url",
            "",
        )

        if not rtsp_url:

            logger.warning(
                "Camera %s enabled but RTSP URL is empty.",
                camera_id,
            )

            return False

        # ----------------------------------------------------
        # Already running
        # ----------------------------------------------------

        with self.lock:

            if camera_id in self.engines:

                existing = self.engines[
                    camera_id
                ]

                if existing.is_running:

                    logger.warning(
                        "Camera %s already running.",
                        camera_id,
                    )

                    return True

                del self.engines[
                    camera_id
                ]

        # ----------------------------------------------------
        # Create engine
        # ----------------------------------------------------

        engine = SafetyDetectorEngine(

            camera_id=camera_id,

            rtsp_url=rtsp_url,

            focal_length=config.get(
                "focal_length",
                600.0,
            ),

            camera_height=config.get(
                "camera_height",
                2.0,
            ),

            camera_pitch=config.get(
                "camera_pitch",
                0.0,
            ),

            danger_distance=config.get(
                "danger_distance",
                7.2,
            ),
        )

        # ----------------------------------------------------
        # Register BEFORE start
        # ----------------------------------------------------

        with self.lock:

            self.engines[
                camera_id
            ] = engine

        logger.info(
            "Starting Camera %s (%s)...",
            camera_id,
            config.get(
                "name",
                f"Camera {camera_id}",
            ),
        )

        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        success = engine.start()

        if success:

            logger.info(
                "Camera %s ONLINE.",
                camera_id,
            )

            return True

        # ----------------------------------------------------
        # Start failed
        # ----------------------------------------------------

        logger.error(
            "Camera %s failed to start: %s",
            camera_id,
            engine.start_error,
        )

        return False

    # ========================================================
    # STOP CAMERA
    # ========================================================

    def stop_camera(
        self,
        camera_id,
    ):

        camera_id = int(
            camera_id
        )

        with self.lock:

            engine = self.engines.get(
                camera_id
            )

        if engine is None:

            logger.info(
                "Camera %s is not active.",
                camera_id,
            )

            return

        logger.info(
            "Stopping Camera %s...",
            camera_id,
        )

        engine.stop()

        with self.lock:

            self.engines.pop(
                camera_id,
                None,
            )

        logger.info(
            "Camera %s stopped.",
            camera_id,
        )

    # ========================================================
    # STOP ALL
    # ========================================================

    def stop_all(self):

        logger.info(
            "Stopping all cameras..."
        )

        with self.lock:

            camera_ids = list(
                self.engines.keys()
            )

        for camera_id in camera_ids:

            try:

                self.stop_camera(
                    camera_id
                )

            except Exception as exc:

                logger.exception(
                    "Error stopping Camera %s: %s",
                    camera_id,
                    exc,
                )

        self.running = False

        logger.info(
            "All cameras stopped."
        )

    # ========================================================
    # GET CAMERA
    # ========================================================

    def get_camera(
        self,
        camera_id,
    ):

        with self.lock:

            return self.engines.get(
                int(camera_id)
            )

    # ========================================================
    # GET ACTIVE IDS
    # ========================================================

    def get_active_camera_ids(
        self,
    ):

        with self.lock:

            return list(
                self.engines.keys()
            )

    # ========================================================
    # GET STATUS
    # ========================================================

    def get_camera_status(
        self,
        camera_id,
    ):

        engine = self.get_camera(
            camera_id
        )

        if engine is None:

            config = self.config.get(
                int(camera_id),
                {},
            )

            if not config.get(
                "enabled",
                False,
            ):

                return "DISABLED"

            return "OFFLINE"

        return engine.get_status()

    # ========================================================
    # GET ALL STATUS
    # ========================================================

    def get_all_status(
        self,
    ):

        result = {}

        for camera_id in range(
            1,
            MAX_CAMERAS + 1,
        ):

            result[
                camera_id
            ] = self.get_camera_status(
                camera_id
            )

        return result

    # ========================================================
    # GET CAMERA DATA
    # ========================================================

    def get_camera_data(
        self,
        camera_id,
    ):

        engine = self.get_camera(
            camera_id
        )

        if engine is None:

            config = self.config.get(
                int(camera_id),
                {},
            )

            return {

                "camera_id":
                    int(camera_id),

                "name":
                    config.get(
                        "name",
                        f"Camera {camera_id}",
                    ),

                "status":
                    (
                        "DISABLED"
                        if not config.get(
                            "enabled",
                            False,
                        )
                        else "OFFLINE"
                    ),

                "fps": 0.0,

                "person_count": 0,

                "min_distance": None,

                "alert_danger": False,

                "focal_length":
                    config.get(
                        "focal_length",
                        600.0,
                    ),

                "camera_height":
                    config.get(
                        "camera_height",
                        2.0,
                    ),

                "camera_pitch":
                    config.get(
                        "camera_pitch",
                        0.0,
                    ),

                "danger_distance":
                    config.get(
                        "danger_distance",
                        7.2,
                    ),

            }

        data = engine.get_data()

        data["name"] = self.config[
            int(camera_id)
        ].get(
            "name",
            f"Camera {camera_id}",
        )

        return data

    # ========================================================
    # GET ALL DATA
    # ========================================================

    def get_all_data(
        self,
    ):

        result = {}

        for camera_id in range(
            1,
            MAX_CAMERAS + 1,
        ):

            result[
                camera_id
            ] = self.get_camera_data(
                camera_id
            )

        return result

    # ========================================================
    # GET FRAME
    # ========================================================

    def get_camera_frame(
        self,
        camera_id,
    ):

        engine = self.get_camera(
            camera_id
        )

        if engine is None:

            return None

        return (
            engine.get_frame_bytes()
        )

    # ========================================================
    # RESTART CAMERA
    # ========================================================

    def restart_camera(
        self,
        camera_id,
    ):

        camera_id = int(
            camera_id
        )

        logger.info(
            "Restarting Camera %s...",
            camera_id,
        )

        self.stop_camera(
            camera_id
        )

        time.sleep(
            0.5
        )

        return self.start_camera(
            camera_id
        )
