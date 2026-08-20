# ============================================================
# Safety Detection Pipeline
#
# Custom version of Hailo GStreamerDetectionApp
# WITHOUT DISPLAY_PIPELINE.
#
# Pipeline:
# RTSP -> Hailo Inference -> Tracker -> Callback -> fakesink
#
# The callback is responsible for producing the JPEG stream
# consumed by FastAPI/dashboard.
# ============================================================

from pathlib import Path

import setproctitle

from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
    get_resource_path,
    handle_list_models_flag,
    resolve_hef_path,
)

from hailo_apps.python.core.common.defines import (
    DETECTION_PIPELINE,
    DETECTION_POSTPROCESS_FUNCTION,
    DETECTION_POSTPROCESS_SO_FILENAME,
    RESOURCES_SO_DIR_NAME,
)

from hailo_apps.python.core.common.hef_utils import (
    get_hef_labels_json,
)

from hailo_apps.python.core.common.hailo_logger import (
    get_logger,
)

from hailo_apps.python.core.gstreamer.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)

from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
)


hailo_logger = get_logger(__name__)


class SafetyDetectionApp(GStreamerApp):
    """
    Custom Hailo detection application untuk safety monitoring.

    Perbedaan utama dengan GStreamerDetectionApp bawaan Hailo:

        Hailo original:
            SOURCE
              ↓
            INFERENCE
              ↓
            TRACKER
              ↓
            CALLBACK
              ↓
            DISPLAY_PIPELINE

        SafetyDetectionApp:
            SOURCE
              ↓
            INFERENCE
              ↓
            TRACKER
              ↓
            CALLBACK
              ↓
            FAKESINK

    Tidak ada autovideosink / ximagesink / fpsdisplaysink.
    """

    def __init__(self, app_callback, user_data, parser=None):

        if parser is None:
            parser = get_pipeline_parser()

        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to custom labels JSON file",
        )

        # Handle --list-models sebelum initialization penuh
        handle_list_models_flag(
            parser,
            DETECTION_PIPELINE
        )

        hailo_logger.info(
            "Initializing Safety Detection App..."
        )

        # Inisialisasi GStreamerApp bawaan Hailo
        super().__init__(
            parser,
            user_data
        )

        hailo_logger.info(
            "Safety GStreamerApp initialized | "
            "arch=%s | input=%s | fps=%s",
            self.arch,
            self.video_source,
            self.frame_rate,
        )

        # ----------------------------------------------------
        # HAILO DETECTION CONFIGURATION
        # ----------------------------------------------------

        # Untuk pipeline single-camera kita tetap gunakan
        # konfigurasi detection dari Hailo.
        if self.batch_size == 1:
            self.batch_size = 2

        nms_score_threshold = 0.6
        nms_iou_threshold = 0.45

        # ----------------------------------------------------
        # HEF
        # ----------------------------------------------------

        self.hef_path = resolve_hef_path(
            self.hef_path,
            app_name=DETECTION_PIPELINE,
            arch=self.arch,
        )

        # ----------------------------------------------------
        # POST PROCESS
        # ----------------------------------------------------

        self.post_process_so = get_resource_path(
            DETECTION_PIPELINE,
            RESOURCES_SO_DIR_NAME,
            self.arch,
            DETECTION_POSTPROCESS_SO_FILENAME,
        )

        self.post_function_name = (
            DETECTION_POSTPROCESS_FUNCTION
        )

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        self.labels_json = (
            self.options_menu.labels_json
        )

        if self.labels_json is None:

            self.labels_json = get_hef_labels_json(
                self.hef_path
            )

            if self.labels_json is not None:
                hailo_logger.info(
                    "Auto detected Labels JSON: %s",
                    self.labels_json,
                )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if (
            self.hef_path is None
            or not Path(self.hef_path).exists()
        ):
            hailo_logger.error(
                "HEF path is invalid or missing: %s",
                self.hef_path,
            )

        if (
            self.post_process_so is None
            or not Path(self.post_process_so).exists()
        ):
            hailo_logger.error(
                "Post-process .so path is invalid or missing: %s",
                self.post_process_so,
            )

        # ----------------------------------------------------
        # CALLBACK
        # ----------------------------------------------------

        self.app_callback = app_callback

        # ----------------------------------------------------
        # NMS PARAMETERS
        # ----------------------------------------------------

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        hailo_logger.debug(
            "Postprocess thresholds: %s",
            self.thresholds_str,
        )

        # Process title
        setproctitle.setproctitle(
            "Hailo Safety Detection"
        )

        # ----------------------------------------------------
        # CREATE PIPELINE
        # ----------------------------------------------------

        self.create_pipeline()

        hailo_logger.info(
            "Safety detection pipeline created successfully."
        )

    # ========================================================
    # CUSTOM PIPELINE
    # ========================================================

    def get_pipeline_string(self):

        # ----------------------------------------------------
        # 1. SOURCE
        # ----------------------------------------------------

        source_pipeline = (
            self.get_source_pipeline()
        )

        # ----------------------------------------------------
        # 2. HAILO INFERENCE
        # ----------------------------------------------------

        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str,
        )

        detection_pipeline_wrapper = (
            INFERENCE_PIPELINE_WRAPPER(
                detection_pipeline
            )
        )

        # ----------------------------------------------------
        # 3. TRACKER
        # ----------------------------------------------------

        tracker_pipeline = TRACKER_PIPELINE(
            class_id=1
        )

        # ----------------------------------------------------
        # 4. USER CALLBACK
        # ----------------------------------------------------

        user_callback_pipeline = (
            USER_CALLBACK_PIPELINE()
        )

        # ----------------------------------------------------
        # 5. NO DISPLAY
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # GStreamerDetectionApp bawaan Hailo melakukan:
        #
        #     ... ! USER_CALLBACK ! DISPLAY_PIPELINE
        #
        # DISPLAY_PIPELINE inilah yang menghasilkan
        # window output Hailo.
        #
        # Kita sengaja TIDAK menggunakan DISPLAY_PIPELINE.
        #
        # Callback tetap mendapatkan buffer dan melakukan
        # processing / JPEG encoding.
        #
        # Setelah callback selesai, buffer dibuang oleh
        # fakesink.
        #
        # ----------------------------------------------------

        sink_pipeline = (
            "fakesink name=safety_fakesink "
            "sync=false "
            "async=false"
        )

        # ----------------------------------------------------
        # FINAL PIPELINE
        # ----------------------------------------------------

        pipeline_string = (
            f"{source_pipeline} ! "
            f"{detection_pipeline_wrapper} ! "
            f"{tracker_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{sink_pipeline}"
        )

        hailo_logger.info(
            "Safety pipeline created WITHOUT display."
        )

        hailo_logger.debug(
            "Pipeline string: %s",
            pipeline_string,
        )

        return pipeline_string


# ============================================================
# OPTIONAL STANDALONE TEST
# ============================================================

def main():

    user_data = app_callback_class()

    app = SafetyDetectionApp(
        dummy_callback,
        user_data,
    )

    app.run()


if __name__ == "__main__":
    main()
