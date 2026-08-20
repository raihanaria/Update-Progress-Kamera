# ============================================================
# detection_multicam.py
# ============================================================
#
# Multi-camera Hailo detection engine
#
# Features:
#   - RTSP input
#   - YOLOv11n HEF
#   - Hailo inference
#   - No Hailo display window
#   - Latest JPEG frame only
#   - Per-camera calibration
#   - Per-camera danger distance
#   - Person detection
#   - Distance estimation
#   - Tracker-based distance smoothing
#   - Telemetry
#
# ============================================================


# ============================================================
# STANDARD LIBRARY
# ============================================================

import os
import sys
import time
import math
import threading
import atexit

from contextlib import contextmanager


# ============================================================
# GSTREAMER ENVIRONMENT
# ============================================================

# Prevent Hailo display window.
os.environ["GST_VIDEO_SINK"] = "fakesink"

# Disable VAAPI decoder.
os.environ[
    "GST_PLUGIN_FEATURE_RANK"
] = "vaapidecodebin:NONE"


# ============================================================
# GI / GST
# ============================================================

import gi

gi.require_version(
    "Gst",
    "1.0",
)

from gi.repository import Gst


# ============================================================
# OPENCV
# ============================================================

import cv2


# ============================================================
# HAILO
# ============================================================

import hailo


# ============================================================
# HAILO HELPERS
# ============================================================

from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

from hailo_apps.python.core.common.hailo_logger import (
    get_logger,
)

from hailo_apps.python.core.gstreamer.gstreamer_app import (
    app_callback_class,
)


# ============================================================
# CUSTOM SAFETY DETECTION APP
# ============================================================

from safety_detection_pipeline import (
    SafetyDetectionApp,
)


# ============================================================
# AUDIO
# ============================================================

from audioalert import (
    AudioAlertManager,
)


# ============================================================
# LOGGER
# ============================================================

hailo_logger = get_logger(
    __name__
)


# ============================================================
# GST INIT
# ============================================================

Gst.init(None)


# ============================================================
# YOLO HEF
# ============================================================

YOLO_HEF_PATH = (
    "/usr/local/hailo/resources/models/"
    "hailo8/yolov11n.hef"
)


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

DEFAULT_FOCAL_LENGTH = 600.0

DEFAULT_CAMERA_HEIGHT = 3.0

DEFAULT_CAMERA_PITCH = 0.0

DEFAULT_DANGER_DISTANCE = 7.2


# ============================================================
# DISTANCE SMOOTHER
# ============================================================

DEFAULT_SMOOTHER_ALPHA = 0.3

DEFAULT_SMOOTHER_TTL = 5.0


# ============================================================
# JPEG
# ============================================================

DEFAULT_JPEG_QUALITY = 70


# ============================================================
# PIPELINE
# ============================================================

PIPELINE_LATENCY_MS = 50

FIRST_FRAME_TIMEOUT = 15.0


# ============================================================
# TEMPORARY ARGV
# ============================================================

@contextmanager
def hailo_argv_for_camera(rtsp_url):
    """
    GStreamerDetectionApp menggunakan argparse.

    Kita memberikan:
        --input
        --hef-path

    Kita TIDAK memberikan:
        --use-frame

    sehingga Hailo tidak membuka display window.
    """

    original_argv = sys.argv.copy()

    try:

        program_name = (
            original_argv[0]
            if original_argv
            else "detection_multicam.py"
        )

        sys.argv = [
            program_name,
            "--input",
            rtsp_url,
            "--hef-path",
            YOLO_HEF_PATH,
        ]

        yield

    finally:

        sys.argv = original_argv


# ============================================================
# DISTANCE ESTIMATION
# ============================================================

def estimate_distance_ground(
    ymax_pixel,
    img_height,
    focal_length,
    camera_height,
    camera_pitch,
):
    """
    Estimate ground distance using:
        - focal length
        - camera height
        - camera pitch
        - person's bounding-box bottom
    """

    try:

        if ymax_pixel is None:
            return 99.0

        if img_height is None:
            return 99.0

        if img_height <= 0:
            return 99.0

        if focal_length <= 0:
            return 99.0

        if camera_height <= 0:
            return 99.0

        ymax_pixel = float(
            ymax_pixel
        )

        img_height = float(
            img_height
        )

        focal_length = float(
            focal_length
        )

        camera_height = float(
            camera_height
        )

        camera_pitch = float(
            camera_pitch
        )

        # ----------------------------------------------------
        # Optical center
        # ----------------------------------------------------

        cy = img_height / 2.0

        # ----------------------------------------------------
        # Pixel angle relative to optical axis
        # ----------------------------------------------------

        angle_offset_rad = math.atan(
            (ymax_pixel - cy)
            / focal_length
        )

        # ----------------------------------------------------
        # Camera pitch
        # ----------------------------------------------------

        pitch_rad = math.radians(
            camera_pitch
        )

        # ----------------------------------------------------
        # Total angle
        # ----------------------------------------------------

        total_angle = (
            pitch_rad
            + angle_offset_rad
        )

        # ----------------------------------------------------
        # Invalid geometry
        # ----------------------------------------------------

        if total_angle <= 0:
            return 99.0

        # ----------------------------------------------------
        # Ground distance
        # ----------------------------------------------------

        distance = (
            camera_height
            / math.tan(total_angle)
        )

        if not math.isfinite(
            distance
        ):
            return 99.0

        return max(
            0.1,
            distance,
        )

    except Exception as exc:

        hailo_logger.debug(
            "Distance estimation error: %s",
            exc,
        )

        return 99.0


# ============================================================
# DISTANCE SMOOTHER
# ============================================================

class DistanceSmoother:

    def __init__(
        self,
        alpha=DEFAULT_SMOOTHER_ALPHA,
    ):

        self.alpha = float(
            alpha
        )

        self.smooth_dist = None

    def update(
        self,
        raw_distance,
    ):

        if self.smooth_dist is None:

            self.smooth_dist = (
                raw_distance
            )

        elif raw_distance >= 99.0:

            return self.smooth_dist

        else:

            self.smooth_dist = (

                self.alpha
                * raw_distance

            ) + (

                (1.0 - self.alpha)
                * self.smooth_dist

            )

        return self.smooth_dist


# ============================================================
# USER DATA
# ============================================================

class MultiCameraUserData(
    app_callback_class
):

    def __init__(
        self,
        camera_id,
        focal_length,
        camera_height,
        camera_pitch,
        danger_distance,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Camera identity
        # ----------------------------------------------------

        self.camera_id = int(
            camera_id
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        self.focal_length = float(
            focal_length
        )

        self.camera_height = float(
            camera_height
        )

        self.camera_pitch = float(
            camera_pitch
        )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        self.danger_distance = float(
            danger_distance
        )

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        self.person_count = 0

        self.min_distance = 99.0

        self.alert_danger = False

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.fps = 0.0

        self.prev_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Latest JPEG
        # ----------------------------------------------------

        self.current_jpeg = None

        self.frame_lock = (
            threading.Lock()
        )

        # ----------------------------------------------------
        # Frame status
        # ----------------------------------------------------

        self.frame_received = False

        self.first_frame_time = None

        # ----------------------------------------------------
        # Distance smoothers
        # ----------------------------------------------------

        self.smoothers = {}

        self.smoother_ttl_sec = (
            DEFAULT_SMOOTHER_TTL
        )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        self.audio_alert = (
            AudioAlertManager(
                sound_path="beep.mp3",
                cooldown_seconds=1.5,
            )
        )


# ============================================================
# HAILO CALLBACK
# ============================================================

def app_callback(
    element,
    buffer,
    user_data,
):
    """
    Hailo callback.

    IMPORTANT:
    current_jpeg hanya menyimpan frame TERBARU.
    Tidak ada frame queue.
    """

    if buffer is None:

        return (
            Gst.PadProbeReturn.OK
        )

    try:

        # ====================================================
        # TIME
        # ====================================================

        current_time = (
            time.perf_counter()
        )

        # ====================================================
        # FIRST FRAME
        # ====================================================

        if not user_data.frame_received:

            user_data.frame_received = True

            user_data.first_frame_time = (
                time.time()
            )

            hailo_logger.info(
                "Camera %s received first frame.",
                user_data.camera_id,
            )

        # ====================================================
        # FPS
        # ====================================================

        delta_time = (
            current_time
            - user_data.prev_time
        )

        user_data.prev_time = (
            current_time
        )

        if delta_time > 0:

            current_fps = (
                1.0
                / delta_time
            )

            if user_data.fps <= 0:

                user_data.fps = (
                    current_fps
                )

            else:

                user_data.fps = (

                    0.9
                    * user_data.fps

                ) + (

                    0.1
                    * current_fps

                )

        # ====================================================
        # CAPS
        # ====================================================

        pad = (
            element.get_static_pad(
                "src"
            )
        )

        if pad is None:

            return (
                Gst.PadProbeReturn.OK
            )

        format_str, width, height = (
            get_caps_from_pad(
                pad
            )
        )

        if width is None:
            width = 1280

        if height is None:
            height = 720

        # ====================================================
        # FRAME
        # ====================================================

        frame = (
            get_numpy_from_buffer(
                buffer,
                format_str,
                width,
                height,
            )
        )

        frame_bgr = None

        if frame is not None:

            try:

                frame_bgr = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR,
                )

            except Exception as exc:

                hailo_logger.warning(
                    "Camera %s frame conversion error: %s",
                    user_data.camera_id,
                    exc,
                )

        # ====================================================
        # HAILO ROI
        # ====================================================

        roi = (
            hailo.get_roi_from_buffer(
                buffer
            )
        )

        # ====================================================
        # DETECTIONS
        # ====================================================

        detections = (
            roi.get_objects_typed(
                hailo.HAILO_DETECTION
            )
        )

        person_count = 0

        min_distance = 99.0

        is_danger = False

        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        for detection in detections:

            try:

                label = (
                    detection.get_label()
                )

                confidence = (
                    detection.get_confidence()
                )

                # ------------------------------------------------
                # PERSON ONLY
                # ------------------------------------------------

                if label != "person":

                    continue

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                if confidence < 0.40:

                    continue

                person_count += 1

                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                bbox = (
                    detection.get_bbox()
                )

                xmin = int(
                    bbox.xmin()
                    * width
                )

                ymin = int(
                    bbox.ymin()
                    * height
                )

                xmax = int(
                    bbox.xmax()
                    * width
                )

                ymax = int(
                    bbox.ymax()
                    * height
                )

                # ------------------------------------------------
                # DISTANCE
                # ------------------------------------------------

                distance_raw = (
                    estimate_distance_ground(

                        ymax_pixel=ymax,

                        img_height=height,

                        focal_length=(
                            user_data.focal_length
                        ),

                        camera_height=(
                            user_data.camera_height
                        ),

                        camera_pitch=(
                            user_data.camera_pitch
                        ),

                    )
                )

                # =================================================
                # TRACK ID
                # =================================================
                #
                # IMPORTANT:
                #
                # get_objects_typed() returns LIST.
                #
                # Correct:
                #
                #   [0].get_id()
                #
                # NOT:
                #
                #   track_id = get_objects_typed(...)
                #
                # =================================================

                track_id = None

                try:

                    track_objects = (
                        detection.get_objects_typed(
                            hailo.HAILO_UNIQUE_ID
                        )
                    )

                    if track_objects:

                        track_id = (
                            track_objects[0].get_id()
                        )

                except Exception as exc:

                    hailo_logger.debug(
                        "Camera %s tracker ID unavailable: %s",
                        user_data.camera_id,
                        exc,
                    )

                    track_id = None

                # =================================================
                # DISTANCE SMOOTHING
                # =================================================

                if track_id is not None:

                    entry = (
                        user_data.smoothers.get(
                            track_id
                        )
                    )

                    if entry is None:

                        entry = {

                            "smoother":
                                DistanceSmoother(
                                    alpha=(
                                        DEFAULT_SMOOTHER_ALPHA
                                    )
                                ),

                            "last_seen":
                                time.time(),

                        }

                        user_data.smoothers[
                            track_id
                        ] = entry

                    entry[
                        "last_seen"
                    ] = time.time()

                    distance = (
                        entry[
                            "smoother"
                        ].update(
                            distance_raw
                        )
                    )

                else:

                    distance = (
                        distance_raw
                    )

                # =================================================
                # MIN DISTANCE
                # =================================================

                if (
                    distance
                    < min_distance
                ):

                    min_distance = (
                        distance
                    )

                # =================================================
                # DANGER
                # =================================================

                if (
                    distance
                    <= user_data.danger_distance
                ):

                    is_danger = True

                # =================================================
                # DRAW
                # =================================================

                if frame_bgr is not None:

                    if is_danger:

                        color = (
                            0,
                            0,
                            255,
                        )

                    else:

                        color = (
                            0,
                            255,
                            0,
                        )

                    cv2.rectangle(

                        frame_bgr,

                        (
                            xmin,
                            ymin,
                        ),

                        (
                            xmax,
                            ymax,
                        ),

                        color,

                        2,

                    )

                    label_text = (
                        f"Person "
                        f"{distance:.1f}m"
                    )

                    cv2.putText(

                        frame_bgr,

                        label_text,

                        (
                            xmin,
                            max(
                                ymin - 10,
                                20,
                            ),
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.6,

                        color,

                        2,

                    )

            except Exception as exc:

                hailo_logger.warning(
                    "Camera %s detection processing error: %s",
                    user_data.camera_id,
                    exc,
                )

        # ====================================================
        # JPEG
        # ====================================================

        if frame_bgr is not None:

            try:

                success, encoded = (
                    cv2.imencode(

                        ".jpg",

                        frame_bgr,

                        [

                            int(
                                cv2.IMWRITE_JPEG_QUALITY
                            ),

                            DEFAULT_JPEG_QUALITY,

                        ],

                    )
                )

                if success:

                    jpeg_bytes = (
                        encoded.tobytes()
                    )

                    # --------------------------------------------
                    # IMPORTANT:
                    # overwrite old frame
                    # --------------------------------------------

                    with (
                        user_data.frame_lock
                    ):

                        user_data.current_jpeg = (
                            jpeg_bytes
                        )

            except Exception as exc:

                hailo_logger.warning(
                    "Camera %s JPEG encode error: %s",
                    user_data.camera_id,
                    exc,
                )

        # ====================================================
        # CLEAN OLD TRACKERS
        # ====================================================

        if user_data.smoothers:

            now = time.time()

            stale_ids = [

                track_id

                for track_id, entry

                in user_data.smoothers.items()

                if (

                    now
                    - entry["last_seen"]

                ) > user_data.smoother_ttl_sec

            ]

            for track_id in stale_ids:

                del user_data.smoothers[
                    track_id
                ]

        # ====================================================
        # TELEMETRY
        # ====================================================

        user_data.person_count = (
            person_count
        )

        if person_count > 0:

            user_data.min_distance = (
                min_distance
            )

        else:

            user_data.min_distance = (
                99.0
            )

        user_data.alert_danger = (
            is_danger
        )

        # ====================================================
        # AUDIO
        # ====================================================

        if (
            is_danger
            and user_data.audio_alert
        ):

            try:

                user_data.audio_alert.play_alert()

            except Exception as exc:

                hailo_logger.warning(
                    "Camera %s audio error: %s",
                    user_data.camera_id,
                    exc,
                )

    except Exception as exc:

        # ====================================================
        # GLOBAL CALLBACK ERROR
        # ====================================================

        hailo_logger.warning(
            "Camera %s callback error: %s",
            user_data.camera_id,
            exc,
        )

    return (
        Gst.PadProbeReturn.OK
    )


# ============================================================
# CAMERA ENGINE
# ============================================================

class SafetyDetectorEngine:

    """
    Satu instance SafetyDetectorEngine
    = satu kamera RTSP.
    """

    def __init__(
        self,
        camera_id,
        rtsp_url,
        focal_length=DEFAULT_FOCAL_LENGTH,
        camera_height=DEFAULT_CAMERA_HEIGHT,
        camera_pitch=DEFAULT_CAMERA_PITCH,
        danger_distance=DEFAULT_DANGER_DISTANCE,
    ):

        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        self.camera_id = int(
            camera_id
        )

        self.rtsp_url = (
            rtsp_url
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        self.focal_length = float(
            focal_length
        )

        self.camera_height = float(
            camera_height
        )

        self.camera_pitch = float(
            camera_pitch
        )

        self.danger_distance = float(
            danger_distance
        )

        # ----------------------------------------------------
        # User data
        # ----------------------------------------------------

        self.user_data = (
            MultiCameraUserData(

                camera_id=self.camera_id,

                focal_length=(
                    self.focal_length
                ),

                camera_height=(
                    self.camera_height
                ),

                camera_pitch=(
                    self.camera_pitch
                ),

                danger_distance=(
                    self.danger_distance
                ),

            )
        )

        # ----------------------------------------------------
        # Hailo app
        # ----------------------------------------------------

        self.app = None

        # ----------------------------------------------------
        # Thread
        # ----------------------------------------------------

        self.thread = None

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.is_running = False

        self.pipeline_started = False

        self.start_error = None

        self._stop_requested = False

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        atexit.register(
            self.stop
        )

        hailo_logger.info(

            "Created Camera %s | "
            "RTSP=%s | "
            "HEF=%s | "
            "focal=%.2f | "
            "height=%.2f | "
            "pitch=%.2f | "
            "danger=%.2f",

            self.camera_id,

            self.rtsp_url,

            YOLO_HEF_PATH,

            self.focal_length,

            self.camera_height,

            self.camera_pitch,

            self.danger_distance,

        )

    # ========================================================
    # START
    # ========================================================

    def start(self):

        if self.is_running:

            return True

        if not self.rtsp_url:

            self.start_error = (
                "RTSP URL is empty"
            )

            return False

        if not os.path.isfile(
            YOLO_HEF_PATH
        ):

            self.start_error = (

                "YOLOv11n HEF not found: "
                + YOLO_HEF_PATH

            )

            hailo_logger.error(
                self.start_error
            )

            return False

        self._stop_requested = False

        self.start_error = None

        hailo_logger.info(
            "Starting Camera %s...",
            self.camera_id,
        )

        try:

            # ------------------------------------------------
            # Create Hailo application
            # ------------------------------------------------

            with hailo_argv_for_camera(
                self.rtsp_url
            ):

                self.app = (
                    SafetyDetectionApp(

                        app_callback,

                        self.user_data,

                    )
                )

            # ------------------------------------------------
            # Reduce GStreamer latency
            # ------------------------------------------------

            if hasattr(
                self.app,
                "pipeline_latency",
            ):

                self.app.pipeline_latency = (
                    PIPELINE_LATENCY_MS
                )

            # ------------------------------------------------
            # Thread
            # ------------------------------------------------

            self.thread = (
                threading.Thread(

                    target=self._run_app,

                    name=(
                        f"hailo-camera-"
                        f"{self.camera_id}"
                    ),

                    daemon=True,

                )
            )

            self.is_running = True

            self.thread.start()

            hailo_logger.info(
                "Camera %s detection thread started.",
                self.camera_id,
            )

            # ------------------------------------------------
            # Wait first frame
            # ------------------------------------------------

            wait_start = (
                time.time()
            )

            while (

                time.time()
                - wait_start

                < FIRST_FRAME_TIMEOUT

            ):

                if (
                    self.user_data.frame_received
                ):

                    self.pipeline_started = True

                    hailo_logger.info(
                        "Camera %s ONLINE.",
                        self.camera_id,
                    )

                    return True

                if not self.is_running:

                    break

                time.sleep(
                    0.1
                )

            # ------------------------------------------------
            # Timeout
            # ------------------------------------------------

            if not (
                self.user_data.frame_received
            ):

                self.start_error = (

                    "No frame received "
                    f"within "
                    f"{FIRST_FRAME_TIMEOUT}s"

                )

                hailo_logger.error(
                    "Camera %s: %s",
                    self.camera_id,
                    self.start_error,
                )

                self.stop()

                return False

            return True

        except Exception as exc:

            self.start_error = (
                str(exc)
            )

            hailo_logger.exception(
                "Camera %s startup failed.",
                self.camera_id,
            )

            self.stop()

            return False

    # ========================================================
    # RUN
    # ========================================================

    def _run_app(self):

        try:

            hailo_logger.info(
                "Camera %s Hailo pipeline running.",
                self.camera_id,
            )

            self.app.run()

        except SystemExit:

            hailo_logger.debug(
                "Camera %s Hailo app exited.",
                self.camera_id,
            )

        except Exception as exc:

            self.start_error = (
                str(exc)
            )

            hailo_logger.exception(
                "Camera %s pipeline error.",
                self.camera_id,
            )

        finally:

            self.is_running = False

            hailo_logger.info(
                "Camera %s detection engine stopped.",
                self.camera_id,
            )

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        if (

            self.app is None
            and not self.is_running

        ):

            return

        hailo_logger.info(
            "Stopping Camera %s...",
            self.camera_id,
        )

        self._stop_requested = True

        self.is_running = False

        try:

            if self.app is not None:

                # ------------------------------------------------
                # Main loop
                # ------------------------------------------------

                if (

                    hasattr(
                        self.app,
                        "loop",
                    )

                    and self.app.loop

                ):

                    try:

                        if (
                            self.app.loop.is_running()
                        ):

                            self.app.loop.quit()

                    except Exception:

                        pass

                # ------------------------------------------------
                # Pipeline
                # ------------------------------------------------

                if (

                    hasattr(
                        self.app,
                        "pipeline",
                    )

                    and self.app.pipeline

                ):

                    try:

                        self.app.pipeline.set_state(
                            Gst.State.NULL
                        )

                    except Exception as exc:

                        hailo_logger.warning(
                            "Camera %s pipeline stop error: %s",
                            self.camera_id,
                            exc,
                        )

                    try:

                        self.app.pipeline.get_state(
                            2 * Gst.SECOND
                        )

                    except Exception:

                        pass

        except Exception as exc:

            hailo_logger.exception(
                "Camera %s cleanup error: %s",
                self.camera_id,
                exc,
            )

        finally:

            self.app = None

            self.thread = None

            self.pipeline_started = False

            self.is_running = False

            hailo_logger.info(
                "Camera %s resources released.",
                self.camera_id,
            )

    # ========================================================
    # FRAME
    # ========================================================

    def get_frame_bytes(self):

        with (
            self.user_data.frame_lock
        ):

            return (
                self.user_data.current_jpeg
            )

    # ========================================================
    # COMPATIBILITY
    # ========================================================

    def get_current_jpeg(self):

        return (
            self.get_frame_bytes()
        )

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        if (
            self.user_data.frame_received
        ):

            if self.is_running:

                return "ONLINE"

            return "OFFLINE"

        if self.start_error:

            return "ERROR"

        if self.is_running:

            return "CONNECTING"

        return "OFFLINE"

    # ========================================================
    # TELEMETRY
    # ========================================================

    def get_data(self):

        min_distance = None

        if (
            self.user_data.person_count
            > 0
        ):

            min_distance = round(
                self.user_data.min_distance,
                2,
            )

        return {

            "camera_id":
                self.camera_id,

            "status":
                self.get_status(),

            "fps":
                round(
                    self.user_data.fps,
                    1,
                ),

            "person_count":
                self.user_data.person_count,

            "min_distance":
                min_distance,

            "alert_danger":
                self.user_data.alert_danger,

            "danger_distance":
                self.danger_distance,

            "focal_length":
                self.focal_length,

            "camera_height":
                self.camera_height,

            "camera_pitch":
                self.camera_pitch,

            "frame_received":
                self.user_data.frame_received,

        }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # CAMERA TEST
    # ========================================================
    #
    # DAHUA
    #

    TEST_RTSP = (
        #"rtsp://admin:imam1977@192.168.1.50:554/"
        #"cam/realmonitor?channel=1&subtype=0"
        "rtsp://admin:_herrscherindo77@192.168.1.64:554/"
        "Streaming/channels/101"
    )

    #
    # HIKVISION
    #
    # Jika ingin test Hikvision, comment TEST_RTSP
    # Dahua di atas lalu gunakan:
    #
    # TEST_RTSP = (
    #     "rtsp://admin:PASSWORD@192.168.1.64:554/"
    #     "Streaming/channels/101"
    # )
    #

    # ========================================================
    # CAMERA CONFIG
    # ========================================================

    engine = (
        SafetyDetectorEngine(

            camera_id=1,

            rtsp_url=TEST_RTSP,

            focal_length=823.33,

            camera_height=1.95,

            camera_pitch=23.0,

            danger_distance=7.2,

        )
    )

    # ========================================================
    # START
    # ========================================================

    try:

        print()
        print(
            "=========================================="
        )

        print(
            "LOW LATENCY HAILO CAMERA TEST"
        )

        print(
            "=========================================="
        )

        print(
            "Camera ID :",
            engine.camera_id,
        )

        print(
            "RTSP      :",
            engine.rtsp_url,
        )

        print(
            "HEF       :",
            YOLO_HEF_PATH,
        )

        print(
            "Latency   :",
            PIPELINE_LATENCY_MS,
            "ms",
        )

        print(
            "=========================================="
        )

        print()

        started = (
            engine.start()
        )

        print(
            "START:",
            started,
        )

        print(
            "STATUS:",
            engine.get_status(),
        )

        if not started:

            print(
                "ERROR:",
                engine.start_error,
            )

            sys.exit(1)

        # ====================================================
        # MONITOR
        # ====================================================

        while True:

            time.sleep(
                1.0
            )

            print(
                engine.get_data()
            )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping..."
        )

    finally:

        engine.stop()

        print(
            "Camera stopped."
        )
