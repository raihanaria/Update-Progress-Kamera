# ============================================================
# CAMERA CONFIGURATION
# ============================================================

CAMERA_CONFIG = {

    1: {
        "enabled": True,

        "name": "Dahua",

        "rtsp_url": (
            "rtsp://admin:imam1977@192.168.1.50:554/"
            "cam/realmonitor?channel=1&subtype=1"
        ),

        # Calibration
        "focal_length": 823.33,
        "camera_height": 1.95,
        "camera_pitch": 23.0,

        # Safety
        "danger_distance": 7.2,
    },

    2: {
        "enabled": True,

        "name": "Hikvision",

        "rtsp_url": (
            "rtsp://admin:_herrscherindo77@192.168.1.64:554/"
            "Streaming/channels/102"
        ),

        # Calibration
        "focal_length": 823.3,
        "camera_height": 1.95,
        "camera_pitch": 23.0,

        # Safety
        "danger_distance": 7.2,
    },

    3: {
        "enabled": False,

        "name": "Camera 3",

        "rtsp_url": "",

        # Calibration
        "focal_length": 600.0,
        "camera_height": 2.00,
        "camera_pitch": 20.0,

        # Safety
        "danger_distance": 7.2,
    },

    4: {
        "enabled": False,

        "name": "Camera 4",

        "rtsp_url": "",

        # Calibration
        "focal_length": 600.0,
        "camera_height": 2.00,
        "camera_pitch": 20.0,

        # Safety
        "danger_distance": 7.2,
    },
}


MAX_CAMERAS = 4
