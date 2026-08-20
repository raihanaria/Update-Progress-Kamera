# ============================================================
# SERVER MULTICAMERA
# ============================================================
#
# FastAPI backend untuk:
#   - 4 camera slots
#   - CameraManager
#   - MJPEG stream per camera
#   - WebSocket telemetry
#   - Dashboard HTML
#
# ============================================================

import asyncio
import json
import logging
import os
import signal
import sys

from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
)

from camera_manager import (
    CameraManager,
)


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = 8000

TEMPLATE_PATH = (
    "templates/index_multicam.html"
)

TELEMETRY_INTERVAL = 0.2

MJPEG_INTERVAL = 0.01


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


logger = logging.getLogger(
    "server_multicam"
)


# ============================================================
# CAMERA MANAGER
# ============================================================

camera_manager = CameraManager()


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "=========================================="
    )

    logger.info(
        "MULTICAMERA SERVER STARTING"
    )

    logger.info(
        "=========================================="
    )

    try:

        # ----------------------------------------------------
        # Start configured cameras
        # ----------------------------------------------------

        camera_manager.start_all()

        logger.info(
            "CameraManager started."
        )

        logger.info(
            "Camera status: %s",
            camera_manager.get_all_data(),
        )

        yield

    except Exception as exc:

        logger.exception(
            "Server startup error: %s",
            exc,
        )

        raise

    finally:

        logger.info(
            "Stopping CameraManager..."
        )

        try:

            camera_manager.stop_all()

        except Exception as exc:

            logger.exception(
                "CameraManager shutdown error: %s",
                exc,
            )

        logger.info(
            "MULTICAMERA SERVER STOPPED"
        )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title="Mining Safety Multicamera",

    description=(
        "Multicamera person detection "
        "using Raspberry Pi 5 + Hailo"
    ),

    lifespan=lifespan,
)


# ============================================================
# SIGNAL HANDLER
# ============================================================

def force_shutdown_handler(
    sig,
    frame,
):

    logger.info(
        "Shutdown signal received: %s",
        sig,
    )

    try:

        camera_manager.stop_all()

    except Exception as exc:

        logger.exception(
            "Error during emergency shutdown: %s",
            exc,
        )

    sys.exit(0)


signal.signal(
    signal.SIGINT,
    force_shutdown_handler,
)

signal.signal(
    signal.SIGTERM,
    force_shutdown_handler,
)


# ============================================================
# DASHBOARD
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def dashboard():

    if not os.path.exists(
        TEMPLATE_PATH
    ):

        return HTMLResponse(
            content=(
                "<h1>Dashboard template "
                "tidak ditemukan.</h1>"
            ),
            status_code=404,
        )

    try:

        with open(
            TEMPLATE_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            html_content = (
                file.read()
            )

        return HTMLResponse(
            content=html_content
        )

    except Exception as exc:

        logger.exception(
            "Failed to load dashboard: %s",
            exc,
        )

        return HTMLResponse(
            content=(
                "<h1>Failed to load dashboard</h1>"
            ),
            status_code=500,
        )


# ============================================================
# API - ALL CAMERA STATUS
# ============================================================

@app.get(
    "/api/cameras"
)
async def api_cameras():

    return {
        "success": True,

        "cameras":
            camera_manager.get_all_data(),

    }


# ============================================================
# API - SINGLE CAMERA
# ============================================================

@app.get(
    "/api/camera/{camera_id}"
)
async def api_camera(
    camera_id: int,
):

    if camera_id < 1 or camera_id > 4:

        return {
            "success": False,
            "error": "Invalid camera ID",
        }

    data = (
        camera_manager.get_camera_data(
            camera_id
        )
    )

    return {
        "success": True,
        "camera": data,
    }


# ============================================================
# API - CAMERA STATUS
# ============================================================

@app.get(
    "/api/camera/{camera_id}/status"
)
async def api_camera_status(
    camera_id: int,
):

    if camera_id < 1 or camera_id > 4:

        return {
            "success": False,
            "error": "Invalid camera ID",
        }

    return {
        "success": True,

        "camera_id":
            camera_id,

        "status":
            camera_manager.get_camera_status(
                camera_id
            ),
    }


# ============================================================
# API - RESTART CAMERA
# ============================================================

@app.post(
    "/api/camera/{camera_id}/restart"
)
async def api_restart_camera(
    camera_id: int,
):

    if camera_id < 1 or camera_id > 4:

        return {
            "success": False,
            "error": "Invalid camera ID",
        }

    logger.info(
        "Restart requested for Camera %s",
        camera_id,
    )

    success = (
        camera_manager.restart_camera(
            camera_id
        )
    )

    return {
        "success": success,

        "camera_id":
            camera_id,

        "status":
            camera_manager.get_camera_status(
                camera_id
            ),
    }


# ============================================================
# MJPEG GENERATOR
# ============================================================

async def generate_mjpeg_stream(
    camera_id: int,
):

    logger.info(
        "MJPEG client connected: Camera %s",
        camera_id,
    )

    try:

        while True:

            frame_bytes = (
                camera_manager.get_camera_frame(
                    camera_id
                )
            )

            if frame_bytes is not None:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n"
                    b"Pragma: no-cache\r\n"
                    b"\r\n"
                    + frame_bytes
                    + b"\r\n"
                )

            await asyncio.sleep(
                MJPEG_INTERVAL
            )

    except asyncio.CancelledError:

        logger.info(
            "MJPEG client disconnected: "
            "Camera %s",
            camera_id,
        )

        raise

    except Exception as exc:

        logger.exception(
            "MJPEG error Camera %s: %s",
            camera_id,
            exc,
        )


# ============================================================
# MJPEG ENDPOINT
# ============================================================

@app.get(
    "/video/{camera_id}"
)
async def video_feed(
    camera_id: int,
):

    if camera_id < 1 or camera_id > 4:

        return HTMLResponse(
            content="Invalid camera ID",
            status_code=404,
        )

    status = (
        camera_manager.get_camera_status(
            camera_id
        )
    )

    if status == "DISABLED":

        return HTMLResponse(
            content="Camera disabled",
            status_code=404,
        )

    return StreamingResponse(

        generate_mjpeg_stream(
            camera_id
        ),

        media_type=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        ),

        headers={
            "Cache-Control":
                "no-cache, no-store, must-revalidate",

            "Pragma":
                "no-cache",

            "Expires":
                "0",
        },
    )


# ============================================================
# WEBSOCKET TELEMETRY
# ============================================================

@app.websocket(
    "/ws/telemetry"
)
async def websocket_telemetry(
    websocket: WebSocket,
):

    await websocket.accept()

    logger.info(
        "Dashboard WebSocket connected."
    )

    try:

        while True:

            data = (
                camera_manager.get_all_data()
            )

            payload = {

                "success": True,

                "cameras": data,

            }

            await websocket.send_json(
                payload
            )

            await asyncio.sleep(
                TELEMETRY_INTERVAL
            )

    except WebSocketDisconnect:

        logger.info(
            "Dashboard WebSocket disconnected."
        )

    except asyncio.CancelledError:

        raise

    except Exception as exc:

        logger.exception(
            "WebSocket error: %s",
            exc,
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host=HOST,

        port=PORT,

        log_level="info",

    )
