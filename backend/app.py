#!/usr/bin/env python3
"""
CSI camera web GUI using GStreamer on Raspberry Pi 5.

Streams video from the CSI camera in the browser with optional inference.

Usage:
  python backend/app.py
  python backend/app.py --hef /models/yolov11n-face.hef
  python backend/app.py --onnx /models/yolov8n-face.onnx
  python backend/app.py --hef /models/yolov11n-face.hef --port 8080
"""

import argparse
import logging
import pathlib
import signal
import sys
from time import sleep

# Project root (one level up from backend/) — needed to import helpers.py
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# hailo_platform is installed system-wide, not inside the venv
_HAILO_PATH = "/usr/lib/python3/dist-packages"
if _HAILO_PATH not in sys.path:
    sys.path.insert(0, _HAILO_PATH)

from flask import Flask, Response, render_template

from capture import GStreamerCapture
from helpers import load_config


def create_app(capture: GStreamerCapture) -> Flask:
    _templates = pathlib.Path(__file__).parent.parent / "frontend" / "templates"
    app = Flask(__name__, template_folder=str(_templates))

    def generate():
        while True:
            data = capture.get_frame()
            if data:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                    + data + b"\r\n"
                )
            sleep(1 / capture.framerate)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/")
    def index():
        if capture.onnx_model is not None:
            mode = "ONNX face detection"
        elif capture.hailo_network_group is not None:
            mode = "Hailo-8L face detection"
        else:
            mode = "live view"
        return render_template("index.html", mode=mode)

    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Two-pass parse: extract --config first so YAML values become argparse defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(_ROOT / "config.yaml"))
    pre_args, _ = pre.parse_known_args()

    cfg = load_config(pre_args.config)
    if cfg:
        logging.info(f"Loaded config: {pre_args.config}")

    parser = argparse.ArgumentParser(description="CSI camera web GUI (GStreamer)")
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--host", help="Bind host")
    parser.add_argument("--port", type=int, help="HTTP port")
    parser.add_argument("--width", type=int, help="Frame width")
    parser.add_argument("--height", type=int, help="Frame height")
    parser.add_argument("--fps", type=int, help="Framerate")
    parser.add_argument("--onnx", default=None, help="Path to ONNX model")
    parser.add_argument("--hef", default=None, help="Path to Hailo HEF model")
    parser.set_defaults(
        host=cfg.get("host", "0.0.0.0"),
        port=cfg.get("port", 5000),
        width=cfg.get("width", 640),
        height=cfg.get("height", 480),
        fps=cfg.get("fps", 30),
    )
    args = parser.parse_args()

    if args.onnx and args.hef:
        logging.warning("Both --onnx and --hef given; using ONNX")
        args.hef = None

    capture = GStreamerCapture(
        width=args.width,
        height=args.height,
        framerate=args.fps,
        onnx_path=args.onnx,
        hef_path=args.hef,
    )

    def shutdown(_signum=None, _frame=None):
        capture.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        capture.start()
    except RuntimeError as e:
        logging.error(str(e))
        sys.exit(1)

    app = create_app(capture)
    logging.info(f"Open http://<this-machine-ip>:{args.port} in your browser")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
