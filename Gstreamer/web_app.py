#!/usr/bin/env python3
"""
CSI camera web GUI using GStreamer on Raspberry Pi 5.

Streams video from the CSI camera (GStreamer pipeline) in the browser.
Optional face detection with ONNX or Hailo-8L backend.

Usage:
  # Stream only (no inference):
  python web_app.py

  # Stream with ONNX face detection:
  python web_app.py --onnx ../yolov8n-face.onnx

  # Stream with Hailo-8L face detection:
  python web_app.py --hef ../yolov8n_face.hef

  # Custom port/host:
  python web_app.py --onnx ../yolov8n-face.onnx --port 8080
"""

import argparse
import logging
import pathlib
import signal
import sys
import threading
from time import sleep, time

# Allow importing helpers.py from the project root regardless of CWD
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from helpers import load_config

# hailo_platform is installed system-wide, not inside the venv
_HAILO_PATH = "/usr/lib/python3/dist-packages"
if _HAILO_PATH not in sys.path:
    sys.path.insert(0, _HAILO_PATH)

import cv2
import numpy as np
from flask import Flask, Response

# -----------------------------------------------------------------------------
# GStreamer pipeline (CSI camera, Pi 5)
# -----------------------------------------------------------------------------

def gstreamer_pipeline(width=640, height=480, framerate=30):
    return (
        "libcamerasrc ! "
        f"video/x-raw,width={width},height={height},framerate={framerate}/1,format=NV12 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false"
    )


# -----------------------------------------------------------------------------
# Drawing and inference helpers (shared with existing scripts)
# -----------------------------------------------------------------------------

CLASSES = ["face"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_onnx_inference(frame, model, conf_threshold=0.25, nms_threshold=0.45):
    """Run ONNX face detection and draw boxes on frame. Returns (frame, num_detections)."""
    height, width, _ = frame.shape
    length = max(height, width)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes, scores, class_ids = [], [], []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        _, maxScore, _, maxClassIndex = cv2.minMaxLoc(classes_scores)
        if maxScore >= conf_threshold:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex[1])

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    if len(result_boxes) > 0 and isinstance(result_boxes[0], list):
        result_boxes = [i[0] for i in result_boxes]

    for index in result_boxes:
        box = boxes[index]
        draw_bounding_box(
            frame,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )
    return frame, len(result_boxes)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------------------------------------------------------
# DFL decoder — used for yolov8n_face.hef raw output [1, 3549, 65]
# -----------------------------------------------------------------------------

def _make_anchor_grids(input_h=416, input_w=416):
    """Build (cx, cy, stride) for every proposal in the yolov8 head."""
    anchors = []
    for stride in [8, 16, 32]:
        gh, gw = input_h // stride, input_w // stride
        for gy in range(gh):
            for gx in range(gw):
                anchors.append(((gx + 0.5) * stride,
                                 (gy + 0.5) * stride,
                                 stride))
    return anchors  # list of (cx, cy, stride)

_ANCHOR_GRIDS_416 = _make_anchor_grids(416, 416)
_ANCHOR_GRIDS_640 = _make_anchor_grids(640, 640)


def _dfl_decode(logits16):
    """Soft-argmax over 16 DFL bins → distance in grid-cell units."""
    e = np.exp(logits16 - logits16.max())
    probs = e / e.sum()
    return float(np.dot(np.arange(16, dtype=np.float32), probs))


def _run_hailo_dfl(configured, infer_model, frame, input_h, input_w,
                   conf_threshold=0.01, nms_threshold=0.45):
    """yolov8n raw DFL output: [1, 3549, 65]
       col  0-15: DFL left,  16-31: top,  32-47: right,  48-63: bottom
       col 64: face class logit (sigmoid; note: typically low after Hailo quant)
    """
    orig_h, orig_w = frame.shape[:2]
    side = max(orig_h, orig_w)
    padded = np.zeros((side, side, 3), dtype=np.uint8)
    padded[:orig_h, :orig_w] = frame
    resized = cv2.resize(padded, (input_w, input_h)).astype(np.float32) / 255.0

    out_buf = np.empty(infer_model.output().shape, dtype=np.float32)
    bindings = configured.create_bindings()
    bindings.input().set_buffer(resized)
    bindings.output().set_buffer(out_buf)
    configured.run([bindings], 2000)

    raw = out_buf[0]  # [N, 65]
    conf = _sigmoid(raw[:, 64])
    top_mask = conf >= conf_threshold
    if top_mask.sum() == 0:
        return frame, 0

    anchors = _ANCHOR_GRIDS_416 if input_w == 416 else _ANCHOR_GRIDS_640
    scale = side / input_w
    boxes, scores = [], []

    for i in np.where(top_mask)[0]:
        cx_a, cy_a, stride = anchors[i]
        left   = _dfl_decode(raw[i,  0:16]) * stride
        top    = _dfl_decode(raw[i, 16:32]) * stride
        right  = _dfl_decode(raw[i, 32:48]) * stride
        bottom = _dfl_decode(raw[i, 48:64]) * stride
        x1 = (cx_a - left)  * scale
        y1 = (cy_a - top)   * scale
        x2 = (cx_a + right) * scale
        y2 = (cy_a + bottom)* scale
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(conf[i]))

    if not boxes:
        return frame, 0

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    if len(indices) > 0 and isinstance(indices[0], (list, tuple, np.ndarray)):
        indices = [k[0] for k in indices]
    for idx in indices:
        x, y, bw, bh = boxes[idx]
        draw_bounding_box(frame, 0, scores[idx],
                          round(x), round(y), round(x + bw), round(y + bh))
    return frame, len(indices)


def _run_hailo_nms(configured, infer_model, frame, input_h, input_w,
                   conf_threshold=0.25):
    """yolov5s NMS built-in output: [802] flat vector.
       raw[0]          = num_detections (float)
       raw[1 + i*5 : ] = [y1, x1, y2, x2, score]  (normalised 0-1 wrt input size)
    """
    orig_h, orig_w = frame.shape[:2]
    side = max(orig_h, orig_w)
    padded = np.zeros((side, side, 3), dtype=np.uint8)
    padded[:orig_h, :orig_w] = frame
    resized = cv2.resize(padded, (input_w, input_h))

    out_buf = np.empty(infer_model.output().shape, dtype=np.float32)
    bindings = configured.create_bindings()
    bindings.input().set_buffer(resized)
    bindings.output().set_buffer(out_buf)
    configured.run([bindings], 2000)

    num_det = int(out_buf[0])
    if num_det == 0:
        return frame, 0

    drawn = 0
    for i in range(num_det):
        base = 1 + i * 5
        y1_n, x1_n, y2_n, x2_n, score = out_buf[base:base + 5]
        if score < conf_threshold:
            continue
        x1 = round(float(x1_n) * side)
        y1 = round(float(y1_n) * side)
        x2 = round(float(x2_n) * side)
        y2 = round(float(y2_n) * side)
        draw_bounding_box(frame, 0, float(score), x1, y1, x2, y2)
        drawn += 1
    return frame, drawn


def run_hailo_inference(frame, configured, infer_model, input_w, input_h,
                        hailo_fmt, conf_threshold=0.25, nms_threshold=0.45):
    """Dispatch to the correct postprocessor based on hailo_fmt."""
    if hailo_fmt == "nms":
        return _run_hailo_nms(configured, infer_model, frame, input_h, input_w,
                              conf_threshold)
    else:  # "dfl"
        return _run_hailo_dfl(configured, infer_model, frame, input_h, input_w,
                              conf_threshold, nms_threshold)


# -----------------------------------------------------------------------------
# Capture thread: GStreamer -> frames -> optional inference -> JPEG buffer
# -----------------------------------------------------------------------------

class GStreamerCapture:
    def __init__(self, width=640, height=480, framerate=30, onnx_path=None, hef_path=None):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.onnx_path = onnx_path
        self.hef_path = hef_path
        self.lock = threading.Lock()
        self.frame_buffer = None  # JPEG bytes
        self.stop_event = threading.Event()
        self.cap = None
        self.onnx_model = None
        self.hailo_network_group = None
        self.hailo_expected_width = None
        self.hailo_expected_height = None

    def _init_onnx(self):
        if self.onnx_path:
            self.onnx_model = cv2.dnn.readNetFromONNX(self.onnx_path)
            logging.info(f"Loaded ONNX model: {self.onnx_path}")

    def _init_hailo(self):
        if not self.hef_path:
            return
        try:
            from hailo_platform import VDevice, FormatType
            self._hailo_vdevice = VDevice()
            infer_model = self._hailo_vdevice.create_infer_model(self.hef_path)
            self.hailo_expected_height, self.hailo_expected_width, _ = infer_model.input().shape

            out_shape = infer_model.output().shape
            if len(out_shape) == 1:
                # NMS built-in output (e.g. yolov5s_personface_h8l.hef)
                self._hailo_fmt = "nms"
                infer_model.input().set_format_type(FormatType.UINT8)
                self._hailo_conf_threshold = 0.25
                logging.info("Hailo output format: NMS built-in")
            else:
                # Raw DFL output (e.g. yolov8n_face.hef)
                self._hailo_fmt = "dfl"
                infer_model.input().set_format_type(FormatType.FLOAT32)
                # yolov8n_face.hef has poor calibration → use very low threshold
                self._hailo_conf_threshold = 0.01
                logging.warning(
                    "Hailo output format: raw DFL (yolov8n_face.hef). "
                    "Note: this HEF has low confidence scores due to suboptimal "
                    "quantization. For better results use yolov5s_personface_h8l.hef."
                )

            infer_model.output().set_format_type(FormatType.FLOAT32)
            self._hailo_infer_model = infer_model
            self._hailo_configured = infer_model.configure()
            self._hailo_configured.__enter__()
            self._hailo_configured.activate()
            self.hailo_network_group = self._hailo_configured  # sentinel
            logging.info(
                f"Hailo-8L ready: {self.hef_path} "
                f"input {self.hailo_expected_width}x{self.hailo_expected_height}"
            )
        except Exception as e:
            logging.error(f"Hailo init failed: {e}")
            self.hef_path = None
            self.hailo_network_group = None

    def start(self):
        self._init_onnx()
        self._init_hailo()

        pipeline = gstreamer_pipeline(self.width, self.height, self.framerate)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open CSI camera (GStreamer)")

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logging.info("GStreamer capture thread started")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        if getattr(self, "_hailo_configured", None):
            try:
                self._hailo_configured.deactivate()
                self._hailo_configured.__exit__(None, None, None)
            except Exception:
                pass
        if getattr(self, "_hailo_vdevice", None):
            try:
                self._hailo_vdevice.release()
            except Exception:
                pass
        logging.info("GStreamer capture stopped")

    def _capture_loop(self):
        frame_interval = 1.0 / self.framerate
        self._inference_error = None  # surface last inference error to overlay

        while not self.stop_event.is_set():
            start = time()

            # --- grab frame ---
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                logging.error(f"GStreamer read error: {e}")
                sleep(0.1)
                continue

            if not ret or frame is None:
                sleep(0.1)
                continue

            # --- inference (separate try so a crash never drops the frame) ---
            detections = 0
            if self.onnx_model is not None:
                try:
                    frame, detections = run_onnx_inference(frame, self.onnx_model)
                    self._inference_error = None
                except Exception as e:
                    self._inference_error = str(e)
                    logging.error(f"ONNX inference error: {e}")
            elif self.hailo_network_group is not None:
                try:
                    frame, detections = run_hailo_inference(
                        frame,
                        self._hailo_configured,
                        self._hailo_infer_model,
                        self.hailo_expected_width,
                        self.hailo_expected_height,
                        self._hailo_fmt,
                        conf_threshold=self._hailo_conf_threshold,
                    )
                    self._inference_error = None
                except Exception as e:
                    self._inference_error = str(e)
                    logging.error(f"Hailo inference error: {e}")

            # --- status overlay ---
            self._draw_status(frame, detections)

            # --- encode and publish ---
            try:
                _, jpeg = cv2.imencode(".jpg", frame)
                with self.lock:
                    self.frame_buffer = jpeg.tobytes()
            except Exception as e:
                logging.error(f"JPEG encode error: {e}")

            elapsed = time() - start
            sleep(max(0, frame_interval - elapsed))

    def _draw_status(self, frame, detections: int):
        """Burn a small status line into the frame so it is always visible."""
        if self.onnx_model is not None:
            backend = "ONNX"
        elif self.hailo_network_group is not None:
            backend = "Hailo"
        else:
            backend = None

        if backend is None:
            return  # no overlay when streaming only

        err = getattr(self, "_inference_error", None)
        if err:
            text = f"{backend} ERROR: {err[:60]}"
            color = (0, 0, 255)
        else:
            text = f"{backend} | detections: {detections}"
            color = (0, 255, 0)

        cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def get_frame(self):
        with self.lock:
            return self.frame_buffer


# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------

def create_app(capture: GStreamerCapture):
    app = Flask(__name__)

    def generate():
        while True:
            data = capture.get_frame()
            if data:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
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
        mode = "live view"
        if capture.onnx_model is not None:
            mode = "ONNX face detection"
        elif capture.hailo_network_group is not None:
            mode = "Hailo-8L face detection"
        return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CSI Camera – {mode}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; min-height: 100vh; }}
    .header {{ padding: 0.75rem 1rem; background: #1a1a1a; border-bottom: 1px solid #333; }}
    .header h1 {{ margin: 0; font-size: 1rem; font-weight: 600; }}
    .container {{ display: flex; justify-content: center; align-items: center; min-height: calc(100vh - 52px); padding: 1rem; }}
    .container img {{ max-width: 100%; height: auto; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>CSI Camera (GStreamer) – {mode}</h1>
  </div>
  <div class="container">
    <img src="/video_feed" alt="Camera stream" />
  </div>
</body>
</html>
"""

    return app


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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
    parser.add_argument("--onnx", default=None, help="Path to ONNX face model")
    parser.add_argument("--hef", default=None, help="Path to Hailo HEF face model")
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
