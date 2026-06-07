import logging
import threading
from time import sleep, time

import cv2
import numpy as np

from inference import run_hailo_inference, run_onnx_inference


def gstreamer_pipeline(width=640, height=480, framerate=30):
    return (
        "libcamerasrc ! "
        f"video/x-raw,width={width},height={height},framerate={framerate}/1,format=NV12 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false"
    )


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
            from hailo_platform import FormatType, VDevice
            self._hailo_vdevice = VDevice()
            infer_model = self._hailo_vdevice.create_infer_model(self.hef_path)
            self.hailo_expected_height, self.hailo_expected_width, _ = infer_model.input().shape

            outputs = infer_model.outputs  # list property, not callable
            if len(outputs) > 1:
                # Decoupled multi-head (YOLOv8/v11): separate box + cls tensors per stride
                self._hailo_fmt = "multihead"
                self._hailo_outputs_meta = outputs
                infer_model.input().set_format_type(FormatType.FLOAT32)
                for out in outputs:
                    out.set_format_type(FormatType.FLOAT32)
                self._hailo_conf_threshold = 0.25
                logging.info(
                    f"Hailo output format: multi-head decoupled ({len(outputs)} outputs)"
                )
            elif len(outputs[0].shape) == 1:
                # NMS built-in flat vector (e.g. yolov5s_personface_h8l.hef)
                self._hailo_fmt = "nms"
                infer_model.input().set_format_type(FormatType.UINT8)
                outputs[0].set_format_type(FormatType.FLOAT32)
                self._hailo_conf_threshold = 0.25
                logging.info("Hailo output format: NMS built-in")
            else:
                # Single-tensor raw DFL (e.g. yolov8n_face.hef [1, 3549, 65])
                self._hailo_fmt = "dfl"
                infer_model.input().set_format_type(FormatType.FLOAT32)
                outputs[0].set_format_type(FormatType.FLOAT32)
                self._hailo_conf_threshold = 0.01
                logging.warning(
                    "Hailo output format: raw DFL (single-output). "
                    "Note: yolov8n_face.hef has low confidence scores due to "
                    "suboptimal quantization."
                )

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
        self._inference_error = None

        while not self.stop_event.is_set():
            start = time()

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                logging.error(f"GStreamer read error: {e}")
                sleep(0.1)
                continue

            if not ret or frame is None:
                sleep(0.1)
                continue

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
                        outputs_meta=getattr(self, "_hailo_outputs_meta", None),
                    )
                    self._inference_error = None
                except Exception as e:
                    self._inference_error = str(e)
                    logging.error(f"Hailo inference error: {e}")

            self._draw_status(frame, detections)

            try:
                _, jpeg = cv2.imencode(".jpg", frame)
                with self.lock:
                    self.frame_buffer = jpeg.tobytes()
            except Exception as e:
                logging.error(f"JPEG encode error: {e}")

            elapsed = time() - start
            sleep(max(0, frame_interval - elapsed))

    def _draw_status(self, frame, detections: int):
        if self.onnx_model is not None:
            backend = "ONNX"
        elif self.hailo_network_group is not None:
            backend = "Hailo"
        else:
            backend = None

        if backend is None:
            return

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
