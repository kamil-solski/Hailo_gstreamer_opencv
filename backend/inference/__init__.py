from .hailo import run_hailo_inference
from .onnx import run_onnx_inference

__all__ = ["run_onnx_inference", "run_hailo_inference"]
