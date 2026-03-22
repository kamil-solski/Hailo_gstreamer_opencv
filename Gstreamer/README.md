# GStreamer CSI camera on Raspberry Pi 5 (local host)

CSI camera capture and inference using GStreamer (`libcamerasrc`). Includes ONNX and Hailo-8L face detection.

## Web GUI

Stream the CSI camera in your browser (no display required):

```bash
# From project root
source ./test_opencv/bin/activate

# Enter gstreamer scripts (CSI cameras support only gestreamer for RPi 5)
cd Gstreamer

# Live view only
python web_app.py

# With ONNX face detection
python web_app.py --onnx ../yolov8n-face.onnx

# With Hailo-8L — system-provided
python web_app.py --hef /usr/share/hailo-models/yolov5s_personface_h8l.hef

# With Hailo-8L — your compiled HEF
python web_app.py --hef ../yolov8n_face.hef

# Custom port
python web_app.py --onnx ../yolov8n-face.onnx --port 8080
```

Then open **http://&lt;your-Pi-IP&gt;:5000** (or the port you chose) in a browser.

## Other scripts

- **Test_opencv_csi_onnx.py** – CSI + ONNX face detection (OpenCV window)
- **hailo_yolo_inference.py** – CSI + Hailo-8L face detection (OpenCV window)

Requires: OpenCV with GStreamer support, Flask; for Hailo: `hailort` and HEF model.
