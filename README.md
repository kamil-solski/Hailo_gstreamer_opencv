# GStreamer CSI Camera — Raspberry Pi 5

CSI camera capture and optional inference via GStreamer (`libcamerasrc`) on Raspberry Pi 5.
Supports ONNX (CPU) and Hailo-8L (NPU) face detection backends.
Fully containerised — OpenCV is built with GStreamer support **inside** the Docker image.

---

The goal of this repo is to provide reproducible code that allows you to run custom models with custom execution. 
- you can choose any model you want (as long as you edit python files)
- modify output and serving of model 

If you need a setup that will run preconfigured models with very limited customization (for an app or used model), run this instead:
```bash
sudo apt install rpicam-apps
rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/hailo_yolov8_inference.json
rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/hailo_yolov5_personface.json
rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/hailo_yolov8_pose.json
```

## Project structure

```
Test_opencv/
├── Dockerfile              # multi-stage: build OpenCV+GStreamer → runtime
├── docker-compose.yml      # web-app + jupyter services
├── entrypoint.sh           # web-app entry point (reads .env, accepts model path)
├── .env                    # default values for HOST, PORT, WIDTH, HEIGHT, FPS
├── .dockerignore
├── pyproject.toml          # Python dependencies (uv)
├── uv.lock                 # locked dependency graph (commit this)
├── models/                 # model files — bind-mounted read-only, never in the image
│   ├── yolov8n-face.onnx
│   └── yolov8n_face.hef
├── notebooks/              # Jupyter notebooks — live bind-mounted
│   └── Test.ipynb
├── Gstreamer/
│   ├── web_app.py          # Flask MJPEG stream + ONNX/Hailo inference
│   ├── Test_opencv_csi_onnx.py
│   └── hailo_yolo_inference.py
└── V4l2/
    └── Test_opencv_usb_v4l2.py
```

---

## 1. Setup

### 1.1. Install docker engine on RPi5
Install docker engine
```bash
# check OS
uname --all
```
If Debian install Debian aarch64 etc.
https://docs.docker.com/engine/install/


### 1.2. Hailo-8L drivers (host — required for NPU inference)

The Hailo kernel driver must be installed on the host. It creates `/dev/hailo0`
and cannot run inside a container.

```bash
# Install drivers, firmware and userspace libraries
sudo apt install hailo-all=4.20.0

# Lock the version — if hailo-all upgrades, the bind-mounted .so versions
# in docker-compose.yml will no longer match the kernel driver and inference will break
sudo apt-mark hold hailo-all

# Reboot to load the kernel module
sudo reboot
```

Verify the NPU is recognised:

```bash
hailortcli fw-control identify
```

Expected output:

```
Executing on device: 0001:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 4.20.0 (release,app,extended context switch buffer)
Board Name: Hailo-8
Device Architecture: HAILO8L
...
Product Name: HAILO-8L AI ACC M.2 B+M KEY MODULE EXT TMP
```

Verify GStreamer Hailo plugins (installed with `hailo-all`):

```bash
gst-inspect-1.0 hailotools   # should list hailofilter, hailooverlay, etc.
gst-inspect-1.0 hailo        # should list hailonet, synchailonet
```

If either command returns nothing, clear the GStreamer registry cache and retry:

```bash
rm ~/.cache/gstreamer-1.0/registry.aarch64.bin
```

### 1.3. CSI camera

No extra packages are required on the host — `libcamera` and the GStreamer
`libcamerasrc` plugin are included inside the Docker image.

`rpicam-apps` is optional and only useful for quick host-side diagnostics
(e.g. `rpicam-hello --list-cameras` to confirm the sensor is detected).

---

## 2. Configuration
Create .env
```bash
cp. .env.example .env
```

Edit `.env` to change resolution, port, etc. These values are read by
`docker-compose.yml` and forwarded into the container as environment variables.

Download models and put inside models/:
https://drive.google.com/file/d/1s54p0ZZVR6T-q7GkeRR40PfRfNJfnGyO/view?usp=sharing
https://drive.google.com/file/d/1ufLhsNPmaOUmqfFiqLpK0thHaum8dNY1/view?usp=sharing

Models might not have best accuracy, but it is only to prove app is working

---

## 3. Quick start

```bash
# Build images — first time compiles OpenCV from source (~30–60 min on Pi 5)
docker compose build
```

### Web stream

```bash
# Stream only (no inference)
docker compose run --rm web-app

# ONNX inference (CPU)
docker compose run --rm web-app /models/yolov8n-face.onnx

# Hailo-8L inference (NPU)
docker compose run --rm web-app /models/yolov8n_face.hef
```

The model path is the only positional argument. The backend is selected
automatically from the file extension (`.onnx` → ONNX, `.hef` → Hailo).

Open in browser: **http://\<pi-ip\>:5000**

### Jupyter

```bash
docker compose up jupyter
```

Open in browser: **http://\<pi-ip\>:8888** (token: `raspberry`)

Models are available inside the notebook at `/models/`:

```python
vdevice = hailo_platform.VDevice()
infer_model = vdevice.create_infer_model("/models/yolov8n_face.hef")

# when you don't need it anymore - release model
infer_model = None   # release infer_model first
vdevice.release()    # then release the device itself
```

---

## 4. Hailo SDK in Docker

`hailo_platform` is not on PyPI. The userspace library and Python bindings
are bind-mounted from the host into both containers:

```yaml
volumes:
  - /usr/lib/python3/dist-packages/hailo_platform:/opt/hailo/hailo_platform:ro
  - /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro
  - /usr/lib/libhailort.so.4.20.0:/usr/lib/libhailort.so.4.20.0:ro
```

The bind-mount version **must match** the kernel driver version on the host.
This is guaranteed automatically as long as the host version is held with
`apt-mark hold hailo-all`.

Alternatively, place Hailo `.whl` files in `./wheels/` and uncomment
`find-links` in `pyproject.toml` to install them inside the image instead.

---

## 5. Diagnostics

Run a one-off diagnostic container (no model needed, exits immediately):

```bash
# OpenCV version and GStreamer support
docker compose run --rm web-app python3 -c \
  "import cv2; [print(l) for l in cv2.getBuildInformation().splitlines() if 'GStreamer' in l]"

# GStreamer runtime version
docker compose run --rm web-app gst-launch-1.0 --version

# Confirm libcamerasrc plugin is loaded
docker compose run --rm web-app gst-inspect-1.0 libcamerasrc

# Full version summary
docker compose run --rm web-app python3 - <<'EOF'
import cv2, numpy, sys
print(f"Python : {sys.version.split()[0]}")
print(f"OpenCV : {cv2.__version__}")
print(f"NumPy  : {numpy.__version__}")
gst = next(
    (l.strip() for l in cv2.getBuildInformation().splitlines() if "GStreamer" in l),
    "GStreamer: not found"
)
print(f"OpenCV : {gst}")
try:
    import hailort
    print(f"Hailo  : {hailort.__version__}")
except ImportError:
    print("Hailo  : not available")
EOF
```

---

## 6. Running without Docker (uv)

```bash
# Create venv and install all deps from uv.lock
uv sync

# Jupyter notebook
uv run jupyter notebook --notebook-dir=notebooks
```

> **Note:** The bare-metal `cv2` from PyPI has no GStreamer support.
> The `libcamerasrc` → `cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` pipeline
> only works inside Docker where OpenCV is compiled from source with `-DWITH_GSTREAMER=ON`.
