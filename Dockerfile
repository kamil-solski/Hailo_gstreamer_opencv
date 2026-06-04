# ── Stage 1: compile OpenCV ───────────────────────────────────────────────────
FROM debian:bookworm-slim AS opencv-builder

ARG OPENCV_VERSION=4.10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git pkg-config \
    python3-dev python3-numpy \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libatlas-base-dev gfortran \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${OPENCV_VERSION} \
      https://github.com/opencv/opencv.git /opencv && \
    git clone --depth 1 --branch ${OPENCV_VERSION} \
      https://github.com/opencv/opencv_contrib.git /opencv_contrib

RUN cmake -B /opencv/build -S /opencv \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/opencv \
    -DWITH_GSTREAMER=ON \
    -DWITH_FFMPEG=OFF \
    -DBUILD_opencv_python3=ON \
    -DPYTHON3_EXECUTABLE=/usr/bin/python3 \
    -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
  && cmake --build /opencv/build -j$(nproc) \
  && cmake --install /opencv/build


# ── Stage 2: build Python venv with uv ───────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS uv-builder

# Compile .py → .pyc at install time (no startup overhead in runtime)
ENV UV_COMPILE_BYTECODE=1
# Copy files instead of hardlinks (required across Docker filesystem boundaries)
ENV UV_LINK_MODE=copy

WORKDIR /app

# Step A: install dependencies only — this layer is cached independently of
# your source code. Changing web_app.py won't re-download Flask.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

# Step B: copy source (no project install — scripts are COPYed directly in runtime stage)
COPY . .


# ── Stage 3: runtime — only artifacts, no tooling ────────────────────────────
FROM debian:bookworm-slim AS runtime

# Add the Raspberry Pi apt repository — required for the correct builds of
# gstreamer1.0-libcamera and libcamera-ipa (both carry +rpt version suffixes
# and are tuned for the RP1/CSI2 pipeline on Raspberry Pi hardware).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
       | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] \
             http://archive.raspberrypi.com/debian/ bookworm main" \
       > /etc/apt/sources.list.d/raspi.list \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-libcamera \
    libcamera-ipa \
    libatlas3-base \
  && rm -rf /var/lib/apt/lists/*

# Only the compiled output — not the 2 GB build tree
COPY --from=opencv-builder /opt/opencv /opt/opencv

# Only the finished venv — not uv, not pip, not compilers
COPY --from=uv-builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
# The uv-built venv uses /usr/local/bin/python3 (uv-bundled), which doesn't exist in
# the debian:bookworm-slim runtime.  Add site-packages to PYTHONPATH so the system
# python3 (/usr/bin/python3) can find all installed packages without relying on
# the broken venv symlink.
ENV PYTHONPATH="/opt/opencv/lib/python3.11/dist-packages:/opt/hailo:/app/.venv/lib/python3.11/site-packages:$PYTHONPATH"
ENV LD_LIBRARY_PATH="/opt/opencv/lib:$LD_LIBRARY_PATH"

# Pre-create the Hailo mount point under /opt so runc doesn't have to
# touch the read-only lower layers of /usr/lib during container init.
RUN mkdir -p /opt/hailo/hailo_platform

WORKDIR /app
COPY entrypoint.sh /entrypoint.sh
COPY Gstreamer/ ./Gstreamer/
COPY helpers.py ./helpers.py
RUN chmod +x /entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["/entrypoint.sh"]