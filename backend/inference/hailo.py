from collections import defaultdict

import cv2
import numpy as np

from .onnx import draw_bounding_box


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------------------------------------------------------
# Legacy single-output DFL decoder — yolov8n_face.hef: [1, 3549, 65]
# -----------------------------------------------------------------------------

def _make_anchor_grids(input_h=416, input_w=416):
    anchors = []
    for stride in [8, 16, 32]:
        gh, gw = input_h // stride, input_w // stride
        for gy in range(gh):
            for gx in range(gw):
                anchors.append(((gx + 0.5) * stride, (gy + 0.5) * stride, stride))
    return anchors


_ANCHOR_GRIDS_416 = _make_anchor_grids(416, 416)
_ANCHOR_GRIDS_640 = _make_anchor_grids(640, 640)


def _dfl_decode(logits16):
    """Soft-argmax over 16 DFL bins → distance in grid-cell units."""
    e = np.exp(logits16 - logits16.max())
    probs = e / e.sum()
    return float(np.dot(np.arange(16, dtype=np.float32), probs))


def _run_hailo_dfl(configured, infer_model, frame, input_h, input_w,
                   conf_threshold=0.01, nms_threshold=0.45):
    """Single-tensor raw DFL output: [1, N, 65].
       cols  0-15: DFL left,  16-31: top,  32-47: right,  48-63: bottom
       col 64: face class logit
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
        x1 = (cx_a - left)   * scale
        y1 = (cy_a - top)    * scale
        x2 = (cx_a + right)  * scale
        y2 = (cy_a + bottom) * scale
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


# -----------------------------------------------------------------------------
# NMS built-in decoder — yolov5s_personface_h8l.hef: [802] flat vector
# -----------------------------------------------------------------------------

def _run_hailo_nms(configured, infer_model, frame, input_h, input_w,
                   conf_threshold=0.25):
    """NMS built-in flat vector: [num_det, y1, x1, y2, x2, score, ...].
       raw[0]        = num_detections (float)
       raw[1 + i*5:] = [y1, x1, y2, x2, score]  (normalised 0-1)
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


# -----------------------------------------------------------------------------
# Decoupled multi-head decoder — YOLOv8/v11: 6 outputs (3 strides × box+cls)
# -----------------------------------------------------------------------------

def _run_hailo_multihead(configured, outputs_meta, frame, input_h, input_w,
                         conf_threshold=0.25, nms_threshold=0.45):
    """Decoupled head: pairs of (box [H,W,64], cls [H,W,C]) per stride.
       Vectorized — no per-proposal Python loop.
    """
    orig_h, orig_w = frame.shape[:2]
    side = max(orig_h, orig_w)
    padded = np.zeros((side, side, 3), dtype=np.uint8)
    padded[:orig_h, :orig_w] = frame
    resized = cv2.resize(padded, (input_w, input_h)).astype(np.float32) / 255.0

    # Allocate one buffer per output and run
    out_bufs = {}
    bindings = configured.create_bindings()
    bindings.input().set_buffer(resized)
    for meta in outputs_meta:
        buf = np.empty(meta.shape, dtype=np.float32)
        bindings.output(meta.name).set_buffer(buf)
        out_bufs[meta.name] = buf
    configured.run([bindings], 2000)

    # Pair outputs by spatial size: C==64 → box DFL head, else → cls head
    by_hw = defaultdict(dict)
    for meta in outputs_meta:
        h, w, c = meta.shape
        by_hw[(h, w)]["box" if c == 64 else "cls"] = out_bufs[meta.name]

    scale = side / input_w
    all_boxes, all_scores = [], []

    for (h, w), tensors in sorted(by_hw.items(), key=lambda x: -x[0][0]):
        if "box" not in tensors or "cls" not in tensors:
            continue
        box_buf = tensors["box"]  # [H, W, 64]
        cls_buf = tensors["cls"]  # [H, W, num_classes]
        stride = input_h // h

        # Anchor centres for this stride
        cx, cy = np.meshgrid(
            (np.arange(w, dtype=np.float32) + 0.5) * stride,
            (np.arange(h, dtype=np.float32) + 0.5) * stride,
        )

        # DFL decode: [H, W, 64] → softmax over 16 bins → [H, W, 4] distances
        dfl = box_buf.reshape(h, w, 4, 16)
        e = np.exp(dfl - dfl.max(axis=-1, keepdims=True))
        dist = (e / e.sum(axis=-1, keepdims=True) * np.arange(16, dtype=np.float32)).sum(axis=-1)

        x1 = (cx - dist[:, :, 0] * stride) * scale
        y1 = (cy - dist[:, :, 1] * stride) * scale
        x2 = (cx + dist[:, :, 2] * stride) * scale
        y2 = (cy + dist[:, :, 3] * stride) * scale

        scores = (1.0 / (1.0 + np.exp(-cls_buf))).max(axis=-1)  # [H, W]
        mask = scores >= conf_threshold
        if not mask.any():
            continue

        for i, j in zip(*np.where(mask)):
            bw = float(x2[i, j] - x1[i, j])
            bh = float(y2[i, j] - y1[i, j])
            if bw > 0 and bh > 0:
                all_boxes.append([float(x1[i, j]), float(y1[i, j]), bw, bh])
                all_scores.append(float(scores[i, j]))

    if not all_boxes:
        return frame, 0

    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, conf_threshold, nms_threshold)
    if len(indices) > 0 and isinstance(indices[0], (list, tuple, np.ndarray)):
        indices = [k[0] for k in indices]
    for idx in indices:
        x, y, bw, bh = all_boxes[idx]
        draw_bounding_box(frame, 0, all_scores[idx],
                          round(x), round(y), round(x + bw), round(y + bh))
    return frame, len(indices)


# -----------------------------------------------------------------------------
# Public dispatcher
# -----------------------------------------------------------------------------

def run_hailo_inference(frame, configured, infer_model, input_w, input_h,
                        hailo_fmt, conf_threshold=0.25, nms_threshold=0.45,
                        outputs_meta=None):
    if hailo_fmt == "nms":
        return _run_hailo_nms(configured, infer_model, frame, input_h, input_w,
                              conf_threshold)
    elif hailo_fmt == "dfl":
        return _run_hailo_dfl(configured, infer_model, frame, input_h, input_w,
                              conf_threshold, nms_threshold)
    else:  # multihead
        return _run_hailo_multihead(configured, outputs_meta, frame, input_h, input_w,
                                    conf_threshold, nms_threshold)
