import cv2
import numpy as np

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
