import cv2
import numpy as np

# Define the class for face detection
onnx_model = "../yolov8n-face.onnx"
CLASSES = ['face']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detect_faces(onnx_model):
    """
    Main function to load ONNX model, perform inference on webcam frames,
    draw bounding boxes, and display the output in real-time.

    Args:
        onnx_model (str): Path to the ONNX model.
    """
    # Load the ONNX model
    model = cv2.dnn.readNetFromONNX(onnx_model)

    # Initialize webcam capture
    cap = cv2.VideoCapture(8, cv2.CAP_V4L2)  # because here my USB device in under /dev/video8
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Get frame dimensions
        height, width, _ = frame.shape

        # Prepare a square image for inference
        length = max(height, width)
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = frame

        # Calculate scale factor
        scale = length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        model.setInput(blob)

        # Perform inference
        outputs = model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            _, maxScore, _, maxClassIndex = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex[1])

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

        # Check if result_boxes is a list of lists or a flat list
        if len(result_boxes) > 0 and isinstance(result_boxes[0], list):
            result_boxes = [i[0] for i in result_boxes]

        # Iterate through NMS results to draw bounding boxes and labels
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

        # Display the frame with bounding boxes
        cv2.imshow("Face Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    
detect_faces(onnx_model)