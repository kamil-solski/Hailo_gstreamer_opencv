"""
Face Detection Inference Script for Hailo‑8L on Raspberry Pi 5 using HailoRT v4.20.0

This script performs face detection with a HEF model on a Hailo‑8L accelerator.
It uses the HailoRT v4.20.0 Python API to load the HEF file, create a network group, 
and run inference. The model is assumed to have the following dimensions:
    - Input: [1, 3, 416, 416]
    - Output: [1, 5, 3549] (each detection: [center_x, center_y, width, height, confidence])
    
Video is captured via a GStreamer pipeline from the Raspberry Pi CSI camera.
Make sure you have converted your ONNX model into a HEF file before running.
Also ensure that HailoRT v4.20.0, its drivers, and required Python modules (e.g., OpenCV, NumPy)
are properly installed.
"""

import cv2
import numpy as np
import hailort  # This is the HailoRT v4.20.0 Python module

# -----------------------------------------------------------------------------
# Model and Visualization Configuration
# -----------------------------------------------------------------------------

# Path to your compiled HEF model.
hef_face_model = "../yolov8n_face.hef"

# The list of class names – here we assume a single class "face".
CLASSES = ['face']
# Generate a random color for drawing the bounding box.
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draw a bounding box along with a label and confidence on the image.
    
    Args:
        img (np.ndarray): The image where the box is drawn.
        class_id (int): The class index (0 for face).
        confidence (float): The detection confidence.
        x, y (int): Top-left coordinates of the box.
        x_plus_w, y_plus_h (int): Bottom-right coordinates.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def gstreamer_pipeline():
    """
    Returns the GStreamer pipeline string for capturing video from the CSI camera.
    Adjust parameters (width, height, framerate) if needed.
    """
    return (
        "libcamerasrc ! "
        "video/x-raw,width=640,height=480,framerate=30/1,format=NV12 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false"
    )


def detect_faces(hef_model):
    """
    Capture frames from the CSI camera, run inference using Hailo‑8L via HailoRT v4.20.0,
    and display face detections.
    
    Args:
        hef_model (str): Path to the HEF model file.
    """
    # -------------------------------------------------------------------------
    # Step 1: Initialize HailoRT – load the HEF model and create a network group.
    # -------------------------------------------------------------------------
    # Load the HEF file.
    hef = hailort.HEF(hef_model)
    
    # Create a device instance (uses the first available Hailo device).
    device = hailort.Device.create()
    
    # Create a network group from the HEF file. Replace "my_network_group" with your
    # network group name as defined in the HEF.
    network_group = hef.create_network_group(device, "my_network_group", hailort.StreamParameters())
    
    # Retrieve the expected input shape. Expected format: [batch, channels, height, width]
    batch, channels, expected_height, expected_width = network_group.get_input_shape()
    print(f"Loaded HEF model: {hef_model}")
    print(f"Expected input dimensions: {expected_width}x{expected_height} with {channels} channels")
    
    # Start the network group; this is needed before running inference.
    network_group.start()
    
    # -----------------------------------------------------------------------------
    # Step 2: Set Up CSI Camera Capture via GStreamer.
    # -----------------------------------------------------------------------------
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Could not open CSI camera")
        return
    print("Face detection started using Hailo‑8L. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to grab frame")
            break

        # ---------------------------------------------------------------------
        # Step 3: Preprocess the Frame
        # ---------------------------------------------------------------------
        # We need to convert the captured frame into a square image (via padding)
        # so that we can resize it to the model's expected dimensions.
        height, width, _ = frame.shape
        side_length = max(height, width)
        padded_frame = np.zeros((side_length, side_length, 3), dtype=np.uint8)
        padded_frame[0:height, 0:width] = frame

        # Resize the padded frame to the required model input size (416x416).
        resized_frame = cv2.resize(padded_frame, (expected_width, expected_height))
        # Normalize the image to the [0, 1] range.
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        # Convert from HWC to CHW layout (channels first).
        chw_frame = np.transpose(normalized_frame, (2, 0, 1))
        # Add a batch dimension resulting in shape [1, 3, 416, 416].
        input_blob = np.expand_dims(chw_frame, axis=0)

        # ---------------------------------------------------------------------
        # Step 4: Run Inference via Hailo‑8L
        # ---------------------------------------------------------------------
        # The infer() method submits the input blob for inference.
        # The output is assumed to be of shape [1, 5, 3549]:
        # For each of the 3549 proposals: [center_x, center_y, width, height, confidence]
        outputs = network_group.infer(input_blob)
        
        # ---------------------------------------------------------------------
        # Step 5: Postprocess the Output
        # ---------------------------------------------------------------------
        # Remove the batch dimension. Detections now has shape [5, 3549].
        detections = outputs[0]
        num_detections = detections.shape[1]
        detection_threshold = 0.25

        boxes = []       # Stores bounding box parameters
        scores = []      # Stores detection confidence scores
        class_ids = []   # Only one class (face) -> always 0

        for i in range(num_detections):
            confidence = detections[4, i]
            if confidence >= detection_threshold:
                center_x = detections[0, i]
                center_y = detections[1, i]
                box_width = detections[2, i]
                box_height = detections[3, i]
                # Convert from center coordinates to top-left coordinates.
                x = center_x - (box_width / 2)
                y = center_y - (box_height / 2)
                boxes.append([x, y, box_width, box_height])
                scores.append(confidence)
                class_ids.append(0)  # For "face"

        # Optionally, perform Non-Maximum Suppression (NMS) to eliminate overlapping boxes.
        result_indices = cv2.dnn.NMSBoxes(boxes, scores, detection_threshold, 0.45)
        if len(result_indices) > 0 and isinstance(result_indices[0], (list, tuple, np.ndarray)):
            result_indices = [i[0] for i in result_indices]

        # Calculate scale factor to map coordinates from the resized (416x416) image
        # back to the original frame dimensions.
        scale_factor = side_length / expected_width
        
        # ---------------------------------------------------------------------
        # Step 6: Draw Detections on the Original Frame
        # ---------------------------------------------------------------------
        for idx in result_indices:
            box = boxes[idx]
            x = round(box[0] * scale_factor)
            y = round(box[1] * scale_factor)
            x_plus_w = round((box[0] + box[2]) * scale_factor)
            y_plus_h = round((box[1] + box[3]) * scale_factor)
            draw_bounding_box(frame, class_ids[idx], scores[idx], x, y, x_plus_w, y_plus_h)

        cv2.imshow("Hailo‑8L Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -------------------------------------------------------------------------
    # Step 7: Clean Up Resources
    # -------------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    # Stop the network group and close the device.
    network_group.stop()
    device.close()


if __name__ == "__main__":
    detect_faces(hef_face_model)
