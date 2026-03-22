import cv2

def gstreamer_pipeline():
    return (
        "libcamerasrc ! "
        "video/x-raw, width=640, height=480, framerate=30/1, format=NV12 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false"
    )

def show_camera():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open CSI camera")
        return

    print("CSI Camera preview started. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to grab frame")
            break

        cv2.imshow("CSI Camera Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera()
