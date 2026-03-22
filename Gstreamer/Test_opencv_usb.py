import cv2

def gstreamer_pipeline():
    return (
        "v4l2src device=/dev/video8 ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! "
        "appsink drop=true sync=false"
    )

def show_camera():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Press 'q' to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        cv2.imshow("Webcam Preview (GStreamer + OpenCV)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera()
