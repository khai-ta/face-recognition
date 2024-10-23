import threading
import cv2
from deepface import DeepFace

# Initialize video capture for the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame counter and face match status
counter = 0
face_match = False

# Load reference image for face comparison
reference_img = cv2.imread('reference.jpg')

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variable for tracking face position
last_face_position = None


def check_face(frame):
    """Verify if the captured frame matches the reference image."""
    global face_match
    try:
        face_match = DeepFace.verify(frame, reference_img.copy())['verified']
    except ValueError:
        face_match = False


def process_frames():
    """Continuously capture frames and process them for face matching."""
    global counter, last_face_position  # Declare as global to modify
    while True:
        ret, frame = cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            # Check face match every 30 frames for better performance
            if counter % 30 == 0:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()

            counter += 1

            # If a face is detected, track its position
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                last_face_position = (x, y, w, h)
            elif last_face_position:
                # If no face detected, use the last known position
                (x, y, w, h) = last_face_position
            else:
                # If no face has been detected previously, set a default position
                (x, y, w, h) = (0, 0, 1, 1)

            # Draw rectangle around the tracked face position
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)

            # Display match status near the tracked face
            if face_match:
                cv2.putText(frame, "MATCH", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO MATCH", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display the processed frame
            cv2.imshow('video', frame)

        # Exit loop if 'q' key is pressed
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


# Start processing video frames
process_frames()

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
