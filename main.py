import threading
import cv2
from deepface import DeepFace

# initialize video capture object for the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# set dimensions of the video frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# set counter to 0
counter = 0

# initialize a boolean to indicate if a face match is found
face_match = False

# load reference image for comparison
reference_img = cv2.imread('reference.jpg')


# function to check if the face in the frame matches the reference image
def check_face(frame):
    global face_match
    try:
        # verify if the captured frame matches the reference image
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


# function to process frames
def process_frames():
    global counter
    while True:
        # capture frame-by-frame from the camera
        ret, frame = cap.read()

        if ret:
            # check face match every 60 frames
            if counter % 60 == 0:
                try:
                    # Start a new thread to process face matching
                    threading.Thread(target=check_face, args=(frame.copy(),)).start()
                except ValueError:
                    pass

            counter += 1

            # display "MATCH" if face match is found, else "NO MATCH"
            if face_match:
                cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # display frame with "MATCH" or "NO MATCH" text
            cv2.imshow('video', frame)

        # check if the 'q' key is pressed; if so, exit loop
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


# start processing frames
process_frames()

# close all openCV windows
cv2.destroyAllWindows()
