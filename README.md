# Face Recognition System

This Python application captures video frames from a camera and checks if the face in each frame matches a reference image using the DeepFace library. It continuously processes the video stream in real-time, displaying either "MATCH" or "NO MATCH" based on the comparison result.

## Features

- Real-Time Face Matching: Uses the DeepFace library to compare the face in the video frame with a reference image
- Multi-threading: Face matching is processed in a separate thread every 30 frames to maintain performance
- Interactive Display: The video stream displays "MATCH" if the face matches the reference image, otherwise "NO MATCH". Press q to exit the script

## Prerequisites

- Python 3.6+
- OpenCV (cv2)
- DeepFace
- A reference image named reference.jpg in the same directory as the script

## Installation

1. Clone the repository or download the source code

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Ensure you have a camera connected to your system

## Usage

1. Place your reference image in the same directory as the script and name it reference.jpg

2. Run the script:
```bash
python -m src.core.face_recognition
```

The video feed from your camera will appear in a window. The script will display "MATCH" or "NO MATCH" depending on whether it finds a matching face in the frame.

Press q to quit the video feed and close the window.

## How It Works

- The script initializes the video capture object for the default camera and sets the video frame dimensions
- It reads and stores the reference image
- A continuous loop captures frames from the camera feed and processes them
- Every 30 frames, the script checks if the face in the current frame matches the reference image. This is done in a separate thread to avoid blocking the video feed
- The result ("MATCH" or "NO MATCH") is displayed on the video feed

## Notes

- Ensure your reference image is clear and well-lit for optimal face matching results
- The video feed window may have some delay depending on your system's performance and camera quality
