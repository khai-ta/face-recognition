# Face Recognition System

This Python application captures video frames from a camera and checks if the face in each frame matches a reference image using the DeepFace library. It continuously processes the video stream in real-time, displaying either "MATCH" or "NO MATCH" based on the comparison result.

## Prerequisites

- Python 3.8+
- OpenCV (cv2)
- DeepFace
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition.git
cd face-recognition
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your reference image in the project directory (default: `reference.jpg`)

2. Run the face recognition system:
```bash
python -m src.app.main
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
