"""Face detection module using OpenCV's Haar Cascade Classifier"""
from typing import Optional, Tuple
import cv2
import numpy as np


class FaceDetector:
    """Handles face detection using OpenCV's Haar Cascade Classifier"""

    def __init__(self):
        """Initialize the face detector with Haar Cascade Classifier"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect faces in the given frame

        Args:
            frame: Input frame in BGR format

        Returns:
            Tuple of (x, y, width, height) if face is detected, None otherwise
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(30, 30),
            maxSize=(300, 300)
        )
        return faces[0] if len(faces) > 0 else None 