"""Video capture and processing utilities"""
import cv2
import numpy as np
from typing import Optional, Tuple


class VideoCapture:
    """Handles video capture and frame processing"""

    def __init__(self, camera_id: int = 0):
        """
        Initialize video capture

        Args:
            camera_id: ID of the camera to use
        """
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the video capture

        Returns:
            Frame in BGR format if successful, None otherwise
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self) -> None:
        """Release the video capture"""
        if self.cap is not None:
            self.cap.release()

    def __del__(self) -> None:
        """Cleanup when the object is destroyed"""
        self.release()


def draw_face_box(
    frame: np.ndarray,
    face_position: Tuple[int, int, int, int],
    is_match: bool
) -> np.ndarray:
    """
    Draw a box around the detected face and display match status

    Args:
        frame: Input frame in BGR format
        face_position: Tuple of (x, y, width, height)
        is_match: Whether the face matches the reference

    Returns:
        Frame with annotations
    """
    x, y, w, h = face_position
    color = (0, 255, 0) if is_match else (0, 0, 255)
    status = "MATCH" if is_match else "NO MATCH"

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
    cv2.putText(frame, status, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame 