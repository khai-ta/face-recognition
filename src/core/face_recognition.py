"""Core face recognition system implementation"""
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

from ..utils.video import VideoCapture, draw_face_box
from .detector import FaceDetector
from .recognizer import FaceRecognizer


class FaceRecognitionSystem:
    """Main face recognition system class"""

    def __init__(self, camera_id: int = 0, reference_image_path: str = 'reference.jpg'):
        """
        Initialize the face recognition system

        Args:
            camera_id: ID of the camera to use
            reference_image_path: Path to the reference image
        """
        self.video = VideoCapture(camera_id)
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer(reference_image_path)
        self.counter = 0
        self.face_match = False
        self.last_face_position: Optional[Tuple[int, int, int, int]] = None

    def process_frame(self, frame: np.ndarray) -> None:
        """
        Process a single frame for face detection and recognition

        Args:
            frame: Input frame in BGR format
        """
        # Detect face
        face_position = self.detector.detect_face(frame)
        if face_position is not None:
            self.last_face_position = face_position
        elif self.last_face_position is None:
            self.last_face_position = (0, 0, 1, 1)

        # Check face match periodically
        if self.counter % 30 == 0:
            threading.Thread(target=self._check_face, args=(frame.copy(),)).start()

        self.counter += 1

        # Draw results
        frame = draw_face_box(frame, self.last_face_position, self.face_match)
        cv2.imshow('Face Recognition', frame)

    def _check_face(self, frame: np.ndarray) -> None:
        """
        Check if the face in the frame matches the reference image

        Args:
            frame: Input frame in BGR format
        """
        self.face_match = self.recognizer.verify_face(frame)

    def run(self) -> None:
        """Run the face recognition system"""
        try:
            while True:
                frame = self.video.read_frame()
                if frame is None:
                    raise RuntimeError("Failed to capture frame")

                self.process_frame(frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        except Exception as e:
            print(f"Error during execution: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources"""
        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run() 