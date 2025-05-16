"""Face recognition module using DeepFace"""
from typing import Optional
import cv2
import numpy as np
from deepface import DeepFace


class FaceRecognizer:
    """Handles face recognition using DeepFace"""

    def __init__(self, reference_image_path: str, confidence_threshold: float = 0.6):
        """
        Initialize the face recognizer with a reference image

        Args:
            reference_image_path: Path to the reference image
            confidence_threshold: Minimum confidence score for a match (0.0 to 1.0)
        """
        self.reference_img = cv2.imread(reference_image_path)
        if self.reference_img is None:
            raise FileNotFoundError(f"Reference image not found at {reference_image_path}")
        self.confidence_threshold = confidence_threshold

    def verify_face(self, frame: np.ndarray) -> bool:
        """
        Verify if the face in the frame matches the reference image

        Args:
            frame: Input frame in BGR format

        Returns:
            True if face matches, False otherwise
        """
        try:
            result = DeepFace.verify(
                frame,
                self.reference_img.copy(),
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False 
            )
            return result['verified'] and result['distance'] < (1 - self.confidence_threshold)
        except ValueError as e:
            print(f"Face verification failed: {str(e)}")
            return False
        except Exception as e:
            print(f"Error during face verification: {str(e)}")
            return False 