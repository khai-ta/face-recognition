"""Face recognition module using DeepFace"""
from typing import Optional
import cv2
import numpy as np
from deepface import DeepFace


class FaceRecognizer:
    """Handles face recognition using DeepFace"""

    def __init__(self, reference_image_path: str):
        """
        Initialize the face recognizer with a reference image

        Args:
            reference_image_path: Path to the reference image
        """
        self.reference_img = cv2.imread(reference_image_path)
        if self.reference_img is None:
            raise FileNotFoundError(f"Reference image not found at {reference_image_path}")

    def verify_face(self, frame: np.ndarray) -> bool:
        """
        Verify if the face in the frame matches the reference image

        Args:
            frame: Input frame in BGR format

        Returns:
            True if face matches, False otherwise
        """
        try:
            result = DeepFace.verify(frame, self.reference_img.copy())
            return result['verified']
        except ValueError:
            return False
        except Exception as e:
            print(f"Error during face verification: {str(e)}")
            return False 