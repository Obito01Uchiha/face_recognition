"""Face recognition module with detection, recognition, and utilities."""

from .detector import FaceDetector
from .recognizer import FAISSFaceRecognizer
from .face_utils import FaceEmbeddingExtractor, FaceDatabase

__all__ = [
    "FaceDetector",
    "FAISSFaceRecognizer", 
    "FaceEmbeddingExtractor",
    "FaceDatabase"
]