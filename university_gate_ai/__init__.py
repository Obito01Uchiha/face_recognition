"""
University Gate AI - Face Recognition System

A real-time face recognition attendance system for university gates.
Built with Python, OpenCV, MTCNN, FaceNet, and FAISS.
"""

__version__ = "1.0.0"
__author__ = "University Gate AI Team"
__description__ = "Real-time face recognition attendance system"

# Import main classes for easy access
from .camera_feed.capture_feed import WebcamCapture
from .face_recognition.detector import FaceDetector
from .face_recognition.recognizer import FAISSFaceRecognizer
from .face_recognition.face_utils import FaceEmbeddingExtractor, FaceDatabase

__all__ = [
    "WebcamCapture",
    "FaceDetector", 
    "FAISSFaceRecognizer",
    "FaceEmbeddingExtractor",
    "FaceDatabase"
]