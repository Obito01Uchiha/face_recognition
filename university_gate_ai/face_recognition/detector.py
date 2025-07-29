"""
Face detection module using MTCNN for robust face detection.
Optimized for real-time performance with batch processing capabilities.
"""

import cv2
import numpy as np
from mtcnn import MTCNN
import torch
from PIL import Image
import time


class FaceDetector:
    """
    Face detection class using MTCNN for accurate face detection.
    Handles multiple faces and provides bounding box coordinates.
    """
    
    def __init__(self, min_face_size=40, confidence_threshold=0.9, device='cpu'):
        """
        Initialize the face detector.
        
        Args:
            min_face_size (int): Minimum face size to detect
            confidence_threshold (float): Minimum confidence for face detection
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Initialize MTCNN
        print("Loading MTCNN face detector...")
        try:
            self.detector = MTCNN(
                min_face_size=min_face_size,
                thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
                factor=0.709,  # Scale factor for image pyramid
                post_process=True,
                device=device
            )
            print("MTCNN loaded successfully")
        except Exception as e:
            print(f"Error loading MTCNN: {e}")
            self.detector = None
            
        # Performance tracking
        self.detection_times = []
        self.max_time_samples = 100
        
    def detect_faces(self, frame):
        """
        Detect faces in a frame.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            list: List of dictionaries containing face information:
                - 'box': [x, y, width, height] bounding box
                - 'confidence': Detection confidence score
                - 'keypoints': Facial keypoints (eyes, nose, mouth)
                - 'face_crop': Cropped face image
        """
        if self.detector is None:
            return []
            
        start_time = time.time()
        
        try:
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector.detect_faces(rgb_frame)
            
            faces = []
            for detection in detections:
                confidence = detection['confidence']
                
                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                    
                # Extract bounding box
                box = detection['box']
                x, y, w, h = box
                
                # Ensure bounding box is within frame boundaries
                frame_height, frame_width = frame.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)
                
                # Skip very small faces
                if w < self.min_face_size or h < self.min_face_size:
                    continue
                    
                # Extract face crop with some padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame_width, x + w + padding)
                y2 = min(frame_height, y + h + padding)
                
                face_crop = frame[y1:y2, x1:x2]
                
                # Extract keypoints
                keypoints = detection.get('keypoints', {})
                
                face_info = {
                    'box': [x, y, w, h],
                    'confidence': confidence,
                    'keypoints': keypoints,
                    'face_crop': face_crop,
                    'padded_box': [x1, y1, x2-x1, y2-y1]
                }
                
                faces.append(face_info)
                
        except Exception as e:
            print(f"Error in face detection: {e}")
            faces = []
            
        # Track detection time
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        if len(self.detection_times) > self.max_time_samples:
            self.detection_times.pop(0)
            
        return faces
        
    def detect_largest_face(self, frame):
        """
        Detect only the largest face in the frame.
        Useful for single-person scenarios or primary face focus.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            dict: Face information for the largest detected face, or None
        """
        faces = self.detect_faces(frame)
        
        if not faces:
            return None
            
        # Find the largest face by area
        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        return largest_face
        
    def draw_detections(self, frame, faces, draw_keypoints=True, draw_confidence=True):
        """
        Draw face detection results on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame to draw on
            faces (list): List of face detection results
            draw_keypoints (bool): Whether to draw facial keypoints
            draw_confidence (bool): Whether to draw confidence scores
            
        Returns:
            numpy.ndarray: Frame with detection results drawn
        """
        output_frame = frame.copy()
        
        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            keypoints = face.get('keypoints', {})
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            if draw_confidence:
                conf_text = f"{confidence:.2f}"
                cv2.putText(output_frame, conf_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                           
            # Draw keypoints
            if draw_keypoints and keypoints:
                # Draw eyes
                if 'left_eye' in keypoints:
                    cv2.circle(output_frame, tuple(map(int, keypoints['left_eye'])), 3, (255, 0, 0), -1)
                if 'right_eye' in keypoints:
                    cv2.circle(output_frame, tuple(map(int, keypoints['right_eye'])), 3, (255, 0, 0), -1)
                    
                # Draw nose
                if 'nose' in keypoints:
                    cv2.circle(output_frame, tuple(map(int, keypoints['nose'])), 3, (0, 0, 255), -1)
                    
                # Draw mouth corners
                if 'mouth_left' in keypoints:
                    cv2.circle(output_frame, tuple(map(int, keypoints['mouth_left'])), 3, (0, 255, 255), -1)
                if 'mouth_right' in keypoints:
                    cv2.circle(output_frame, tuple(map(int, keypoints['mouth_right'])), 3, (0, 255, 255), -1)
                    
        return output_frame
        
    def get_performance_stats(self):
        """
        Get performance statistics for the detector.
        
        Returns:
            dict: Performance statistics including average detection time and FPS
        """
        if not self.detection_times:
            return {'avg_time': 0, 'fps': 0, 'samples': 0}
            
        avg_time = np.mean(self.detection_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_time': avg_time,
            'fps': fps,
            'samples': len(self.detection_times),
            'min_time': np.min(self.detection_times),
            'max_time': np.max(self.detection_times)
        }
        
    def preprocess_for_recognition(self, face_crop, target_size=(160, 160)):
        """
        Preprocess face crop for recognition model.
        
        Args:
            face_crop (numpy.ndarray): Cropped face image
            target_size (tuple): Target size for the face image
            
        Returns:
            numpy.ndarray: Preprocessed face image
        """
        if face_crop is None or face_crop.size == 0:
            return None
            
        try:
            # Resize to target size
            resized = cv2.resize(face_crop, target_size)
            
            # Convert to RGB if needed
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
            # Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            print(f"Error in face preprocessing: {e}")
            return None


def test_detector():
    """Test function to verify face detection works."""
    print("Testing face detector...")
    
    # Initialize detector
    detector = FaceDetector(confidence_threshold=0.8)
    
    if detector.detector is None:
        print("Failed to initialize detector")
        return
        
    # Test with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    print("Press 'q' to quit test")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw results
            output_frame = detector.draw_detections(frame, faces)
            
            # Show performance stats
            stats = detector.get_performance_stats()
            if stats['samples'] > 0:
                fps_text = f"FPS: {stats['fps']:.1f}"
                cv2.putText(output_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                           
            cv2.imshow('Face Detection Test', output_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Face detection test completed")
        
        # Print final stats
        stats = detector.get_performance_stats()
        print(f"Performance stats: {stats}")


if __name__ == "__main__":
    test_detector()