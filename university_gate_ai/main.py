"""
Main application entry point for University Gate AI Face Recognition System.
Integrates camera feed, face detection, and recognition for real-time attendance.
"""

import cv2
import numpy as np
import time
import threading
import argparse
import os
import sys
from datetime import datetime

# Import our modules
from camera_feed.capture_feed import WebcamCapture
from face_recognition.detector import FaceDetector
from face_recognition.recognizer import FAISSFaceRecognizer
from face_recognition.face_utils import FaceEmbeddingExtractor


class UniversityGateAI:
    """
    Main application class for the University Gate AI system.
    Handles real-time face recognition for student attendance.
    """
    
    def __init__(self, camera_id=0, confidence_threshold=0.9, similarity_threshold=0.6,
                 show_preview=True, save_logs=True):
        """
        Initialize the University Gate AI system.
        
        Args:
            camera_id (int): Camera device ID
            confidence_threshold (float): Face detection confidence threshold
            similarity_threshold (float): Face recognition similarity threshold
            show_preview (bool): Whether to show live preview window
            save_logs (bool): Whether to save attendance logs
        """
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.show_preview = show_preview
        self.save_logs = save_logs
        
        # System components
        self.camera = None
        self.detector = None
        self.recognizer = None
        
        # State management
        self.running = False
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Attendance tracking
        self.attendance_log = []
        self.recent_recognitions = {}  # Prevent duplicate entries
        self.recognition_cooldown = 5.0  # seconds
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'avg_processing_time': 0.0,
            'processing_times': []
        }
        
        print("🎓 University Gate AI - Face Recognition System")
        print("=" * 50)
        
    def initialize_system(self):
        """Initialize all system components."""
        print("🔧 Initializing system components...")
        
        try:
            # Initialize camera
            print("📹 Starting camera...")
            self.camera = WebcamCapture(camera_id=self.camera_id, buffer_size=2)
            if not self.camera.start():
                raise Exception("Failed to start camera")
            
            # Initialize face detector
            print("👤 Loading face detector...")
            self.detector = FaceDetector(
                min_face_size=40,
                confidence_threshold=self.confidence_threshold,
                device='cpu'  # Change to 'cuda' if GPU available
            )
            
            if self.detector.detector is None:
                raise Exception("Failed to initialize face detector")
            
            # Initialize face recognizer
            print("🧠 Loading face recognition system...")
            embedding_extractor = FaceEmbeddingExtractor(
                model_type='facenet',
                device='cpu'  # Change to 'cuda' if GPU available
            )
            
            self.recognizer = FAISSFaceRecognizer(
                embedding_extractor=embedding_extractor,
                similarity_threshold=self.similarity_threshold
            )
            
            print("✅ System initialization complete!")
            
            # Display system info
            self._display_system_info()
            
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
            
    def _display_system_info(self):
        """Display system information and database stats."""
        print("\n📊 System Information:")
        
        # Camera info
        if self.camera:
            camera_info = self.camera.get_camera_info()
            if camera_info:
                print(f"   📹 Camera: {camera_info['width']}x{camera_info['height']} @ {camera_info['fps']}fps")
                print(f"   🔧 Backend: {camera_info['backend']}")
        
        # Database info
        if self.recognizer:
            db_stats = self.recognizer.get_database_stats()
            print(f"   👥 Registered persons: {db_stats['total_persons']}")
            print(f"   🎯 Similarity threshold: {db_stats['similarity_threshold']}")
            
            if db_stats['total_persons'] > 0:
                print(f"   📝 Persons: {', '.join(db_stats['persons'][:5])}")
                if len(db_stats['persons']) > 5:
                    print(f"        ... and {len(db_stats['persons']) - 5} more")
        
        print("-" * 50)
        
    def process_frame(self, frame):
        """
        Process a single frame for face detection and recognition.
        
        Args:
            frame (numpy.ndarray): Input frame from camera
            
        Returns:
            numpy.ndarray: Processed frame with annotations
        """
        start_time = time.time()
        
        # Detect faces
        faces = self.detector.detect_faces(frame)
        self.performance_stats['total_faces_detected'] += len(faces)
        
        # Process each detected face
        for face in faces:
            face_crop = face['face_crop']
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # Recognize the face
            name, similarity = self.recognizer.recognize_best_match(face_image=face_crop)
            
            # Draw bounding box and label
            if name and similarity >= self.similarity_threshold:
                # Recognized face
                color = (0, 255, 0)  # Green
                label = f"{name} ({similarity:.2f})"
                
                # Log attendance (with cooldown to prevent duplicates)
                self._log_attendance(name, similarity)
                self.performance_stats['total_faces_recognized'] += 1
                
            else:
                # Unknown face
                color = (0, 0, 255)  # Red
                label = f"Unknown ({confidence:.2f})"
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Add background for text readability
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Track performance
        processing_time = time.time() - start_time
        self.performance_stats['processing_times'].append(processing_time)
        if len(self.performance_stats['processing_times']) > 100:
            self.performance_stats['processing_times'].pop(0)
        
        self.performance_stats['avg_processing_time'] = np.mean(
            self.performance_stats['processing_times']
        )
        
        return frame
        
    def _log_attendance(self, name, similarity):
        """
        Log attendance with duplicate prevention.
        
        Args:
            name (str): Person's name
            similarity (float): Recognition similarity score
        """
        current_time = time.time()
        
        # Check if this person was recently recognized
        if name in self.recent_recognitions:
            if current_time - self.recent_recognitions[name] < self.recognition_cooldown:
                return  # Skip duplicate recognition
        
        # Log the attendance
        timestamp = datetime.now()
        attendance_entry = {
            'name': name,
            'timestamp': timestamp,
            'similarity': similarity,
            'status': 'present'
        }
        
        self.attendance_log.append(attendance_entry)
        self.recent_recognitions[name] = current_time
        
        # Print attendance notification
        print(f"✅ {timestamp.strftime('%H:%M:%S')} - {name} recognized (similarity: {similarity:.3f})")
        
        # Save to file if enabled
        if self.save_logs:
            self._save_attendance_log()
            
    def _save_attendance_log(self):
        """Save attendance log to file."""
        try:
            os.makedirs('logs', exist_ok=True)
            
            log_file = f"logs/attendance_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # Check if file exists to write header
            write_header = not os.path.exists(log_file)
            
            with open(log_file, 'a') as f:
                if write_header:
                    f.write("timestamp,name,similarity,status\n")
                    
                # Write the latest entry
                if self.attendance_log:
                    entry = self.attendance_log[-1]
                    f.write(f"{entry['timestamp'].isoformat()},"
                           f"{entry['name']},{entry['similarity']:.3f},"
                           f"{entry['status']}\n")
                    
        except Exception as e:
            print(f"⚠️  Failed to save attendance log: {e}")
            
    def draw_ui_overlay(self, frame):
        """
        Draw UI overlay with system information.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            
        Returns:
            numpy.ndarray: Frame with UI overlay
        """
        # Calculate FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # System status
        status_text = [
            f"FPS: {self.current_fps}",
            f"Faces Detected: {self.performance_stats['total_faces_detected']}",
            f"Faces Recognized: {self.performance_stats['total_faces_recognized']}",
            f"Processing: {self.performance_stats['avg_processing_time']*1000:.1f}ms"
        ]
        
        # Draw status text
        for i, text in enumerate(status_text):
            cv2.putText(frame, text, (15, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw recent attendance
        if self.attendance_log:
            recent_entries = self.attendance_log[-3:]  # Show last 3 entries
            cv2.putText(frame, "Recent Attendance:", (15, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            for i, entry in enumerate(recent_entries):
                time_str = entry['timestamp'].strftime('%H:%M:%S')
                text = f"{time_str} - {entry['name']}"
                cv2.putText(frame, text, (15, 160 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame
        
    def run(self):
        """Run the main application loop."""
        if not self.initialize_system():
            return False
            
        print("\n🚀 Starting University Gate AI system...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to show system stats")
        print("  - Press 'r' to register new face")
        print("  - Press 'c' to clear attendance log")
        print("-" * 50)
        
        self.running = True
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                self.performance_stats['total_frames'] += 1
                
                # Show preview if enabled
                if self.show_preview:
                    # Draw UI overlay
                    display_frame = self.draw_ui_overlay(processed_frame)
                    
                    # Resize for display
                    display_frame = cv2.resize(display_frame, (1024, 768))
                    cv2.imshow('University Gate AI - Face Recognition', display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._show_system_stats()
                    elif key == ord('r'):
                        self._register_new_face(frame)
                    elif key == ord('c'):
                        self._clear_attendance_log()
                
        except KeyboardInterrupt:
            print("\n⚠️  System interrupted by user")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            self._cleanup()
            
        return True
        
    def _show_system_stats(self):
        """Display detailed system statistics."""
        print("\n📊 System Statistics:")
        print(f"   Total frames processed: {self.performance_stats['total_frames']}")
        print(f"   Total faces detected: {self.performance_stats['total_faces_detected']}")
        print(f"   Total faces recognized: {self.performance_stats['total_faces_recognized']}")
        print(f"   Average processing time: {self.performance_stats['avg_processing_time']*1000:.2f}ms")
        print(f"   Current FPS: {self.current_fps}")
        print(f"   Attendance entries: {len(self.attendance_log)}")
        
        if self.recognizer:
            db_stats = self.recognizer.get_database_stats()
            print(f"   Database persons: {db_stats['total_persons']}")
            if 'search_fps' in db_stats:
                print(f"   Search FPS: {db_stats['search_fps']:.1f}")
        print("-" * 50)
        
    def _register_new_face(self, frame):
        """Register a new face from the current frame."""
        faces = self.detector.detect_faces(frame)
        if not faces:
            print("⚠️  No faces detected for registration")
            return
            
        # Use the largest face
        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        face_crop = face['face_crop']
        
        # Get name from user
        print("👤 Enter name for the detected face:")
        name = input("Name: ").strip()
        
        if name:
            success = self.recognizer.add_person(name, face_image=face_crop)
            if success:
                self.recognizer.save_database()
                print(f"✅ Successfully registered {name}")
            else:
                print(f"❌ Failed to register {name}")
        else:
            print("⚠️  Registration cancelled - no name provided")
            
    def _clear_attendance_log(self):
        """Clear the attendance log."""
        self.attendance_log.clear()
        self.recent_recognitions.clear()
        print("🗑️  Attendance log cleared")
        
    def _cleanup(self):
        """Cleanup system resources."""
        print("\n🧹 Cleaning up system resources...")
        
        self.running = False
        
        if self.camera:
            self.camera.stop()
            
        if self.show_preview:
            cv2.destroyAllWindows()
            
        # Save final attendance log
        if self.save_logs and self.attendance_log:
            self._save_attendance_log()
            
        print("✅ Cleanup complete")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='University Gate AI - Face Recognition System')
    
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera device ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.9,
                       help='Face detection confidence threshold (default: 0.9)')
    parser.add_argument('--similarity', type=float, default=0.6,
                       help='Face recognition similarity threshold (default: 0.6)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable live preview window')
    parser.add_argument('--no-logs', action='store_true',
                       help='Disable attendance logging')
    
    args = parser.parse_args()
    
    # Create and run the system
    system = UniversityGateAI(
        camera_id=args.camera,
        confidence_threshold=args.confidence,
        similarity_threshold=args.similarity,
        show_preview=not args.no_preview,
        save_logs=not args.no_logs
    )
    
    success = system.run()
    
    if success:
        print("🎓 University Gate AI system completed successfully")
    else:
        print("❌ University Gate AI system failed to start")
        sys.exit(1)


if __name__ == "__main__":
    main()