"""
Multithreaded webcam stream capture for real-time face recognition.
Optimized for performance with separate thread for frame reading.
"""

import cv2
import threading
import time
from queue import Queue
import numpy as np


class WebcamCapture:
    """
    Multithreaded webcam capture class for optimal performance.
    Separates frame reading from processing to prevent blocking.
    """
    
    def __init__(self, camera_id=0, buffer_size=2):
        """
        Initialize webcam capture.
        
        Args:
            camera_id (int): Camera device ID (0 for default camera)
            buffer_size (int): Maximum frames to buffer
        """
        self.camera_id = camera_id
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size)
        self.capture = None
        self.capture_thread = None
        self.running = False
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
        
    def start(self):
        """Start the camera capture and processing thread."""
        if self.running:
            print("Camera capture is already running!")
            return False
            
        # Initialize camera
        self.capture = cv2.VideoCapture(self.camera_id)
        if not self.capture.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
            
        # Set camera properties for optimal performance
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
        
        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        print(f"Camera {self.camera_id} started successfully")
        return True
        
    def _capture_frames(self):
        """Internal method to continuously capture frames in separate thread."""
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                print("Warning: Failed to read frame from camera")
                continue
                
            # Add frame to queue (remove oldest if full)
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # Remove oldest frame and add new one
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except:
                    pass
                    
            # Small delay to prevent excessive CPU usage
            time.sleep(1/self.fps)
            
    def get_frame(self):
        """
        Get the latest frame from the camera.
        
        Returns:
            numpy.ndarray: Latest camera frame, or None if no frame available
        """
        if not self.running:
            return None
            
        try:
            # Get the most recent frame
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            return frame
        except:
            return None
            
    def stop(self):
        """Stop the camera capture and cleanup resources."""
        if not self.running:
            return
            
        print("Stopping camera capture...")
        self.running = False
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        # Release camera resources
        if self.capture:
            self.capture.release()
            
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
                
        print("Camera capture stopped")
        
    def is_running(self):
        """Check if camera capture is currently running."""
        return self.running
        
    def get_camera_info(self):
        """Get camera properties and information."""
        if not self.capture or not self.running:
            return None
            
        info = {
            'width': int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.capture.get(cv2.CAP_PROP_FPS)),
            'backend': self.capture.getBackendName()
        }
        return info
        
    def set_resolution(self, width, height):
        """
        Set camera resolution.
        
        Args:
            width (int): Frame width
            height (int): Frame height
        """
        if self.capture and self.running:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.frame_width = width
            self.frame_height = height
            
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


def test_camera():
    """Test function to verify camera capture works."""
    print("Testing camera capture...")
    
    camera = WebcamCapture(camera_id=0)
    
    if not camera.start():
        print("Failed to start camera")
        return
        
    print("Camera info:", camera.get_camera_info())
    print("Press 'q' to quit test")
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                # Resize for display
                display_frame = cv2.resize(frame, (800, 600))
                cv2.imshow('Camera Test', display_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)  # Small delay if no frame
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera test completed")


if __name__ == "__main__":
    test_camera()