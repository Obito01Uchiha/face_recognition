"""
Face registration script for adding new students to the University Gate AI system.
Supports batch registration from images or live camera feed.
"""

import cv2
import numpy as np
import os
import argparse
import time
from pathlib import Path
import json
from datetime import datetime

# Import our modules
from face_recognition.detector import FaceDetector
from face_recognition.recognizer import FAISSFaceRecognizer
from face_recognition.face_utils import FaceEmbeddingExtractor


class FaceRegistrationSystem:
    """
    System for registering new faces into the University Gate AI database.
    Supports multiple registration methods and quality validation.
    """
    
    def __init__(self, confidence_threshold=0.9, min_face_size=80):
        """
        Initialize the face registration system.
        
        Args:
            confidence_threshold (float): Minimum confidence for face detection
            min_face_size (int): Minimum face size for registration
        """
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        
        # Initialize components
        self.detector = None
        self.recognizer = None
        self.embedding_extractor = None
        
        # Registration statistics
        self.registration_stats = {
            'total_attempts': 0,
            'successful_registrations': 0,
            'failed_registrations': 0,
            'duplicate_attempts': 0
        }
        
        print("👥 University Gate AI - Face Registration System")
        print("=" * 50)
        
    def initialize_system(self):
        """Initialize all system components."""
        print("🔧 Initializing registration system...")
        
        try:
            # Initialize face detector
            print("👤 Loading face detector...")
            self.detector = FaceDetector(
                min_face_size=self.min_face_size,
                confidence_threshold=self.confidence_threshold,
                device='cpu'
            )
            
            if self.detector.detector is None:
                raise Exception("Failed to initialize face detector")
            
            # Initialize embedding extractor
            print("🧠 Loading face embedding extractor...")
            self.embedding_extractor = FaceEmbeddingExtractor(
                model_type='facenet',
                device='cpu'
            )
            
            if self.embedding_extractor.model is None:
                raise Exception("Failed to initialize embedding extractor")
            
            # Initialize face recognizer (database)
            print("💾 Loading face database...")
            self.recognizer = FAISSFaceRecognizer(
                embedding_extractor=self.embedding_extractor,
                similarity_threshold=0.6
            )
            
            print("✅ Registration system initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
            
    def validate_face_quality(self, face_info):
        """
        Validate face quality for registration.
        
        Args:
            face_info (dict): Face information from detector
            
        Returns:
            tuple: (is_valid, quality_score, issues)
        """
        issues = []
        quality_score = 0.0
        
        # Check face size
        x, y, w, h = face_info['box']
        face_area = w * h
        
        if w < self.min_face_size or h < self.min_face_size:
            issues.append(f"Face too small ({w}x{h}, minimum: {self.min_face_size}x{self.min_face_size})")
        else:
            quality_score += 0.3
            
        # Check detection confidence
        confidence = face_info['confidence']
        if confidence < self.confidence_threshold:
            issues.append(f"Low detection confidence ({confidence:.2f}, minimum: {self.confidence_threshold})")
        else:
            quality_score += 0.3
            
        # Check face crop quality
        face_crop = face_info['face_crop']
        if face_crop is None or face_crop.size == 0:
            issues.append("Invalid face crop")
        else:
            # Check if face crop is not too blurry
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 100:  # Threshold for blur detection
                issues.append(f"Face appears blurry (variance: {laplacian_var:.1f})")
            else:
                quality_score += 0.2
                
        # Check aspect ratio (faces should be roughly square)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            issues.append(f"Unusual face aspect ratio ({aspect_ratio:.2f})")
        else:
            quality_score += 0.2
            
        is_valid = len(issues) == 0
        return is_valid, quality_score, issues
        
    def register_face_from_image(self, image_path, name, validate_quality=True):
        """
        Register a face from an image file.
        
        Args:
            image_path (str): Path to the image file
            name (str): Person's name/ID
            validate_quality (bool): Whether to validate face quality
            
        Returns:
            dict: Registration result with success status and details
        """
        self.registration_stats['total_attempts'] += 1
        
        try:
            # Load image
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f"Image file not found: {image_path}",
                    'name': name
                }
                
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': f"Could not load image: {image_path}",
                    'name': name
                }
                
            # Detect faces
            faces = self.detector.detect_faces(image)
            
            if not faces:
                self.registration_stats['failed_registrations'] += 1
                return {
                    'success': False,
                    'error': "No faces detected in image",
                    'name': name
                }
                
            if len(faces) > 1:
                print(f"⚠️  Multiple faces detected, using the largest one")
                
            # Use the largest face
            face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
            
            # Validate face quality if requested
            if validate_quality:
                is_valid, quality_score, issues = self.validate_face_quality(face)
                if not is_valid:
                    self.registration_stats['failed_registrations'] += 1
                    return {
                        'success': False,
                        'error': f"Face quality validation failed: {', '.join(issues)}",
                        'name': name,
                        'quality_score': quality_score
                    }
                    
            # Check if person already exists
            existing_stats = self.recognizer.get_database_stats()
            if name in existing_stats['persons']:
                self.registration_stats['duplicate_attempts'] += 1
                return {
                    'success': False,
                    'error': f"Person '{name}' already exists in database",
                    'name': name,
                    'is_duplicate': True
                }
                
            # Register the face
            success = self.recognizer.add_person(name, face_image=face['face_crop'])
            
            if success:
                # Save database
                self.recognizer.save_database()
                self.registration_stats['successful_registrations'] += 1
                
                return {
                    'success': True,
                    'name': name,
                    'face_box': face['box'],
                    'confidence': face['confidence'],
                    'quality_score': quality_score if validate_quality else None,
                    'image_path': image_path
                }
            else:
                self.registration_stats['failed_registrations'] += 1
                return {
                    'success': False,
                    'error': "Failed to add person to database",
                    'name': name
                }
                
        except Exception as e:
            self.registration_stats['failed_registrations'] += 1
            return {
                'success': False,
                'error': f"Registration error: {str(e)}",
                'name': name
            }
            
    def register_from_camera(self, name=None, capture_count=1):
        """
        Register face(s) from live camera feed.
        
        Args:
            name (str): Person's name (if None, will prompt for each capture)
            capture_count (int): Number of images to capture for this person
            
        Returns:
            list: List of registration results
        """
        print("📹 Starting camera for face registration...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return [{'success': False, 'error': 'Could not open camera'}]
            
        results = []
        captured_count = 0
        
        print("\nCamera Controls:")
        print("  - Press SPACE to capture face")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to restart current capture")
        print("-" * 30)
        
        try:
            while captured_count < capture_count:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Detect faces
                faces = self.detector.detect_faces(frame)
                
                # Draw face detection results
                display_frame = self.detector.draw_detections(frame, faces)
                
                # Add instructions overlay
                cv2.putText(display_frame, f"Capture {captured_count + 1}/{capture_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if faces:
                    cv2.putText(display_frame, "Press SPACE to capture", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No face detected", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow('Face Registration - Camera', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and faces:  # Space key
                    # Get name if not provided
                    current_name = name
                    if current_name is None:
                        cv2.destroyWindow('Face Registration - Camera')
                        current_name = input(f"Enter name for capture {captured_count + 1}: ").strip()
                        
                    if current_name:
                        # Use the largest face
                        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                        
                        # Validate and register
                        is_valid, quality_score, issues = self.validate_face_quality(face)
                        
                        if is_valid:
                            # Register the face
                            success = self.recognizer.add_person(current_name, face_image=face['face_crop'])
                            
                            if success:
                                self.recognizer.save_database()
                                captured_count += 1
                                
                                result = {
                                    'success': True,
                                    'name': current_name,
                                    'capture_number': captured_count,
                                    'confidence': face['confidence'],
                                    'quality_score': quality_score
                                }
                                results.append(result)
                                
                                print(f"✅ Captured {current_name} ({captured_count}/{capture_count})")
                                
                                # Brief pause to show success
                                cv2.putText(display_frame, "CAPTURED!", 
                                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                                cv2.imshow('Face Registration - Camera', display_frame)
                                cv2.waitKey(1000)
                                
                            else:
                                result = {
                                    'success': False,
                                    'error': 'Failed to register face',
                                    'name': current_name
                                }
                                results.append(result)
                                print(f"❌ Failed to register {current_name}")
                                
                        else:
                            result = {
                                'success': False,
                                'error': f"Quality validation failed: {', '.join(issues)}",
                                'name': current_name,
                                'quality_score': quality_score
                            }
                            results.append(result)
                            print(f"❌ Quality validation failed for {current_name}: {', '.join(issues)}")
                            
                elif key == ord('r'):
                    # Restart current capture
                    print("🔄 Restarting current capture...")
                    
        except KeyboardInterrupt:
            print("\n⚠️  Camera registration interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
        return results
        
    def batch_register_from_directory(self, directory_path, name_from_filename=True):
        """
        Register faces from all images in a directory.
        
        Args:
            directory_path (str): Path to directory containing images
            name_from_filename (bool): Use filename as person name
            
        Returns:
            list: List of registration results
        """
        if not os.path.exists(directory_path):
            return [{'success': False, 'error': f"Directory not found: {directory_path}"}]
            
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(directory_path).glob(f"*{ext}"))
            image_files.extend(Path(directory_path).glob(f"*{ext.upper()}"))
            
        if not image_files:
            return [{'success': False, 'error': f"No image files found in {directory_path}"}]
            
        print(f"📁 Found {len(image_files)} images for batch registration")
        
        results = []
        
        for i, image_file in enumerate(image_files, 1):
            # Determine name
            if name_from_filename:
                name = image_file.stem  # Filename without extension
            else:
                name = input(f"Enter name for {image_file.name}: ").strip()
                if not name:
                    print(f"⏭️  Skipping {image_file.name} (no name provided)")
                    continue
                    
            print(f"📸 Processing {i}/{len(image_files)}: {image_file.name} -> {name}")
            
            # Register the face
            result = self.register_face_from_image(str(image_file), name)
            results.append(result)
            
            # Print result
            if result['success']:
                print(f"✅ Successfully registered {name}")
            else:
                print(f"❌ Failed to register {name}: {result['error']}")
                
        return results
        
    def export_registration_report(self, results, output_path='registration_report.json'):
        """
        Export registration results to a JSON report.
        
        Args:
            results (list): List of registration results
            output_path (str): Path for the output report
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'statistics': self.registration_stats.copy(),
                'results': results,
                'summary': {
                    'total_processed': len(results),
                    'successful': sum(1 for r in results if r.get('success', False)),
                    'failed': sum(1 for r in results if not r.get('success', False)),
                    'duplicates': sum(1 for r in results if r.get('is_duplicate', False))
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"📊 Registration report saved to {output_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to save registration report: {e}")
            
    def print_registration_summary(self, results):
        """Print a summary of registration results."""
        print("\n📊 Registration Summary:")
        print("=" * 30)
        
        successful = sum(1 for r in results if r.get('success', False))
        failed = sum(1 for r in results if not r.get('success', False))
        duplicates = sum(1 for r in results if r.get('is_duplicate', False))
        
        print(f"Total processed: {len(results)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"🔄 Duplicates: {duplicates}")
        
        if results:
            # Show failed registrations
            failed_results = [r for r in results if not r.get('success', False)]
            if failed_results:
                print(f"\n❌ Failed registrations:")
                for result in failed_results[:5]:  # Show first 5
                    print(f"   {result.get('name', 'Unknown')}: {result.get('error', 'Unknown error')}")
                if len(failed_results) > 5:
                    print(f"   ... and {len(failed_results) - 5} more")
                    
        # Database stats
        if self.recognizer:
            db_stats = self.recognizer.get_database_stats()
            print(f"\n💾 Database now contains {db_stats['total_persons']} persons")
            
        print("=" * 30)


def main():
    """Main entry point for face registration."""
    parser = argparse.ArgumentParser(description='University Gate AI - Face Registration')
    
    parser.add_argument('--mode', choices=['camera', 'image', 'batch'], required=True,
                       help='Registration mode: camera, single image, or batch from directory')
    parser.add_argument('--image', type=str, help='Path to image file (for image mode)')
    parser.add_argument('--directory', type=str, help='Path to directory (for batch mode)')
    parser.add_argument('--name', type=str, help='Person name (for camera/image mode)')
    parser.add_argument('--count', type=int, default=1, help='Number of captures (for camera mode)')
    parser.add_argument('--confidence', type=float, default=0.9, help='Detection confidence threshold')
    parser.add_argument('--min-size', type=int, default=80, help='Minimum face size')
    parser.add_argument('--no-validation', action='store_true', help='Skip quality validation')
    parser.add_argument('--report', type=str, help='Save registration report to file')
    
    args = parser.parse_args()
    
    # Initialize registration system
    registration_system = FaceRegistrationSystem(
        confidence_threshold=args.confidence,
        min_face_size=args.min_size
    )
    
    if not registration_system.initialize_system():
        print("❌ Failed to initialize registration system")
        return
        
    results = []
    
    try:
        if args.mode == 'camera':
            # Camera registration
            results = registration_system.register_from_camera(
                name=args.name,
                capture_count=args.count
            )
            
        elif args.mode == 'image':
            # Single image registration
            if not args.image:
                print("❌ --image parameter required for image mode")
                return
                
            if not args.name:
                args.name = input("Enter person name: ").strip()
                
            if not args.name:
                print("❌ Person name is required")
                return
                
            result = registration_system.register_face_from_image(
                args.image, 
                args.name, 
                validate_quality=not args.no_validation
            )
            results = [result]
            
        elif args.mode == 'batch':
            # Batch registration from directory
            if not args.directory:
                print("❌ --directory parameter required for batch mode")
                return
                
            results = registration_system.batch_register_from_directory(args.directory)
            
        # Print summary
        registration_system.print_registration_summary(results)
        
        # Save report if requested
        if args.report:
            registration_system.export_registration_report(results, args.report)
            
    except KeyboardInterrupt:
        print("\n⚠️  Registration interrupted by user")
    except Exception as e:
        print(f"❌ Registration error: {e}")
        
    print("\n👥 Face registration completed")


if __name__ == "__main__":
    main()