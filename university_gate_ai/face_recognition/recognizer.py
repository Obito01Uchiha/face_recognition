"""
Face recognition module using FAISS for fast similarity search.
Optimized for real-time face recognition with large databases.
"""

import numpy as np
import faiss
import pickle
import os
import time
from typing import List, Dict, Optional, Tuple
from .face_utils import FaceEmbeddingExtractor, FaceDatabase


class FAISSFaceRecognizer:
    """
    High-performance face recognizer using FAISS for similarity search.
    Supports thousands of identities with sub-millisecond search times.
    """
    
    def __init__(self, embedding_extractor=None, database_path='database/student_faces.npy', 
                 faiss_index_path='database/faiss_index.pkl', similarity_threshold=0.6):
        """
        Initialize FAISS-based face recognizer.
        
        Args:
            embedding_extractor: FaceEmbeddingExtractor instance
            database_path (str): Path to face database file
            faiss_index_path (str): Path to FAISS index file
            similarity_threshold (float): Minimum similarity for positive match
        """
        self.embedding_extractor = embedding_extractor or FaceEmbeddingExtractor()
        self.database_path = database_path
        self.faiss_index_path = faiss_index_path
        self.similarity_threshold = similarity_threshold
        
        # FAISS index and metadata
        self.faiss_index = None
        self.id_to_name = {}  # Maps FAISS index ID to person name
        self.name_to_id = {}  # Maps person name to FAISS index ID
        self.embeddings = {}  # Store embeddings for backup/reconstruction
        self.embedding_dim = 512
        
        # Performance tracking
        self.search_times = []
        self.max_time_samples = 1000
        
        # Initialize FAISS index
        self._initialize_faiss_index()
        
        # Load existing data
        self._load_database()
        
    def _initialize_faiss_index(self):
        """Initialize FAISS index for similarity search."""
        try:
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            # This is more accurate than L2 distance for face embeddings
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # For larger databases, consider using IndexIVFFlat or IndexHNSWFlat
            # self.faiss_index = faiss.IndexIVFFlat(
            #     faiss.IndexFlatIP(self.embedding_dim), self.embedding_dim, 100
            # )
            
            print("FAISS index initialized successfully")
            
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
            self.faiss_index = None
            
    def add_person(self, name, face_image=None, embedding=None):
        """
        Add a person to the recognition database.
        
        Args:
            name (str): Person's name/ID
            face_image (numpy.ndarray): Face image (if embedding not provided)
            embedding (numpy.ndarray): Pre-computed face embedding
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        if self.faiss_index is None:
            print("FAISS index not initialized")
            return False
            
        # Extract embedding if not provided
        if embedding is None:
            if face_image is None:
                print("Either face_image or embedding must be provided")
                return False
                
            embedding = self.embedding_extractor.extract_embedding(face_image)
            if embedding is None:
                print(f"Failed to extract embedding for {name}")
                return False
                
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        # Check if person already exists
        if name in self.name_to_id:
            print(f"Person {name} already exists. Use update_person() to modify.")
            return False
            
        try:
            # Add to FAISS index
            embedding_2d = embedding.reshape(1, -1).astype(np.float32)
            self.faiss_index.add(embedding_2d)
            
            # Get the new ID (current total count - 1)
            new_id = self.faiss_index.ntotal - 1
            
            # Update mappings
            self.id_to_name[new_id] = name
            self.name_to_id[name] = new_id
            self.embeddings[name] = embedding
            
            print(f"Added {name} to database (ID: {new_id})")
            return True
            
        except Exception as e:
            print(f"Error adding person {name}: {e}")
            return False
            
    def remove_person(self, name):
        """
        Remove a person from the database.
        Note: FAISS doesn't support direct removal, so we rebuild the index.
        
        Args:
            name (str): Person's name to remove
            
        Returns:
            bool: True if successfully removed, False otherwise
        """
        if name not in self.name_to_id:
            print(f"Person {name} not found in database")
            return False
            
        try:
            # Remove from mappings
            old_id = self.name_to_id[name]
            del self.name_to_id[name]
            del self.id_to_name[old_id]
            del self.embeddings[name]
            
            # Rebuild FAISS index without this person
            self._rebuild_faiss_index()
            
            print(f"Removed {name} from database")
            return True
            
        except Exception as e:
            print(f"Error removing person {name}: {e}")
            return False
            
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from current embeddings."""
        try:
            # Initialize new index
            self._initialize_faiss_index()
            
            # Clear mappings
            self.id_to_name.clear()
            self.name_to_id.clear()
            
            # Re-add all embeddings
            for name, embedding in self.embeddings.items():
                embedding_2d = embedding.reshape(1, -1).astype(np.float32)
                self.faiss_index.add(embedding_2d)
                
                # Update mappings
                new_id = self.faiss_index.ntotal - 1
                self.id_to_name[new_id] = name
                self.name_to_id[name] = new_id
                
            print("FAISS index rebuilt successfully")
            
        except Exception as e:
            print(f"Error rebuilding FAISS index: {e}")
            
    def recognize_face(self, face_image=None, embedding=None, top_k=1):
        """
        Recognize a face and return the best matches.
        
        Args:
            face_image (numpy.ndarray): Face image to recognize
            embedding (numpy.ndarray): Pre-computed face embedding
            top_k (int): Number of top matches to return
            
        Returns:
            list: List of (name, similarity_score) tuples, sorted by similarity
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
            
        # Extract embedding if not provided
        if embedding is None:
            if face_image is None:
                return []
                
            embedding = self.embedding_extractor.extract_embedding(face_image)
            if embedding is None:
                return []
                
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        start_time = time.time()
        
        try:
            # Search in FAISS index
            embedding_2d = embedding.reshape(1, -1).astype(np.float32)
            similarities, indices = self.faiss_index.search(embedding_2d, min(top_k, self.faiss_index.ntotal))
            
            # Track search time
            search_time = time.time() - start_time
            self.search_times.append(search_time)
            if len(self.search_times) > self.max_time_samples:
                self.search_times.pop(0)
            
            # Process results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                    
                if similarity >= self.similarity_threshold:
                    name = self.id_to_name.get(idx, f"Unknown_{idx}")
                    results.append((name, float(similarity)))
                    
            return results
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return []
            
    def recognize_best_match(self, face_image=None, embedding=None):
        """
        Get the best matching person for a face.
        
        Args:
            face_image (numpy.ndarray): Face image to recognize
            embedding (numpy.ndarray): Pre-computed face embedding
            
        Returns:
            tuple: (name, similarity_score) or (None, 0.0) if no match
        """
        results = self.recognize_face(face_image, embedding, top_k=1)
        
        if results:
            return results[0]
        else:
            return None, 0.0
            
    def batch_recognize(self, face_images=None, embeddings=None, top_k=1):
        """
        Recognize multiple faces in batch for better performance.
        
        Args:
            face_images (list): List of face images
            embeddings (numpy.ndarray): Pre-computed embeddings (N, 512)
            top_k (int): Number of top matches per face
            
        Returns:
            list: List of recognition results for each face
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
            
        # Extract embeddings if not provided
        if embeddings is None:
            if face_images is None:
                return []
                
            embeddings = self.embedding_extractor.extract_embeddings_batch(face_images)
            if embeddings.size == 0:
                return []
                
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        try:
            # Batch search in FAISS
            similarities, indices = self.faiss_index.search(
                embeddings.astype(np.float32), 
                min(top_k, self.faiss_index.ntotal)
            )
            
            # Process results for each face
            batch_results = []
            for face_similarities, face_indices in zip(similarities, indices):
                face_results = []
                for similarity, idx in zip(face_similarities, face_indices):
                    if idx == -1 or similarity < self.similarity_threshold:
                        continue
                        
                    name = self.id_to_name.get(idx, f"Unknown_{idx}")
                    face_results.append((name, float(similarity)))
                    
                batch_results.append(face_results)
                
            return batch_results
            
        except Exception as e:
            print(f"Error in batch recognition: {e}")
            return []
            
    def update_similarity_threshold(self, threshold):
        """Update the similarity threshold for recognition."""
        self.similarity_threshold = threshold
        print(f"Updated similarity threshold to {threshold}")
        
    def get_database_stats(self):
        """Get statistics about the face database."""
        stats = {
            'total_persons': len(self.embeddings),
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'embedding_dimension': self.embedding_dim,
            'similarity_threshold': self.similarity_threshold,
            'persons': list(self.embeddings.keys())
        }
        
        # Add performance stats if available
        if self.search_times:
            stats.update({
                'avg_search_time_ms': np.mean(self.search_times) * 1000,
                'search_fps': 1.0 / np.mean(self.search_times) if self.search_times else 0,
                'search_samples': len(self.search_times)
            })
            
        return stats
        
    def save_database(self):
        """Save the face database and FAISS index to disk."""
        try:
            # Save face database (embeddings and mappings)
            database_data = {
                'embeddings': self.embeddings,
                'id_to_name': self.id_to_name,
                'name_to_id': self.name_to_id,
                'similarity_threshold': self.similarity_threshold,
                'embedding_dim': self.embedding_dim
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            
            np.save(self.database_path, database_data)
            
            # Save FAISS index
            if self.faiss_index and self.faiss_index.ntotal > 0:
                os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
                faiss.write_index(self.faiss_index, self.faiss_index_path.replace('.pkl', '.index'))
                
            print(f"Database saved successfully")
            return True
            
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
            
    def _load_database(self):
        """Load the face database and FAISS index from disk."""
        try:
            # Load face database
            if os.path.exists(self.database_path):
                data = np.load(self.database_path, allow_pickle=True).item()
                
                self.embeddings = data.get('embeddings', {})
                self.id_to_name = data.get('id_to_name', {})
                self.name_to_id = data.get('name_to_id', {})
                self.similarity_threshold = data.get('similarity_threshold', self.similarity_threshold)
                self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
                
                print(f"Loaded database with {len(self.embeddings)} persons")
                
                # Rebuild FAISS index from embeddings
                if self.embeddings:
                    self._rebuild_faiss_index()
                    
            # Try to load FAISS index directly (if available)
            faiss_index_file = self.faiss_index_path.replace('.pkl', '.index')
            if os.path.exists(faiss_index_file) and self.faiss_index and self.faiss_index.ntotal > 0:
                try:
                    loaded_index = faiss.read_index(faiss_index_file)
                    if loaded_index.ntotal == self.faiss_index.ntotal:
                        self.faiss_index = loaded_index
                        print("FAISS index loaded from disk")
                except Exception as e:
                    print(f"Could not load FAISS index, using rebuilt index: {e}")
                    
        except Exception as e:
            print(f"Error loading database: {e}")
            
    def clear_database(self):
        """Clear all data from the database."""
        self.embeddings.clear()
        self.id_to_name.clear()
        self.name_to_id.clear()
        self._initialize_faiss_index()
        print("Database cleared")
        
    def export_embeddings(self, output_path):
        """Export embeddings to a standard format for backup."""
        try:
            export_data = {
                'embeddings': self.embeddings,
                'metadata': {
                    'total_persons': len(self.embeddings),
                    'embedding_dim': self.embedding_dim,
                    'export_timestamp': time.time()
                }
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(export_data, f)
                
            print(f"Embeddings exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting embeddings: {e}")
            return False
            
    def import_embeddings(self, input_path):
        """Import embeddings from a backup file."""
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
                
            imported_embeddings = data.get('embeddings', {})
            
            # Add each person to the database
            for name, embedding in imported_embeddings.items():
                self.add_person(name, embedding=embedding)
                
            print(f"Imported {len(imported_embeddings)} persons from {input_path}")
            return True
            
        except Exception as e:
            print(f"Error importing embeddings: {e}")
            return False


def test_faiss_recognizer():
    """Test function for FAISS face recognizer."""
    print("Testing FAISS face recognizer...")
    
    # Initialize components
    from .detector import FaceDetector
    
    detector = FaceDetector(confidence_threshold=0.8)
    recognizer = FAISSFaceRecognizer()
    
    if detector.detector is None:
        print("Failed to initialize detector")
        return
        
    # Test with webcam
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current face")
    print("- Press 'c' to clear database")
    print("- Press 'i' to show database info")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Recognize each face
            for face in faces:
                face_crop = face['face_crop']
                x, y, w, h = face['box']
                
                # Recognize face
                name, similarity = recognizer.recognize_best_match(face_image=face_crop)
                
                # Draw results
                if name:
                    color = (0, 255, 0)  # Green for recognized
                    label = f"{name} ({similarity:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unknown
                    label = "Unknown"
                    
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            # Show performance stats
            stats = recognizer.get_database_stats()
            info_text = f"Persons: {stats['total_persons']}"
            if 'search_fps' in stats:
                info_text += f" | Search FPS: {stats['search_fps']:.1f}"
                
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('FAISS Face Recognition Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and faces:
                # Save the first detected face
                face_crop = faces[0]['face_crop']
                name = input("Enter name for this face: ")
                if name.strip():
                    success = recognizer.add_person(name.strip(), face_image=face_crop)
                    if success:
                        recognizer.save_database()
                        print(f"Saved {name} to database")
            elif key == ord('c'):
                recognizer.clear_database()
                print("Database cleared")
            elif key == ord('i'):
                stats = recognizer.get_database_stats()
                print("Database Stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("FAISS face recognition test completed")


if __name__ == "__main__":
    test_faiss_recognizer()