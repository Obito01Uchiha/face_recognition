"""
Face utilities for embedding extraction and preprocessing.
Supports multiple face recognition models including FaceNet and InsightFace.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import pickle
import os
from typing import List, Dict, Optional, Tuple


class FaceEmbeddingExtractor:
    """
    Face embedding extraction using FaceNet or InsightFace models.
    Converts face images to 512-dimensional feature vectors.
    """
    
    def __init__(self, model_type='facenet', device='cpu', pretrained=True):
        """
        Initialize the face embedding extractor.
        
        Args:
            model_type (str): Type of model ('facenet' or 'insightface')
            device (str): Device to run inference on ('cpu' or 'cuda')
            pretrained (bool): Use pretrained weights
        """
        self.model_type = model_type
        self.device = device
        self.model = None
        self.embedding_size = 512
        
        print(f"Loading {model_type} face embedding model...")
        
        if model_type == 'facenet':
            self._load_facenet_model(pretrained)
        elif model_type == 'insightface':
            self._load_insightface_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def _load_facenet_model(self, pretrained=True):
        """Load FaceNet model using facenet-pytorch."""
        try:
            # Load pretrained FaceNet model
            self.model = InceptionResnetV1(
                pretrained='vggface2' if pretrained else None,
                device=self.device
            ).eval()
            
            # Move to device
            self.model = self.model.to(self.device)
            
            print("FaceNet model loaded successfully")
            
        except Exception as e:
            print(f"Error loading FaceNet model: {e}")
            self.model = None
            
    def _load_insightface_model(self):
        """Load InsightFace model (placeholder for future implementation)."""
        try:
            # This would load InsightFace model
            # For now, we'll use FaceNet as fallback
            print("InsightFace not implemented yet, falling back to FaceNet")
            self._load_facenet_model(pretrained=True)
            
        except Exception as e:
            print(f"Error loading InsightFace model: {e}")
            self.model = None
            
    def preprocess_face(self, face_image, target_size=(160, 160)):
        """
        Preprocess face image for embedding extraction.
        
        Args:
            face_image (numpy.ndarray): Face image in BGR format
            target_size (tuple): Target size for resizing
            
        Returns:
            torch.Tensor: Preprocessed face tensor
        """
        if face_image is None or face_image.size == 0:
            return None
            
        try:
            # Convert BGR to RGB
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
                
            # Resize to target size
            resized = cv2.resize(rgb_image, target_size)
            
            # Convert to PIL Image for better compatibility
            pil_image = Image.fromarray(resized)
            
            # Convert to tensor and normalize
            tensor_image = torch.tensor(np.array(pil_image)).float()
            
            # Rearrange dimensions to (C, H, W)
            if len(tensor_image.shape) == 3:
                tensor_image = tensor_image.permute(2, 0, 1)
            else:
                # Handle grayscale
                tensor_image = tensor_image.unsqueeze(0)
                tensor_image = tensor_image.repeat(3, 1, 1)
                
            # Normalize to [-1, 1] range (standard for FaceNet)
            tensor_image = (tensor_image - 127.5) / 128.0
            
            # Add batch dimension
            tensor_image = tensor_image.unsqueeze(0)
            
            # Move to device
            tensor_image = tensor_image.to(self.device)
            
            return tensor_image
            
        except Exception as e:
            print(f"Error in face preprocessing: {e}")
            return None
            
    def extract_embedding(self, face_image):
        """
        Extract embedding from a face image.
        
        Args:
            face_image (numpy.ndarray): Face image in BGR format
            
        Returns:
            numpy.ndarray: 512-dimensional face embedding, or None if failed
        """
        if self.model is None:
            print("Model not loaded")
            return None
            
        # Preprocess the face
        face_tensor = self.preprocess_face(face_image)
        if face_tensor is None:
            return None
            
        try:
            with torch.no_grad():
                # Extract embedding
                embedding = self.model(face_tensor)
                
                # Normalize embedding (L2 normalization)
                embedding = F.normalize(embedding, p=2, dim=1)
                
                # Convert to numpy array
                embedding_np = embedding.cpu().numpy().flatten()
                
                return embedding_np
                
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
            
    def extract_embeddings_batch(self, face_images):
        """
        Extract embeddings from multiple face images in batch.
        
        Args:
            face_images (list): List of face images in BGR format
            
        Returns:
            numpy.ndarray: Array of embeddings with shape (N, 512)
        """
        if self.model is None or not face_images:
            return np.array([])
            
        embeddings = []
        batch_tensors = []
        
        # Preprocess all faces
        for face_image in face_images:
            face_tensor = self.preprocess_face(face_image)
            if face_tensor is not None:
                batch_tensors.append(face_tensor)
                
        if not batch_tensors:
            return np.array([])
            
        try:
            # Concatenate all tensors into a batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            with torch.no_grad():
                # Extract embeddings for the batch
                batch_embeddings = self.model(batch_tensor)
                
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                # Convert to numpy
                embeddings_np = batch_embeddings.cpu().numpy()
                
                return embeddings_np
                
        except Exception as e:
            print(f"Error extracting batch embeddings: {e}")
            return np.array([])
            
    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (numpy.ndarray): First embedding
            embedding2 (numpy.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
            
    def calculate_distance(self, embedding1, embedding2):
        """
        Calculate Euclidean distance between two embeddings.
        
        Args:
            embedding1 (numpy.ndarray): First embedding
            embedding2 (numpy.ndarray): Second embedding
            
        Returns:
            float: Euclidean distance
        """
        try:
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)
            
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return float('inf')


class FaceDatabase:
    """
    Database for storing and managing face embeddings with names.
    Supports saving/loading from disk and efficient similarity search.
    """
    
    def __init__(self, database_path='database/student_faces.npy'):
        """
        Initialize face database.
        
        Args:
            database_path (str): Path to save/load the database file
        """
        self.database_path = database_path
        self.embeddings = {}  # {name: embedding_vector}
        self.embedding_size = 512
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        
        # Load existing database if available
        self.load_database()
        
    def add_person(self, name, embedding):
        """
        Add a person's face embedding to the database.
        
        Args:
            name (str): Person's name/ID
            embedding (numpy.ndarray): Face embedding vector
        """
        if embedding is None or len(embedding) != self.embedding_size:
            print(f"Invalid embedding for {name}")
            return False
            
        self.embeddings[name] = embedding.astype(np.float32)
        print(f"Added {name} to database")
        return True
        
    def remove_person(self, name):
        """
        Remove a person from the database.
        
        Args:
            name (str): Person's name/ID to remove
        """
        if name in self.embeddings:
            del self.embeddings[name]
            print(f"Removed {name} from database")
            return True
        else:
            print(f"{name} not found in database")
            return False
            
    def find_best_match(self, query_embedding, threshold=0.6):
        """
        Find the best matching person for a query embedding.
        
        Args:
            query_embedding (numpy.ndarray): Query face embedding
            threshold (float): Minimum similarity threshold for a match
            
        Returns:
            tuple: (name, similarity_score) or (None, 0.0) if no match
        """
        if not self.embeddings or query_embedding is None:
            return None, 0.0
            
        best_name = None
        best_similarity = 0.0
        
        for name, stored_embedding in self.embeddings.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_name = name
                
        return best_name, best_similarity
        
    def find_all_matches(self, query_embedding, threshold=0.6, top_k=5):
        """
        Find all matching persons above threshold, sorted by similarity.
        
        Args:
            query_embedding (numpy.ndarray): Query face embedding
            threshold (float): Minimum similarity threshold
            top_k (int): Maximum number of matches to return
            
        Returns:
            list: List of (name, similarity_score) tuples, sorted by similarity
        """
        if not self.embeddings or query_embedding is None:
            return []
            
        matches = []
        
        for name, stored_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            
            if similarity >= threshold:
                matches.append((name, similarity))
                
        # Sort by similarity (descending) and return top_k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
        
    def _cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
            
    def save_database(self):
        """Save the database to disk."""
        try:
            # Convert to format suitable for numpy save
            data = {
                'embeddings': self.embeddings,
                'embedding_size': self.embedding_size
            }
            
            np.save(self.database_path, data)
            print(f"Database saved to {self.database_path}")
            return True
            
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
            
    def load_database(self):
        """Load the database from disk."""
        try:
            if os.path.exists(self.database_path):
                data = np.load(self.database_path, allow_pickle=True).item()
                
                self.embeddings = data.get('embeddings', {})
                self.embedding_size = data.get('embedding_size', 512)
                
                print(f"Loaded database with {len(self.embeddings)} persons")
                return True
            else:
                print("No existing database found, starting fresh")
                return False
                
        except Exception as e:
            print(f"Error loading database: {e}")
            self.embeddings = {}
            return False
            
    def get_database_stats(self):
        """Get statistics about the database."""
        return {
            'total_persons': len(self.embeddings),
            'embedding_size': self.embedding_size,
            'database_path': self.database_path,
            'persons': list(self.embeddings.keys())
        }
        
    def clear_database(self):
        """Clear all entries from the database."""
        self.embeddings.clear()
        print("Database cleared")


def test_embedding_extractor():
    """Test function for face embedding extraction."""
    print("Testing face embedding extractor...")
    
    # Initialize extractor
    extractor = FaceEmbeddingExtractor(model_type='facenet', device='cpu')
    
    if extractor.model is None:
        print("Failed to load model")
        return
        
    # Test with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    print("Press 'q' to quit, 's' to save current face")
    
    database = FaceDatabase()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Simple face detection (you would use the detector module here)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Extract face
                face_crop = frame[y:y+h, x:x+w]
                
                # Extract embedding
                embedding = extractor.extract_embedding(face_crop)
                
                if embedding is not None:
                    # Find match in database
                    name, similarity = database.find_best_match(embedding, threshold=0.6)
                    
                    # Draw bounding box and name
                    color = (0, 255, 0) if name else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    label = f"{name} ({similarity:.2f})" if name else "Unknown"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
            cv2.imshow('Face Recognition Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(faces) > 0:
                # Save the first detected face
                x, y, w, h = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                embedding = extractor.extract_embedding(face_crop)
                
                if embedding is not None:
                    name = input("Enter name for this face: ")
                    database.add_person(name, embedding)
                    database.save_database()
                    print(f"Saved {name} to database")
                    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Face embedding test completed")


if __name__ == "__main__":
    test_embedding_extractor()