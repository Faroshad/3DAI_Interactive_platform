import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

class VectorDatabase:
    """
    Vector database for storing and retrieving 3D model descriptions.
    Uses sentence embeddings to enable semantic search.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2", db_path="vector_db"):
        """
        Initialize the vector database.
        
        Args:
            model_name: Name of the sentence transformer model to use
            db_path: Path to store the vector database files
        """
        self.model_name = model_name
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Load or initialize the sentence transformer model
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize empty database
        self.index = None
        self.texts = []
        self.metadata = []
        
        print(f"Vector database initialized at {db_path}")
    
    def add_from_analysis_file(self, analysis_file):
        """
        Add descriptions from an image analysis JSON file to the database.
        
        Args:
            analysis_file: Path to the image analysis JSON file
            
        Returns:
            Number of descriptions added
        """
        if not os.path.exists(analysis_file):
            print(f"Analysis file not found: {analysis_file}")
            return 0
        
        try:
            with open(analysis_file, "r") as f:
                analysis_data = json.load(f)
            
            count = 0
            
            # Handle both single image analysis and batch analysis formats
            if isinstance(analysis_data, list):
                image_analyses = analysis_data
            else:
                image_analyses = [analysis_data]
            
            for image_analysis in image_analyses:
                if "error" in image_analysis:
                    continue
                
                for description in image_analysis.get("descriptions", []):
                    text = description.get("text", "").strip()
                    
                    if text:
                        # Create metadata for this description
                        meta = {
                            "image_path": image_analysis.get("image_path", ""),
                            "filename": image_analysis.get("filename", ""),
                            "source": description.get("source", "unknown"),
                            "width": image_analysis.get("width", 0),
                            "height": image_analysis.get("height", 0)
                        }
                        
                        # Add to database
                        self.texts.append(text)
                        self.metadata.append(meta)
                        count += 1
            
            print(f"Added {count} descriptions from {analysis_file}")
            return count
        except Exception as e:
            print(f"Error adding descriptions from analysis file: {e}")
            return 0
    
    def build_index(self):
        """
        Build the FAISS index from the current texts.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.texts:
            print("No texts to index.")
            return False
        
        try:
            print(f"Building index for {len(self.texts)} descriptions...")
            
            # Encode texts to vectors
            vectors = []
            
            # Process in batches to avoid memory issues
            batch_size = 32
            for i in tqdm(range(0, len(self.texts), batch_size), desc="Encoding texts"):
                batch = self.texts[i:i+batch_size]
                batch_vectors = self.model.encode(batch)
                vectors.extend(batch_vectors)
            
            vectors = np.array(vectors).astype('float32')
            
            # Create FAISS index
            dimension = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(vectors)
            
            print(f"Index built with {len(self.texts)} descriptions.")
            return True
        except Exception as e:
            print(f"Error building index: {e}")
            return False
    
    def save(self):
        """
        Save the vector database to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.index:
            print("No index to save.")
            return False
        
        try:
            # Save index
            index_path = os.path.join(self.db_path, "index.faiss")
            faiss.write_index(self.index, index_path)
            
            # Save texts and metadata
            data = {
                "texts": self.texts,
                "metadata": self.metadata,
                "model_name": self.model_name
            }
            
            data_path = os.path.join(self.db_path, "data.pkl")
            with open(data_path, "wb") as f:
                pickle.dump(data, f)
            
            print(f"Vector database saved to {self.db_path}")
            return True
        except Exception as e:
            print(f"Error saving vector database: {e}")
            return False
    
    def load(self):
        """
        Load the vector database from disk.
        
        Returns:
            True if successful, False otherwise
        """
        index_path = os.path.join(self.db_path, "index.faiss")
        data_path = os.path.join(self.db_path, "data.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(data_path):
            print(f"Vector database files not found at {self.db_path}")
            return False
        
        try:
            # Load index
            self.index = faiss.read_index(index_path)
            
            # Load texts and metadata
            with open(data_path, "rb") as f:
                data = pickle.load(f)
            
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            
            # Check if model name matches
            if data["model_name"] != self.model_name:
                print(f"Warning: Loaded database uses model '{data['model_name']}', but current model is '{self.model_name}'")
            
            print(f"Loaded vector database with {len(self.texts)} descriptions.")
            return True
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return False
    
    def search(self, query, k=5):
        """
        Search the vector database for similar descriptions.
        
        Args:
            query: Text query to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        if not self.index:
            print("No index available. Build or load an index first.")
            return []
        
        try:
            # Encode query
            query_vector = self.model.encode([query]).astype('float32')
            
            # Search
            k = min(k, len(self.texts))
            if k == 0:
                return []
                
            D, I = self.index.search(query_vector, k)
            
            # Format results
            results = []
            
            for i, idx in enumerate(I[0]):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "score": float(D[0][i])
                })
            
            return results
        except Exception as e:
            print(f"Error searching vector database: {e}")
            return []

def create_vector_database(analysis_files, db_path="vector_db", model_name="all-MiniLM-L6-v2"):
    """
    Create a vector database from image analysis files.
    
    Args:
        analysis_files: List of paths to image analysis JSON files
        db_path: Path to store the vector database
        model_name: Name of the sentence transformer model to use
        
    Returns:
        Path to the vector database
    """
    # Create vector database
    db = VectorDatabase(model_name=model_name, db_path=db_path)
    
    # Add descriptions from analysis files
    total_count = 0
    
    for file_path in analysis_files:
        count = db.add_from_analysis_file(file_path)
        total_count += count
    
    if total_count > 0:
        # Build index
        db.build_index()
        
        # Save database
        db.save()
        
        print(f"Vector database created with {total_count} descriptions at {db_path}")
    else:
        print("No descriptions found in analysis files.")
    
    return db_path 