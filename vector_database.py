import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from pathlib import Path
import torch

class VectorDatabase:
    """
    A vector database for storing and retrieving 3D model features and descriptions.
    Uses sentence embeddings for semantic search and FAISS for efficient similarity search.
    """
    
    def __init__(self, db_path="vector_db", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to the vector database
            model_name: Name of the sentence transformer model to use
        """
        self.db_path = db_path
        self.model_name = model_name
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Load sentence transformer model
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize empty database
        self.texts = []  # Textual descriptions
        self.features_3d = []  # 3D geometric features
        self.metadata = []  # Metadata for each entry
        self.index = None  # FAISS index
        
        # Check if CUDA is available for faster processing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.model = self.model.to(self.device)
            print("Using GPU for vector encoding")
        
    def add_from_analysis_file(self, analysis_file):
        """
        Add 3D model analysis data from a JSON file.
        
        Args:
            analysis_file: Path to the analysis JSON file
        """
        print(f"Adding data from {analysis_file}")
        
        try:
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            # Check if this is a model analysis file
            if 'geometric_features' in analysis_data:
                self._add_from_model_analysis(analysis_data)
            # Check if this is a scene graph file
            elif 'nodes' in analysis_data and 'edges' in analysis_data:
                self._add_from_scene_graph(analysis_data)
            # Check if this is an image analysis file (for backward compatibility)
            elif 'image_path' in analysis_data or 'description' in analysis_data:
                self._add_from_image_analysis(analysis_data)
            else:
                print(f"Warning: Unknown analysis file format: {analysis_file}")
        except Exception as e:
            print(f"Error processing {analysis_file}: {e}")
    
    def _add_from_model_analysis(self, analysis_data):
        """Add data from a model analysis file"""
        # Extract geometric features
        features = analysis_data.get('geometric_features', {})
        
        # Create a textual description from the geometric features
        description = self._generate_description_from_features(features)
        
        # Add to database
        self.texts.append(description)
        self.features_3d.append(features)
        
        # Add metadata
        metadata = {
            'type': '3d_features',
            'timestamp': analysis_data.get('timestamp', ''),
            'model_info': analysis_data.get('model_info', {}),
            'feature_source': 'geometric_analysis'
        }
        self.metadata.append(metadata)
        
        print(f"Added 3D geometric features with description: {description[:100]}...")
    
    def _add_from_scene_graph(self, scene_graph):
        """Add data from a scene graph file"""
        # Extract nodes and edges
        nodes = scene_graph.get('nodes', [])
        edges = scene_graph.get('edges', [])
        
        # Create a textual description from the scene graph
        description = self._generate_description_from_scene_graph(nodes, edges)
        
        # Add to database
        self.texts.append(description)
        
        # Create a feature vector from the scene graph
        features = {
            'node_count': len(nodes),
            'edge_count': len(edges),
            'node_types': {},
            'edge_types': {},
            'spatial_relationships': {}
        }
        
        # Count node types
        for node in nodes:
            node_type = node.get('type', 'unknown')
            if node_type not in features['node_types']:
                features['node_types'][node_type] = 0
            features['node_types'][node_type] += 1
        
        # Count edge types and spatial relationships
        for edge in edges:
            edge_type = edge.get('type', 'unknown')
            if edge_type not in features['edge_types']:
                features['edge_types'][edge_type] = 0
            features['edge_types'][edge_type] += 1
            
            # Track spatial relationships
            if edge_type in ['above', 'below', 'east_of', 'west_of', 'north_of', 'south_of']:
                if edge_type not in features['spatial_relationships']:
                    features['spatial_relationships'][edge_type] = 0
                features['spatial_relationships'][edge_type] += 1
        
        self.features_3d.append(features)
        
        # Add metadata
        metadata = {
            'type': 'scene_graph',
            'node_count': len(nodes),
            'edge_count': len(edges),
            'feature_source': 'scene_graph_analysis'
        }
        self.metadata.append(metadata)
        
        print(f"Added scene graph with {len(nodes)} nodes and {len(edges)} edges")
    
    def _add_from_image_analysis(self, analysis_data):
        """Add data from an image analysis file (for backward compatibility)"""
        # Check if this is a batch of analyses or a single analysis
        if isinstance(analysis_data, list):
            for item in analysis_data:
                self._add_single_image_analysis(item)
        else:
            self._add_single_image_analysis(analysis_data)
    
    def _add_single_image_analysis(self, analysis_data):
        """Add a single image analysis to the database"""
        # Extract description
        description = analysis_data.get('description', '')
        if not description:
            return
        
        # Add to database
        self.texts.append(description)
        
        # Create a placeholder for 3D features
        features = {
            'source': 'image_analysis',
            'is_2d': True
        }
        self.features_3d.append(features)
        
        # Add metadata
        metadata = {
            'type': 'image_description',
            'image_path': analysis_data.get('image_path', ''),
            'feature_source': 'image_analysis'
        }
        self.metadata.append(metadata)
    
    def _generate_description_from_features(self, features):
        """Generate a textual description from geometric features"""
        description_parts = []
        
        # Bounding box dimensions
        if 'bbox_dimensions' in features:
            dims = features['bbox_dimensions']
            description_parts.append(f"This is a 3D model with dimensions approximately {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} units.")
        
        # Volume and surface area
        if 'volume' in features and features['volume'] > 0:
            description_parts.append(f"The model has a volume of approximately {features['volume']:.2f} cubic units.")
        
        if 'surface_area' in features:
            description_parts.append(f"The surface area is approximately {features['surface_area']:.2f} square units.")
        
        # Watertightness
        if 'is_watertight' in features:
            if features['is_watertight']:
                description_parts.append("The model is watertight (closed).")
            else:
                description_parts.append("The model is not watertight (has holes or openings).")
        
        # Curvature
        if 'mean_curvature' in features:
            if features['mean_curvature'] < 0.01:
                description_parts.append("The model consists mostly of flat surfaces.")
            elif features['mean_curvature'] < 0.05:
                description_parts.append("The model has some curved surfaces.")
            else:
                description_parts.append("The model has many curved or complex surfaces.")
        
        # PCA eigenvalues (shape elongation)
        if 'pca_eigenvalues' in features:
            eigenvalues = features['pca_eigenvalues']
            if len(eigenvalues) >= 3:
                # Check if one dimension is much larger than others (elongated)
                if eigenvalues[0] > 3 * eigenvalues[1]:
                    description_parts.append("The model is elongated in one dimension.")
                # Check if two dimensions are much larger than the third (flat)
                elif eigenvalues[1] > 3 * eigenvalues[2]:
                    description_parts.append("The model is relatively flat or planar.")
                # All dimensions similar (cubic/spherical)
                elif eigenvalues[0] < 2 * eigenvalues[2]:
                    description_parts.append("The model has similar dimensions in all directions (roughly cubic or spherical).")
        
        # Combine all parts
        description = " ".join(description_parts)
        
        return description
    
    def _generate_description_from_scene_graph(self, nodes, edges):
        """Generate a textual description from a scene graph"""
        description_parts = []
        
        # Count node types
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
        
        # Describe node composition
        if 'component' in node_types:
            description_parts.append(f"The model consists of {node_types['component']} distinct components.")
        
        if 'region' in node_types:
            description_parts.append(f"The model space is divided into {node_types['region']} regions.")
        
        # Count relationship types
        relationship_counts = {}
        for edge in edges:
            edge_type = edge.get('type', 'unknown')
            if edge_type not in relationship_counts:
                relationship_counts[edge_type] = 0
            relationship_counts[edge_type] += 1
        
        # Describe spatial relationships
        spatial_relationships = ['above', 'below', 'east_of', 'west_of', 'north_of', 'south_of']
        has_spatial = any(rel in relationship_counts for rel in spatial_relationships)
        
        if has_spatial:
            description_parts.append("The model has clear spatial organization with components arranged in relation to each other.")
        
        if 'has_component' in relationship_counts:
            description_parts.append(f"The model has {relationship_counts['has_component']} hierarchical component relationships.")
        
        # Combine all parts
        description = " ".join(description_parts)
        
        # If description is empty, provide a default
        if not description:
            description = "This is a 3D model with a scene graph representation."
        
        return description
    
    def build_index(self):
        """Build a FAISS index for efficient similarity search"""
        if not self.texts:
            print("No texts to index")
            return False
        
        print(f"Building index for {len(self.texts)} descriptions...")
        
        # Encode texts into vectors
        print("Encoding texts...")
        vectors = self.model.encode(self.texts, show_progress_bar=True)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Create FAISS index
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(vectors)
        
        print(f"Index built with {len(self.texts)} vectors of dimension {dimension}")
        return True
    
    def save(self):
        """Save the vector database to disk"""
        if not self.index:
            print("No index to save")
            return False
        
        # Create paths
        index_path = os.path.join(self.db_path, "index.faiss")
        data_path = os.path.join(self.db_path, "data.pkl")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save texts and metadata
        with open(data_path, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'features_3d': self.features_3d,
                'metadata': self.metadata,
                'model_name': self.model_name
            }, f)
        
        print(f"Vector database saved to {self.db_path}")
        return True
    
    def load(self):
        """Load the vector database from disk"""
        # Check if files exist
        index_path = os.path.join(self.db_path, "index.faiss")
        data_path = os.path.join(self.db_path, "data.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(data_path):
            print(f"Database files not found in {self.db_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load texts and metadata
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.texts = data['texts']
            self.metadata = data['metadata']
            
            # Handle 3D features (may not exist in older versions)
            if 'features_3d' in data:
                self.features_3d = data['features_3d']
            else:
                # Create empty features for backward compatibility
                self.features_3d = [{} for _ in range(len(self.texts))]
            
            # Check if model name matches
            if data.get('model_name', '') != self.model_name:
                print(f"Warning: Loaded model name ({data.get('model_name', '')}) "
                      f"doesn't match current model ({self.model_name})")
            
            print(f"Loaded vector database with {len(self.texts)} descriptions.")
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False
    
    def search(self, query, k=5, include_features=False):
        """
        Search the vector database for similar descriptions.
        
        Args:
            query: The query text
            k: Number of results to return
            include_features: Whether to include 3D features in results
            
        Returns:
            List of dictionaries with search results
        """
        if not self.index:
            print("No index available. Please build or load an index first.")
            return []
        
        # Encode query
        query_vector = self.model.encode([query])[0]
        
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_vector, min(k, len(self.texts)))
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
                
            result = {
                'text': self.texts[idx],
                'score': float(distances[0][i]),
                'metadata': self.metadata[idx]
            }
            
            # Include 3D features if requested
            if include_features and idx < len(self.features_3d):
                result['features_3d'] = self.features_3d[idx]
            
            results.append(result)
        
        return results
    
    def search_by_feature(self, feature_query, k=5):
        """
        Search for models with specific 3D feature characteristics.
        
        Args:
            feature_query: Dictionary of feature constraints
            k: Number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        if not self.features_3d:
            print("No 3D features available in the database.")
            return []
        
        # Filter results based on feature constraints
        matches = []
        
        for idx, features in enumerate(self.features_3d):
            match_score = 0
            total_constraints = 0
            
            for key, value in feature_query.items():
                if key not in features:
                    continue
                    
                total_constraints += 1
                
                # Handle different types of constraints
                if isinstance(value, dict) and 'min' in value and 'max' in value:
                    # Range constraint
                    if value['min'] <= features[key] <= value['max']:
                        match_score += 1
                elif isinstance(value, dict) and 'min' in value:
                    # Minimum constraint
                    if features[key] >= value['min']:
                        match_score += 1
                elif isinstance(value, dict) and 'max' in value:
                    # Maximum constraint
                    if features[key] <= value['max']:
                        match_score += 1
                elif isinstance(value, (list, tuple)):
                    # List of possible values
                    if features[key] in value:
                        match_score += 1
                else:
                    # Exact match
                    if features[key] == value:
                        match_score += 1
            
            # Calculate match percentage
            if total_constraints > 0:
                match_percentage = match_score / total_constraints
                
                if match_score > 0:
                    matches.append({
                        'index': idx,
                        'score': match_percentage
                    })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Format results
        results = []
        for match in matches[:k]:
            idx = match['index']
            result = {
                'text': self.texts[idx],
                'score': match['score'],
                'metadata': self.metadata[idx],
                'features_3d': self.features_3d[idx]
            }
            results.append(result)
        
        return results

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
        db.add_from_analysis_file(file_path)
        total_count += 1
    
    if total_count > 0:
        # Build index
        db.build_index()
        
        # Save database
        db.save()
        
        print(f"Vector database created with {total_count} descriptions at {db_path}")
    else:
        print("No descriptions found in analysis files.")
    
    return db_path 