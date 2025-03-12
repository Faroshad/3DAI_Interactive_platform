import os
import json
import re
import requests
import numpy as np
from vector_database import VectorDatabase

# Check if GPT API key is available
GPT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

class ModelQueryEngine:
    """
    Query engine for answering questions about 3D models using geometric features,
    scene graphs, and vector embeddings.
    """
    
    def __init__(self, db_path="vector_db", use_gpt=False, max_gpt_calls=20):
        """
        Initialize the query engine.
        
        Args:
            db_path: Path to the vector database
            use_gpt: Whether to use GPT API for enhanced responses
            max_gpt_calls: Maximum number of GPT API calls to make (for cost control)
        """
        self.db_path = db_path
        
        # Set up OpenAI API key
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.use_gpt = use_gpt and self.api_key
        self.max_gpt_calls = max_gpt_calls
        self.gpt_calls_made = 0
        
        # Load vector database
        self.db = VectorDatabase(db_path=db_path)
        if not self.db.load():
            print("Warning: Failed to load vector database. Query capabilities will be limited.")
        
        if self.use_gpt:
            print("GPT API will be used for enhanced responses.")
        else:
            print("Using local response generation.")
    
    def query(self, question, k=5):
        """
        Answer a question about the 3D model.
        
        Args:
            question: The question to answer
            k: Number of relevant descriptions to retrieve
            
        Returns:
            Answer to the question
        """
        if not question:
            return "Please ask a valid question about the 3D model."
        
        # Analyze the question to determine the best approach
        query_type = self._analyze_query_type(question)
        
        # Handle different types of queries
        if query_type == "geometric":
            return self._handle_geometric_query(question, k)
        elif query_type == "spatial":
            return self._handle_spatial_query(question, k)
        elif query_type == "measurement":
            return self._handle_measurement_query(question, k)
        else:
            # Default to semantic search for general questions
            return self._handle_semantic_query(question, k)
    
    def _analyze_query_type(self, question):
        """Determine the type of query based on the question"""
        question = question.lower()
        
        # Geometric queries about shape, structure, etc.
        geometric_keywords = ["shape", "structure", "geometry", "form", "design", 
                             "curved", "flat", "symmetrical", "layout"]
        
        # Spatial queries about relationships between components
        spatial_keywords = ["above", "below", "next to", "adjacent", "connected", 
                           "inside", "outside", "between", "surrounding", "relation"]
        
        # Measurement queries about dimensions, distances, etc.
        measurement_keywords = ["dimension", "size", "height", "width", "length", 
                               "distance", "volume", "area", "measure", "how big", 
                               "how tall", "how wide", "how long"]
        
        # Count keyword occurrences
        geometric_count = sum(1 for keyword in geometric_keywords if keyword in question)
        spatial_count = sum(1 for keyword in spatial_keywords if keyword in question)
        measurement_count = sum(1 for keyword in measurement_keywords if keyword in question)
        
        # Determine the dominant query type
        if measurement_count > geometric_count and measurement_count > spatial_count:
            return "measurement"
        elif spatial_count > geometric_count:
            return "spatial"
        elif geometric_count > 0:
            return "geometric"
        else:
            return "semantic"
    
    def _handle_geometric_query(self, question, k=5):
        """Handle queries about geometric properties"""
        # Search for relevant 3D features
        results = self.db.search(question, k=k, include_features=True)
        
        if not results:
            return "I don't have enough information about the geometric properties of this 3D model."
        
        # Extract geometric features from results
        geometric_data = []
        for result in results:
            if 'features_3d' in result and result['features_3d']:
                geometric_data.append(result)
        
        # If we have geometric features, use them to answer the question
        if geometric_data:
            # If GPT API is available, use it for enhanced response
            if self.use_gpt and self.gpt_calls_made < self.max_gpt_calls:
                self.gpt_calls_made += 1
                return self._generate_gpt_response_with_features(question, geometric_data)
            else:
                return self._generate_local_geometric_response(question, geometric_data)
        else:
            # Fall back to semantic search
            return self._handle_semantic_query(question, k)
    
    def _handle_spatial_query(self, question, k=5):
        """Handle queries about spatial relationships"""
        # Search for scene graphs
        results = self.db.search(question, k=k, include_features=True)
        
        if not results:
            return "I don't have enough information about the spatial relationships in this 3D model."
        
        # Extract scene graph data from results
        scene_graph_data = []
        for result in results:
            if 'metadata' in result and result['metadata'].get('type') == 'scene_graph':
                scene_graph_data.append(result)
        
        # If we have scene graph data, use it to answer the question
        if scene_graph_data:
            # If GPT API is available, use it for enhanced response
            if self.use_gpt and self.gpt_calls_made < self.max_gpt_calls:
                self.gpt_calls_made += 1
                return self._generate_gpt_response_with_scene_graph(question, scene_graph_data)
            else:
                return self._generate_local_spatial_response(question, scene_graph_data)
        else:
            # Fall back to semantic search
            return self._handle_semantic_query(question, k)
    
    def _handle_measurement_query(self, question, k=5):
        """Handle queries about measurements and dimensions"""
        # Search for relevant 3D features with measurement data
        results = self.db.search(question, k=k, include_features=True)
        
        if not results:
            return "I don't have enough information about the measurements of this 3D model."
        
        # Extract measurement data from results
        measurement_data = []
        for result in results:
            if 'features_3d' in result and result['features_3d']:
                # Check if this result has measurement-related features
                features = result['features_3d']
                if any(key in features for key in ['bbox_dimensions', 'volume', 'surface_area']):
                    measurement_data.append(result)
        
        # If we have measurement data, use it to answer the question
        if measurement_data:
            # If GPT API is available, use it for enhanced response
            if self.use_gpt and self.gpt_calls_made < self.max_gpt_calls:
                self.gpt_calls_made += 1
                return self._generate_gpt_response_with_measurements(question, measurement_data)
            else:
                return self._generate_local_measurement_response(question, measurement_data)
        else:
            # Fall back to semantic search
            return self._handle_semantic_query(question, k)
    
    def _handle_semantic_query(self, question, k=5):
        """Handle general semantic queries using vector search"""
        # Search vector database for relevant descriptions
        if self.db.index is not None:
            results = self.db.search(question, k=k)
        else:
            results = []
        
        if not results:
            return "I don't have enough information to answer that question about the 3D model."
        
        # If GPT API is available and enabled, use it for enhanced response
        if self.use_gpt and self.gpt_calls_made < self.max_gpt_calls:
            self.gpt_calls_made += 1
            return self._generate_gpt_response(question, results)
        elif self.use_gpt and self.gpt_calls_made >= self.max_gpt_calls:
            print(f"Reached maximum GPT API calls ({self.max_gpt_calls}). Using local response generation.")
            return self._generate_local_response(question, results)
        else:
            return self._generate_local_response(question, results)
    
    def _generate_local_geometric_response(self, question, results):
        """Generate a response about geometric properties without using GPT"""
        # Extract the most relevant geometric features
        features = results[0]['features_3d']
        
        response = "Based on the 3D model's geometric analysis:\n\n"
        
        # Add information about shape
        if 'pca_eigenvalues' in features:
            eigenvalues = features['pca_eigenvalues']
            if len(eigenvalues) >= 3:
                if eigenvalues[0] > 3 * eigenvalues[1]:
                    response += "The model is elongated in one dimension, like a rod or column.\n"
                elif eigenvalues[1] > 3 * eigenvalues[2]:
                    response += "The model is relatively flat or planar, like a sheet or panel.\n"
                elif eigenvalues[0] < 2 * eigenvalues[2]:
                    response += "The model has similar dimensions in all directions, resembling a cube or sphere.\n"
        
        # Add information about curvature
        if 'mean_curvature' in features:
            if features['mean_curvature'] < 0.01:
                response += "The model consists primarily of flat surfaces with minimal curvature.\n"
            elif features['mean_curvature'] < 0.05:
                response += "The model has some curved surfaces, but is not highly complex.\n"
            else:
                response += "The model has many curved or complex surfaces.\n"
        
        # Add information about watertightness
        if 'is_watertight' in features:
            if features['is_watertight']:
                response += "The model is watertight (closed), with no holes or openings.\n"
            else:
                response += "The model is not watertight, suggesting it has openings or is incomplete.\n"
        
        return response
    
    def _generate_local_spatial_response(self, question, results):
        """Generate a response about spatial relationships without using GPT"""
        # Extract the most relevant scene graph data
        metadata = results[0]['metadata']
        text = results[0]['text']
        
        response = "Based on the 3D model's spatial analysis:\n\n"
        response += text + "\n\n"
        
        if 'node_count' in metadata and 'edge_count' in metadata:
            response += f"The model has {metadata['node_count']} components with {metadata['edge_count']} spatial relationships between them.\n"
        
        # Look for specific spatial relationships in the question
        question_lower = question.lower()
        spatial_terms = {
            "above": "positioned above",
            "below": "positioned below",
            "adjacent": "adjacent to",
            "connected": "connected to",
            "inside": "contained within",
            "outside": "external to",
            "surrounding": "surrounding"
        }
        
        for term, description in spatial_terms.items():
            if term in question_lower:
                response += f"The model contains components that are {description} other components.\n"
        
        return response
    
    def _generate_local_measurement_response(self, question, results):
        """Generate a response about measurements without using GPT"""
        # Extract the most relevant measurement data
        features = results[0]['features_3d']
        
        response = "Based on the 3D model's measurements:\n\n"
        
        # Add information about dimensions
        if 'bbox_dimensions' in features:
            dims = features['bbox_dimensions']
            response += f"The model's bounding box dimensions are approximately {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} units.\n"
            
            # Determine which dimension is being asked about
            question_lower = question.lower()
            if "height" in question_lower or "tall" in question_lower:
                response += f"The height (Y dimension) is approximately {dims[1]:.2f} units.\n"
            elif "width" in question_lower or "wide" in question_lower:
                response += f"The width (X dimension) is approximately {dims[0]:.2f} units.\n"
            elif "depth" in question_lower or "deep" in question_lower:
                response += f"The depth (Z dimension) is approximately {dims[2]:.2f} units.\n"
        
        # Add information about volume
        if 'volume' in features and features['volume'] > 0:
            response += f"The model has a volume of approximately {features['volume']:.2f} cubic units.\n"
        
        # Add information about surface area
        if 'surface_area' in features:
            response += f"The surface area is approximately {features['surface_area']:.2f} square units.\n"
        
        return response
    
    def _generate_local_response(self, question, results):
        """Generate a response locally using retrieved descriptions"""
        # Extract texts from results
        texts = [result["text"] for result in results]
        
        # Combine texts into a single context
        context = "\n\n".join(texts)
        
        # Generate a simple response based on the context
        response = f"Based on the 3D model analysis, I found the following information:\n\n"
        
        # Add the most relevant description
        response += texts[0]
        
        # Check if the question is about specific aspects
        aspects = {
            "color": ["color", "colours", "colored", "hue", "shade"],
            "size": ["size", "dimension", "width", "height", "length", "large", "small"],
            "structure": ["structure", "shape", "form", "design", "layout", "arrangement"],
            "material": ["material", "texture", "surface", "made of", "composed of"],
            "windows": ["window", "glass", "opening"],
            "doors": ["door", "entrance", "exit"],
            "walls": ["wall", "partition", "barrier"],
            "floors": ["floor", "ground", "base"],
            "ceilings": ["ceiling", "roof", "top"]
        }
        
        # Check if question is about a specific aspect
        for aspect, keywords in aspects.items():
            if any(keyword in question.lower() for keyword in keywords):
                # Find sentences in the context that mention this aspect
                sentences = re.split(r'[.!?]+', context)
                relevant_sentences = [s for s in sentences if any(keyword in s.lower() for keyword in keywords)]
                
                if relevant_sentences:
                    response += f"\n\nRegarding the {aspect}, I found:\n"
                    response += "\n".join(relevant_sentences)
        
        return response
    
    def _generate_gpt_response(self, question, results):
        """Generate an enhanced response using GPT API"""
        if not self.api_key:
            return self._generate_local_response(question, results)
        
        try:
            # Extract texts from results
            texts = [result["text"] for result in results]
            
            # Combine texts into a single context
            context = "\n\n".join(texts)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Use a more concise prompt to reduce token usage
            prompt = f"""
            Information about a 3D model:
            {context}
            
            Question: {question}
            
            Answer the question based only on the information provided. Be concise.
            """
            
            payload = {
                "model": "gpt-3.5-turbo",  # Use GPT-3.5 for cost efficiency
                "messages": [
                    {"role": "system", "content": "You analyze 3D architectural models based on provided descriptions."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 250,  # Limit response tokens for cost efficiency
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"GPT API error: {response.status_code}")
                print(response.text)
                return self._generate_local_response(question, results)
        except Exception as e:
            print(f"Error generating GPT response: {e}")
            return self._generate_local_response(question, results)
    
    def _generate_gpt_response_with_features(self, question, results):
        """Generate an enhanced response using GPT API with 3D features"""
        if not self.api_key:
            return self._generate_local_geometric_response(question, results)
        
        try:
            # Extract the most relevant features
            features = results[0]['features_3d']
            
            # Format features as a structured text
            feature_text = "3D Model Geometric Features:\n"
            
            if 'bbox_dimensions' in features:
                dims = features['bbox_dimensions']
                feature_text += f"- Dimensions: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} units\n"
            
            if 'volume' in features:
                feature_text += f"- Volume: {features['volume']:.2f} cubic units\n"
            
            if 'surface_area' in features:
                feature_text += f"- Surface Area: {features['surface_area']:.2f} square units\n"
            
            if 'is_watertight' in features:
                feature_text += f"- Watertight: {'Yes' if features['is_watertight'] else 'No'}\n"
            
            if 'mean_curvature' in features:
                feature_text += f"- Mean Curvature: {features['mean_curvature']:.4f}\n"
            
            if 'pca_eigenvalues' in features and len(features['pca_eigenvalues']) >= 3:
                eigenvalues = features['pca_eigenvalues']
                feature_text += f"- PCA Eigenvalues: [{eigenvalues[0]:.2f}, {eigenvalues[1]:.2f}, {eigenvalues[2]:.2f}]\n"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            prompt = f"""
            {feature_text}
            
            Question about the 3D model: {question}
            
            Answer the question based on the geometric features provided. Be specific and technical.
            """
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert in 3D geometry and computer graphics."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 250,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"GPT API error: {response.status_code}")
                print(response.text)
                return self._generate_local_geometric_response(question, results)
        except Exception as e:
            print(f"Error generating GPT response with features: {e}")
            return self._generate_local_geometric_response(question, results)
    
    def _generate_gpt_response_with_scene_graph(self, question, results):
        """Generate an enhanced response using GPT API with scene graph data"""
        if not self.api_key:
            return self._generate_local_spatial_response(question, results)
        
        try:
            # Extract the most relevant scene graph data
            metadata = results[0]['metadata']
            text = results[0]['text']
            
            # Format scene graph as structured text
            scene_graph_text = "3D Model Spatial Relationships:\n"
            scene_graph_text += text + "\n\n"
            
            if 'node_count' in metadata and 'edge_count' in metadata:
                scene_graph_text += f"- Components: {metadata['node_count']}\n"
                scene_graph_text += f"- Relationships: {metadata['edge_count']}\n"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            prompt = f"""
            {scene_graph_text}
            
            Question about the 3D model's spatial relationships: {question}
            
            Answer the question based on the spatial relationship information provided. Be specific about how components relate to each other.
            """
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert in 3D spatial relationships and architecture."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 250,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"GPT API error: {response.status_code}")
                print(response.text)
                return self._generate_local_spatial_response(question, results)
        except Exception as e:
            print(f"Error generating GPT response with scene graph: {e}")
            return self._generate_local_spatial_response(question, results)
    
    def _generate_gpt_response_with_measurements(self, question, results):
        """Generate an enhanced response using GPT API with measurement data"""
        if not self.api_key:
            return self._generate_local_measurement_response(question, results)
        
        try:
            # Extract the most relevant measurement data
            features = results[0]['features_3d']
            
            # Format measurements as structured text
            measurement_text = "3D Model Measurements:\n"
            
            if 'bbox_dimensions' in features:
                dims = features['bbox_dimensions']
                measurement_text += f"- Width (X): {dims[0]:.2f} units\n"
                measurement_text += f"- Height (Y): {dims[1]:.2f} units\n"
                measurement_text += f"- Depth (Z): {dims[2]:.2f} units\n"
            
            if 'volume' in features:
                measurement_text += f"- Volume: {features['volume']:.2f} cubic units\n"
            
            if 'surface_area' in features:
                measurement_text += f"- Surface Area: {features['surface_area']:.2f} square units\n"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            prompt = f"""
            {measurement_text}
            
            Question about the 3D model's measurements: {question}
            
            Answer the question based on the measurement data provided. Be precise with numbers and units.
            """
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert in 3D modeling and architectural measurements."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 250,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"GPT API error: {response.status_code}")
                print(response.text)
                return self._generate_local_measurement_response(question, results)
        except Exception as e:
            print(f"Error generating GPT response with measurements: {e}")
            return self._generate_local_measurement_response(question, results)

def query_3d_model(question, db_path="vector_db", use_gpt=False, max_gpt_calls=20):
    """
    Answer a question about the 3D model.
    
    Args:
        question: The question to answer
        db_path: Path to the vector database
        use_gpt: Whether to use GPT API for enhanced responses
        max_gpt_calls: Maximum number of GPT API calls to make
        
    Returns:
        Answer to the question
    """
    engine = ModelQueryEngine(db_path=db_path, use_gpt=use_gpt, max_gpt_calls=max_gpt_calls)
    return engine.query(question) 