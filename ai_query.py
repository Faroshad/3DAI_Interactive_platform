from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import faiss
import re

class ArchitecturalModelQueryEngine:
    """
    Advanced AI query engine for 3D architectural models.
    Provides semantic understanding and natural language interaction with 3D model data.
    """
    
    def __init__(self, vector_db_path="model_vector_db.json", metadata_path="model_metadata.json"):
        print("Initializing Architectural Model Query Engine...")
        
        # Load the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded sentence transformer model")
        
        # Load vector database and metadata
        self.vector_db = self._load_json(vector_db_path)
        self.metadata = self._load_json(metadata_path)
        
        # Initialize knowledge base
        self.knowledge_base = self._build_knowledge_base()
        
        # Build vector index
        self.index, self.texts = self._build_vector_index()
        
        print("Query engine initialized and ready")
    
    def _load_json(self, file_path):
        """Load JSON data from file with error handling"""
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Using empty data.")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def _build_knowledge_base(self):
        """Build a comprehensive knowledge base from metadata"""
        knowledge_base = []
        
        # Add semantic descriptions
        if self.vector_db and "descriptions" in self.vector_db:
            for desc in self.vector_db["descriptions"]:
                knowledge_base.append({
                    "text": desc,
                    "type": "description"
                })
        
        # Add architectural metrics
        if self.vector_db and "metrics" in self.vector_db:
            metrics = self.vector_db["metrics"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    knowledge_base.append({
                        "text": f"The {key.replace('_', ' ')} is {value}.",
                        "type": "metric",
                        "metric": key,
                        "value": value
                    })
        
        # Add element counts
        if self.vector_db and "elements" in self.vector_db:
            elements = self.vector_db["elements"]
            for element, count in elements.items():
                knowledge_base.append({
                    "text": f"There are {count} {element} in the model.",
                    "type": "element_count",
                    "element": element,
                    "count": count
                })
        
        # Add spatial relationships
        if self.vector_db and "relationships" in self.vector_db:
            for rel in self.vector_db["relationships"]:
                if rel["type"] == "window_in_wall":
                    knowledge_base.append({
                        "text": f"There is a window (ID: {rel['window_id']}) in wall (ID: {rel['wall_id']}).",
                        "type": "relationship",
                        "relationship": "window_in_wall"
                    })
                elif rel["type"] == "door_in_wall":
                    knowledge_base.append({
                        "text": f"There is a door (ID: {rel['door_id']}) in wall (ID: {rel['wall_id']}).",
                        "type": "relationship",
                        "relationship": "door_in_wall"
                    })
        
        # Add architectural analysis
        if self.metadata and "architectural_metrics" in self.metadata:
            metrics = self.metadata["architectural_metrics"]
            
            # Window-to-wall ratio analysis
            wwr = metrics.get("window_to_wall_ratio", 0)
            if wwr > 0:
                if wwr > 0.5:
                    knowledge_base.append({
                        "text": f"The window-to-wall ratio is {wwr:.2f}, which is very high and may lead to excessive solar gain.",
                        "type": "analysis",
                        "analysis_type": "window_to_wall_ratio"
                    })
                elif wwr > 0.3:
                    knowledge_base.append({
                        "text": f"The window-to-wall ratio is {wwr:.2f}, which is moderate and balanced.",
                        "type": "analysis",
                        "analysis_type": "window_to_wall_ratio"
                    })
                else:
                    knowledge_base.append({
                        "text": f"The window-to-wall ratio is {wwr:.2f}, which is relatively low and may limit natural light.",
                        "type": "analysis",
                        "analysis_type": "window_to_wall_ratio"
                    })
            
            # Room count analysis
            room_count = metrics.get("room_count", 0)
            if room_count > 0:
                knowledge_base.append({
                    "text": f"The model contains approximately {room_count} rooms or distinct spaces.",
                    "type": "analysis",
                    "analysis_type": "room_count"
                })
            
            # Ceiling height analysis
            ceiling_height = metrics.get("ceiling_height", 0)
            if ceiling_height > 0:
                if ceiling_height > 3.5:
                    knowledge_base.append({
                        "text": f"The ceiling height is {ceiling_height:.2f} units, which is quite high and creates a spacious feeling.",
                        "type": "analysis",
                        "analysis_type": "ceiling_height"
                    })
                elif ceiling_height > 2.5:
                    knowledge_base.append({
                        "text": f"The ceiling height is {ceiling_height:.2f} units, which is standard for residential spaces.",
                        "type": "analysis",
                        "analysis_type": "ceiling_height"
                    })
                else:
                    knowledge_base.append({
                        "text": f"The ceiling height is {ceiling_height:.2f} units, which is relatively low.",
                        "type": "analysis",
                        "analysis_type": "ceiling_height"
                    })
        
        return knowledge_base
    
    def _build_vector_index(self):
        """Build a vector index for semantic search"""
        texts = [item["text"] for item in self.knowledge_base]
        
        if not texts:
            # Add default texts if knowledge base is empty
            texts = [
                "This is a 3D architectural model.",
                "The model contains geometric data that can be analyzed.",
                "You can ask questions about the model's structure and features."
            ]
        
        # Encode texts to vectors
        vectors = self.model.encode(texts)
        
        # Create FAISS index
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors.astype('float32'))
        
        return index, texts
    
    def query(self, question):
        """
        Process a natural language query about the 3D model
        and return a relevant, insightful response.
        """
        if not question or not isinstance(question, str):
            return "Please ask a valid question about the 3D model."
        
        # Check for direct metric queries first
        direct_answer = self._check_direct_queries(question)
        if direct_answer:
            return direct_answer
        
        # Encode the query
        query_vector = self.model.encode([question]).astype('float32')
        
        # Search the vector index
        k = min(5, len(self.texts))  # Get top k results
        if k == 0:
            return "I don't have enough information about this 3D model yet."
            
        D, I = self.index.search(query_vector, k)
        
        # Get the most relevant texts
        relevant_items = [self.knowledge_base[idx] for idx in I[0]]
        
        # Generate a response based on the relevant items
        response = self._generate_response(question, relevant_items)
        
        return response
    
    def _check_direct_queries(self, question):
        """Check for direct queries about specific metrics or counts"""
        question_lower = question.lower()
        
        # Check for count questions
        count_patterns = {
            r"how many (windows|window)": ("windows", "element_count"),
            r"how many (walls|wall)": ("walls", "element_count"),
            r"how many (doors|door)": ("doors", "element_count"),
            r"how many (floors|floor)": ("floors", "element_count"),
            r"how many (ceilings|ceiling)": ("ceilings", "element_count"),
            r"how many (rooms|room|spaces|space)": ("room_count", "metric"),
            r"how many (vertices|vertex)": ("num_vertices", "model_info"),
            r"how many (triangles|triangle|faces|face)": ("num_triangles", "model_info"),
        }
        
        for pattern, (key, data_type) in count_patterns.items():
            if re.search(pattern, question_lower):
                if data_type == "element_count" and self.vector_db and "elements" in self.vector_db:
                    count = self.vector_db["elements"].get(key, 0)
                    return f"There are {count} {key} in the model."
                elif data_type == "metric" and self.metadata and "architectural_metrics" in self.metadata:
                    value = self.metadata["architectural_metrics"].get(key, 0)
                    return f"There are approximately {value} rooms/spaces in the model."
                elif data_type == "model_info" and self.metadata and "model_info" in self.metadata:
                    value = self.metadata["model_info"].get(key, 0)
                    return f"The model has {value} {key}."
        
        # Check for dimension/measurement questions
        dimension_patterns = {
            r"(what is|what's) the (ceiling height|height of the ceiling)": "ceiling_height",
            r"how (tall|high) (is|are) the ceiling": "ceiling_height",
            r"(what is|what's) the (floor area|area of the floor)": "total_floor_area",
            r"(what is|what's) the (wall area|area of the walls)": "total_wall_area",
            r"(what is|what's) the (window area|area of the windows)": "total_window_area",
            r"(what is|what's) the (window.wall ratio|window to wall ratio)": "window_to_wall_ratio",
        }
        
        for pattern, key in dimension_patterns.items():
            if re.search(pattern, question_lower) and self.metadata and "architectural_metrics" in self.metadata:
                value = self.metadata["architectural_metrics"].get(key, 0)
                if key == "ceiling_height":
                    return f"The ceiling height is approximately {value:.2f} units."
                elif key == "total_floor_area":
                    return f"The total floor area is approximately {value:.2f} square units."
                elif key == "total_wall_area":
                    return f"The total wall area is approximately {value:.2f} square units."
                elif key == "total_window_area":
                    return f"The total window area is approximately {value:.2f} square units."
                elif key == "window_to_wall_ratio":
                    return f"The window-to-wall ratio is {value:.2f}."
        
        return None
    
    def _generate_response(self, question, relevant_items):
        """Generate a comprehensive response based on relevant knowledge items"""
        # Group items by type
        descriptions = [item for item in relevant_items if item["type"] == "description"]
        metrics = [item for item in relevant_items if item["type"] == "metric"]
        counts = [item for item in relevant_items if item["type"] == "element_count"]
        analyses = [item for item in relevant_items if item["type"] == "analysis"]
        relationships = [item for item in relevant_items if item["type"] == "relationship"]
        
        # Start with a general response
        response_parts = []
        
        # Add descriptions
        if descriptions:
            response_parts.append(descriptions[0]["text"])
        
        # Add counts if relevant
        if "how many" in question.lower() and counts:
            for count_item in counts:
                response_parts.append(count_item["text"])
        
        # Add metrics if relevant
        metrics_keywords = ["size", "area", "height", "dimension", "ratio", "volume"]
        if any(keyword in question.lower() for keyword in metrics_keywords) and metrics:
            for metric_item in metrics:
                response_parts.append(metric_item["text"])
        
        # Add analysis if available
        if analyses:
            response_parts.append(analyses[0]["text"])
        
        # Add relationships if relevant
        relationship_keywords = ["connection", "connected", "between", "relation", "window", "door", "wall"]
        if any(keyword in question.lower() for keyword in relationship_keywords) and relationships:
            for rel_item in relationships:
                response_parts.append(rel_item["text"])
        
        # If no specific information was added, use general knowledge
        if not response_parts and relevant_items:
            for item in relevant_items:
                response_parts.append(item["text"])
        
        # If still no response, give a default answer
        if not response_parts:
            return "I don't have specific information about that aspect of the 3D model."
        
        # Combine response parts into a coherent answer
        response = " ".join(response_parts)
        
        return response

# Initialize the query engine
query_engine = None

def initialize_query_engine():
    """Initialize the query engine if not already initialized"""
    global query_engine
    if query_engine is None:
        try:
            query_engine = ArchitecturalModelQueryEngine()
            return True
        except Exception as e:
            print(f"Error initializing query engine: {e}")
            return False
    return True

def query_3d_model(question):
    """
    Process a natural language query about the 3D model
    and return a relevant, insightful response.
    """
    if not initialize_query_engine():
        return "Sorry, I couldn't initialize the AI query system. Please check the logs for details."
    
    try:
        return query_engine.query(question)
    except Exception as e:
        print(f"Error processing query: {e}")
        return f"Sorry, I encountered an error while processing your question: {str(e)}"

# Example test query
if __name__ == "__main__":
    test_questions = [
        "How many windows are in the model?",
        "What is the ceiling height?",
        "Describe the overall structure of the model.",
        "What is the window to wall ratio?",
        "How many rooms does the model have?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        print(f"A: {query_3d_model(question)}") 