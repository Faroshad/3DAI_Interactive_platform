import os
import json
import re
import requests
from vector_database import VectorDatabase

# Check if GPT API key is available
GPT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

class ModelQueryEngine:
    """
    Query engine for answering questions about 3D models using the vector database.
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
            print("GPT API will be used for enhanced responses (cost-effective mode).")
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