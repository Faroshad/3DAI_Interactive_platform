import os
import cv2
import numpy as np
from PIL import Image
import json
import time
import requests
import base64
from tqdm import tqdm
from io import BytesIO

# Check if GPT API key is available
GPT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Make pytesseract optional
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. OCR functionality will be limited.")
    print("You can install it with: pip install pytesseract")
    print("And install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")

class ImageAnalyzer:
    """
    Analyzes images of 3D models using OCR and vision AI to extract textual descriptions.
    """
    
    def __init__(self, use_gpt_vision=False, max_gpt_calls=10):
        """
        Initialize the image analyzer.
        
        Args:
            use_gpt_vision: Whether to use GPT Vision API for enhanced analysis
            max_gpt_calls: Maximum number of GPT API calls to make (for cost control)
        """
        # Set up OpenAI API key
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key and use_gpt_vision:
            print("Warning: OpenAI API key not found. Using basic image analysis only.")
            use_gpt_vision = False
        
        self.use_gpt_vision = use_gpt_vision
        self.max_gpt_calls = max_gpt_calls
        self.gpt_calls_made = 0
        
        # Check if OCR is available
        self.ocr_available = TESSERACT_AVAILABLE
        
        if self.use_gpt_vision:
            print("GPT Vision API will be used for enhanced image analysis.")
        else:
            print("Using basic image analysis only.")
            if not self.ocr_available:
                print("Note: OCR is not available. Text extraction will be limited.")
    
    def analyze_image(self, image_path):
        """
        Analyze a single image and extract textual description.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with analysis results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Failed to load image: {image_path}"}
        
        # Convert to RGB (OpenCV uses BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Basic image properties
        height, width, channels = image.shape
        
        results = {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "width": width,
            "height": height,
            "channels": channels,
            "descriptions": []
        }
        
        # If GPT Vision API is available and enabled, use it
        if self.use_gpt_vision:
            gpt_description = self._analyze_with_gpt_vision(image_path)
            if gpt_description:
                results["descriptions"].append({
                    "source": "gpt_vision",
                    "text": gpt_description
                })
        
        # Extract text using OCR
        if self.ocr_available:
            ocr_text = self._extract_text_with_ocr(image_rgb)
            if ocr_text:
                results["descriptions"].append({
                    "source": "ocr",
                    "text": ocr_text
                })
        
        # Analyze image content
        content_description = self._analyze_image_content(image_rgb)
        if content_description:
            results["descriptions"].append({
                "source": "content_analysis",
                "text": content_description
            })
        
        return results
    
    def analyze_image_batch(self, image_paths, output_dir=None):
        """
        Analyze a batch of images.
        
        Args:
            image_paths: List of paths to images
            output_dir: Directory to save analysis results
            
        Returns:
            List of analysis results
        """
        results = []
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Use tqdm for progress tracking
        for i, image_path in enumerate(tqdm(image_paths, desc="Analyzing images")):
            # Check if we've reached the maximum number of GPT calls
            if self.use_gpt_vision and self.gpt_calls_made >= self.max_gpt_calls:
                print(f"\nReached maximum GPT API calls ({self.max_gpt_calls}). Switching to OCR only.")
                self.use_gpt_vision = False
            
            result = self.analyze_image(image_path)
            results.append(result)
            
            # Save result to file
            if output_dir:
                base_name = os.path.basename(image_path)
                file_name = os.path.splitext(base_name)[0]
                output_file = os.path.join(output_dir, f"{file_name}_analysis.json")
                
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
        
        return results
    
    def _extract_text_with_ocr(self, image):
        """
        Extract text from image using OCR.
        
        Args:
            image: Image to extract text from
            
        Returns:
            Extracted text
        """
        if not self.ocr_available:
            return ""  # Return empty string instead of error message
        
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = image
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(image_pil)
            return text.strip()
        except Exception as e:
            # Only print the error once
            if not hasattr(self, '_ocr_error_shown'):
                print(f"Error extracting text with OCR: {e}")
                print("OCR will be disabled for this session.")
                self._ocr_error_shown = True
            return ""
    
    def _analyze_image_content(self, image):
        """Analyze image content and generate a description"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 100, 200)
            
            # Count edges as a measure of complexity
            edge_count = np.count_nonzero(edges)
            
            # Calculate average brightness
            brightness = np.mean(gray)
            
            # Generate a basic description
            description = []
            
            if edge_count > 10000:
                description.append("The image shows a complex structure with many details.")
            elif edge_count > 5000:
                description.append("The image shows a moderately complex structure.")
            else:
                description.append("The image shows a simple structure with few details.")
            
            if brightness > 200:
                description.append("The image is very bright.")
            elif brightness > 150:
                description.append("The image has normal brightness.")
            else:
                description.append("The image is relatively dark.")
            
            # Detect dominant colors
            pixels = np.float32(image.reshape(-1, 3))
            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            
            # Sort by count
            dominant_color = palette[np.argmax(counts)]
            
            # Name the color (very basic)
            color_names = {
                "red": [255, 0, 0],
                "green": [0, 255, 0],
                "blue": [0, 0, 255],
                "yellow": [255, 255, 0],
                "cyan": [0, 255, 255],
                "magenta": [255, 0, 255],
                "white": [255, 255, 255],
                "black": [0, 0, 0],
                "gray": [128, 128, 128]
            }
            
            # Find closest color
            min_dist = float('inf')
            closest_color = "unknown"
            
            for name, rgb in color_names.items():
                dist = np.sum((dominant_color - rgb) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_color = name
            
            description.append(f"The dominant color appears to be {closest_color}.")
            
            return " ".join(description)
        except Exception as e:
            print(f"Content analysis error: {e}")
            return ""
    
    def _analyze_with_gpt_vision(self, image_path):
        """Analyze image using GPT Vision API"""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return ""
        
        try:
            # Read image and encode as base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this 3D model view in detail. Focus on architectural elements, spatial relationships, materials, and any notable features. Provide a comprehensive description that could be used to understand the model's structure and design."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
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
                print(f"GPT Vision API error: {response.status_code}")
                print(response.text)
                return ""
        except Exception as e:
            print(f"GPT Vision analysis error: {e}")
            return ""

def analyze_model_images(image_paths, output_dir, use_gpt_vision=False):
    """
    Analyze images of a 3D model and extract textual descriptions.
    
    Args:
        image_paths: List of paths to image files
        output_dir: Directory to save analysis results
        use_gpt_vision: Whether to use GPT Vision API for image analysis
        
    Returns:
        Path to the analysis results file
    """
    # Create analyzer
    analyzer = ImageAnalyzer(use_gpt_vision=use_gpt_vision)
    
    # Analyze images
    results = analyzer.analyze_image_batch(image_paths, output_dir)
    
    # Return path to results file
    return os.path.join(output_dir, "image_analysis.json") 