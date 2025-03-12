# 3D Model Analysis System

An interactive platform for analyzing 3D architectural models using direct 3D geometry analysis, computer vision, and vector search technologies.

## Features

- **3D Geometric Analysis**: Extracts features directly from 3D geometry without relying on 2D images
- **Scene Graph Generation**: Creates a graph representation of spatial relationships between model components
- **Point Cloud Analysis**: Generates and analyzes point clouds for shape understanding
- **360° Image Capture**: Automatically captures images of 3D models from multiple viewpoints (optional)
- **Vector Database**: Stores 3D features and descriptions in a searchable format using embeddings
- **Specialized Query Engine**: Answers geometric, spatial, and measurement questions about 3D models
- **Interactive Mode**: Provides a conversational interface for model exploration
- **Advanced 3D Analysis**: Supports detailed geometric and spatial analysis of 3D models

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Faroshad/3DAI_Interactive_platform.git
cd 3DAI_Interactive_platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR (optional, only needed if using image-based OCR):
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

## Usage

### Basic Usage

```bash
# List available 3D models
python main.py --list_models

# Perform 3D geometric analysis (recommended approach)
python main.py --analyze_3d --model models/your_model.obj

# Build vector database from 3D analysis
python main.py --build_database

# Query the model
python main.py --query "What is the shape of the model?"

# Interactive mode
python main.py --interactive
```

### Legacy Image-Based Approach (Optional)

```bash
# Capture 360° images of a model
python main.py --capture_images --model models/your_model.obj

# Analyze captured images
python main.py --analyze_images

# Build vector database including image analysis
python main.py --build_database
```

### Advanced Options

- `--use_gpt`: Use GPT API for enhanced responses
- `--cost_effective`: Use cost-effective settings (fewer API calls)
- `--max_gpt_vision_calls`: Maximum GPT Vision API calls (default: 5)
- `--max_gpt_query_calls`: Maximum GPT query API calls (default: 10)
- `--install_help`: Show installation help for missing dependencies
- `--api_key`: OpenAI API key for GPT features (e.g., `--api_key "your-api-key-here"`)
- `--skip_images`: Skip image capture and analysis, use only 3D features
- `--analyze_3d`: Perform direct 3D geometric analysis (recommended)

## How It Works

1. **Model Loading**: The system loads 3D models from various formats (FBX, OBJ, etc.)
2. **3D Geometric Analysis**: 
   - Extracts point clouds, geometric features, and shape characteristics
   - Calculates measurements like volume, surface area, and dimensions
   - Analyzes curvature and shape distribution
3. **Scene Graph Generation**:
   - Identifies components and their spatial relationships
   - Creates a graph representation of the model's structure
   - Enables spatial reasoning queries (above, below, adjacent, etc.)
4. **Vector Database**: Stores 3D features and descriptions in a searchable format
5. **Specialized Query Engine**: 
   - Analyzes query type (geometric, spatial, measurement)
   - Uses appropriate 3D features to answer specific questions
   - Provides accurate responses based on actual 3D geometry

## Requirements

- Python 3.7+
- OpenAI API key (optional, for GPT features)
- 3D models in supported formats (OBJ, PLY, STL, OFF, FBX)
- GPU acceleration (optional, for faster processing)

## License

MIT License 