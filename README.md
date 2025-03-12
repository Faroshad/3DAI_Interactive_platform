# 3D Model Analysis System

An interactive platform for analyzing 3D architectural models using computer vision, natural language processing, and vector search technologies.

## Features

- **360° Image Capture**: Automatically captures images of 3D models from multiple viewpoints
- **Image Analysis**: Processes captured images using OCR and optionally GPT Vision API
- **Vector Database**: Stores descriptions in a searchable format using sentence embeddings
- **Query Engine**: Answers questions about 3D models based on the stored descriptions
- **Interactive Mode**: Provides a conversational interface for model exploration

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

3. Install Tesseract OCR (optional, for text extraction):
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

## Usage

### Basic Usage

```bash
# List available 3D models
python main.py --list_models

# Capture 360° images of a model
python main.py --capture_images --model models/your_model.obj

# Analyze captured images
python main.py --analyze_images

# Build vector database from image analysis
python main.py --build_database

# Query the model
python main.py --query "What is the shape of the model?"

# Interactive mode
python main.py --interactive
```

### Advanced Options

- `--use_gpt`: Use GPT API for enhanced responses
- `--cost_effective`: Use cost-effective settings (fewer API calls)
- `--max_gpt_vision_calls`: Maximum GPT Vision API calls (default: 5)
- `--max_gpt_query_calls`: Maximum GPT query API calls (default: 10)
- `--install_help`: Show installation help for missing dependencies
- `--api_key`: OpenAI API key for GPT features (e.g., `--api_key "your-api-key-here"`)

## How It Works

1. **Model Loading**: The system loads 3D models from various formats (FBX, OBJ, etc.)
2. **Image Capture**: The `ModelVisualizer` captures 360° images around the 3D model
3. **Image Analysis**: The `ImageAnalyzer` processes these images to generate textual descriptions
4. **Vector Database**: The `VectorDatabase` stores descriptions in a searchable format
5. **Query Engine**: The `ModelQueryEngine` answers questions about the 3D model

## Requirements

- Python 3.7+
- OpenAI API key (optional, for GPT features)
- Tesseract OCR (optional, for text extraction)
- 3D models in supported formats (OBJ, PLY, STL, OFF, FBX)

## License

MIT License 