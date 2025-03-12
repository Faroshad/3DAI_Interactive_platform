import os
import argparse
import time
from pathlib import Path
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

from model_loader import load_3d_model, check_model_directory
from model_visualizer import ModelVisualizer
from image_analyzer import ImageAnalyzer
from vector_database import VectorDatabase
from query_engine import ModelQueryEngine

# Set the OpenAI API key if provided as an environment variable
# You should set this in your environment or use the --api_key parameter
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def main():
    """
    Main function to run the 3D model analysis system.
    """
    parser = argparse.ArgumentParser(description="3D Model Analysis System")
    parser.add_argument("--model", type=str, default="models/House.fbx", help="Path to the 3D model file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--capture_images", action="store_true", help="Capture 360° images of the model")
    parser.add_argument("--analyze_images", action="store_true", help="Analyze captured images using OCR and vision AI")
    parser.add_argument("--analyze_3d", action="store_true", help="Analyze 3D model using geometric features and scene graphs")
    parser.add_argument("--build_database", action="store_true", help="Build vector database from model analysis")
    parser.add_argument("--query", type=str, help="Query the 3D model database")
    parser.add_argument("--use_gpt", action="store_true", help="Use GPT API for enhanced responses")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive query mode")
    parser.add_argument("--max_gpt_vision_calls", type=int, default=5, help="Maximum GPT Vision API calls (for cost control)")
    parser.add_argument("--max_gpt_query_calls", type=int, default=10, help="Maximum GPT query API calls (for cost control)")
    parser.add_argument("--cost_effective", action="store_true", help="Use cost-effective settings (fewer API calls, smaller models)")
    parser.add_argument("--install_help", action="store_true", help="Show installation help for missing dependencies")
    parser.add_argument("--list_models", action="store_true", help="List available 3D models in the models directory")
    parser.add_argument("--api_key", type=str, help="OpenAI API key for GPT features")
    parser.add_argument("--skip_images", action="store_true", help="Skip image capture and analysis, use only 3D features")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Show installation help if requested
    if args.install_help:
        show_installation_help()
        return
    
    # List available models if requested
    if args.list_models:
        model_files = check_model_directory()
        return
    
    # If cost_effective flag is set, adjust settings
    if args.cost_effective:
        args.max_gpt_vision_calls = min(args.max_gpt_vision_calls, 5)
        args.max_gpt_query_calls = min(args.max_gpt_query_calls, 10)
        print("Using cost-effective settings to minimize API usage")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Analysis directory
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Vector database directory
    db_dir = output_dir / "vector_db"
    db_dir.mkdir(exist_ok=True)
    
    # Step 1: Load the 3D model
    mesh = None
    if args.capture_images or args.analyze_3d or (args.analyze_images and args.model != "models/House.fbx"):
        print(f"Loading 3D model: {args.model}")
        
        # Check if model file exists
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            print("Use --list_models to see available models or place your model in the models directory.")
            return
        
        mesh = load_3d_model(args.model)
        if mesh is None:
            print(f"Failed to load model: {args.model}")
            print("Please check the file format and try again, or use --list_models to see available models.")
            return
        print(f"Model loaded successfully")
    
    # Step 2: Capture 360° images (optional if using 3D analysis)
    if args.capture_images and not args.skip_images:
        if mesh is None:
            print("Error: Cannot capture images without a loaded model.")
            return
            
        print("Capturing 360° images of the model...")
        visualizer = ModelVisualizer(mesh, str(images_dir))
        
        # Use fewer viewpoints in cost-effective mode
        if args.cost_effective:
            h_views, v_views = 8, 4  # Fewer images = fewer API calls later
        else:
            h_views, v_views = 12, 6
            
        visualizer.capture_model_views(horizontal_views=h_views, vertical_views=v_views)
        print(f"Images captured and saved to {images_dir}")
    
    # Step 3: Analyze images (optional if using 3D analysis)
    if args.analyze_images and not args.skip_images:
        print("Analyzing captured images...")
        analyzer = ImageAnalyzer(
            use_gpt_vision=args.use_gpt,
            max_gpt_calls=args.max_gpt_vision_calls
        )
        
        # Get all image files from all subdirectories
        image_files = []
        for img_dir in images_dir.glob("*"):
            if img_dir.is_dir():
                image_files.extend([str(f) for f in img_dir.glob("*.png")])
        
        if not image_files:
            print(f"No images found in {images_dir} or its subdirectories")
            print("Please run with --capture_images first to generate images.")
            return
        
        print(f"Found {len(image_files)} images to analyze")
        results = analyzer.analyze_image_batch(image_files, str(analysis_dir))
        print(f"Image analysis complete. Results saved to {analysis_dir}")
    
    # New Step: 3D Geometric Analysis
    if args.analyze_3d:
        if mesh is None:
            print("Error: Cannot perform 3D analysis without a loaded model.")
            return
            
        print("Performing 3D geometric analysis...")
        visualizer = ModelVisualizer(mesh, str(output_dir))
        
        # Perform comprehensive 3D analysis
        analysis_results = visualizer.analyze_model(
            capture_views=not args.skip_images,  # Skip image capture if requested
            extract_features=True,
            generate_point_cloud=True,
            create_scene_graph=True,
            horizontal_views=8 if args.cost_effective else 12,
            vertical_views=4 if args.cost_effective else 6
        )
        
        print(f"3D analysis complete. Results saved to {output_dir}")
        
        # Print summary of analysis
        if 'geometric_features' in analysis_results:
            features = analysis_results['geometric_features']
            if 'bbox_dimensions' in features:
                dims = features['bbox_dimensions']
                print(f"Model dimensions: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} units")
            
            if 'volume' in features and features['volume'] > 0:
                print(f"Model volume: {features['volume']:.2f} cubic units")
            
            if 'surface_area' in features:
                print(f"Model surface area: {features['surface_area']:.2f} square units")
        
        if 'scene_graph' in analysis_results:
            sg = analysis_results['scene_graph']
            print(f"Scene graph generated with {sg['num_nodes']} nodes and {sg['num_edges']} relationships")
    
    # Step 4: Build vector database
    if args.build_database:
        print("Building vector database...")
        db = VectorDatabase(db_path=str(db_dir))
        
        # Get all analysis files (both image analysis and 3D analysis)
        analysis_files = []
        
        # Add image analysis files if they exist
        image_analysis_files = [str(f) for f in analysis_dir.glob("*.json")]
        if image_analysis_files:
            print(f"Found {len(image_analysis_files)} image analysis files")
            analysis_files.extend(image_analysis_files)
        
        # Add 3D feature files if they exist
        features_dir = output_dir / "features"
        if features_dir.exists():
            feature_files = [str(f) for f in features_dir.glob("*.json")]
            if feature_files:
                print(f"Found {len(feature_files)} 3D feature files")
                analysis_files.extend(feature_files)
        
        # Add scene graph files if they exist
        scene_graph_dir = output_dir / "scene_graphs"
        if scene_graph_dir.exists():
            scene_graph_files = [str(f) for f in scene_graph_dir.glob("*.json")]
            if scene_graph_files:
                print(f"Found {len(scene_graph_files)} scene graph files")
                analysis_files.extend(scene_graph_files)
        
        # Add model analysis files if they exist
        model_analysis_files = [str(f) for f in output_dir.glob("model_analysis_*.json")]
        if model_analysis_files:
            print(f"Found {len(model_analysis_files)} model analysis files")
            analysis_files.extend(model_analysis_files)
        
        if not analysis_files:
            print(f"No analysis files found")
            print("Please run with --analyze_images or --analyze_3d first to generate analysis files.")
            return
        
        print(f"Found a total of {len(analysis_files)} analysis files")
        for file in analysis_files:
            db.add_from_analysis_file(file)
        
        db.build_index()
        db.save()
        print(f"Vector database built and saved to {db_dir}")
    
    # Step 5: Query the database
    if args.query:
        # Check if vector database exists
        if not (db_dir / "index.faiss").exists() or not (db_dir / "data.pkl").exists():
            print("Vector database not found. Please run with --build_database first.")
            return
            
        print(f"Querying: {args.query}")
        engine = ModelQueryEngine(
            db_path=str(db_dir), 
            use_gpt=args.use_gpt,
            max_gpt_calls=args.max_gpt_query_calls
        )
        answer = engine.query(args.query)
        print("\nAnswer:")
        print(answer)
    
    # Interactive query mode
    if args.interactive:
        # Check if vector database exists
        if not (db_dir / "index.faiss").exists() or not (db_dir / "data.pkl").exists():
            print("Vector database not found. Please run with --build_database first.")
            return
            
        print("\nEntering interactive query mode. Type 'exit' to quit.")
        engine = ModelQueryEngine(
            db_path=str(db_dir), 
            use_gpt=args.use_gpt,
            max_gpt_calls=args.max_gpt_query_calls
        )
        
        while True:
            query = input("\nEnter your question about the 3D model: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            start_time = time.time()
            answer = engine.query(query)
            elapsed = time.time() - start_time
            
            print("\nAnswer:")
            print(answer)
            print(f"\n(Response generated in {elapsed:.2f} seconds)")

def show_installation_help():
    """Show installation help for missing dependencies"""
    print("\n" + "=" * 80)
    print("3D Model Analysis System - Installation Help")
    print("=" * 80)
    
    print("\nRequired Dependencies:")
    
    # Core dependencies
    print("\n1. Core Dependencies:")
    print("   pip install numpy open3d matplotlib pillow tqdm")
    
    # 3D model processing
    print("\n2. 3D Model Processing:")
    print("   pip install trimesh pyglet")
    
    # Image analysis and OCR (optional)
    print("\n3. Image Analysis and OCR (Optional):")
    print("   pip install opencv-python pytesseract requests")
    print("\n   Tesseract OCR Installation (Optional):")
    print("   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki")
    print("   - Linux: sudo apt-get install tesseract-ocr")
    print("   - macOS: brew install tesseract")
    
    # Vector database and embeddings
    print("\n4. Vector Database and Embeddings:")
    print("   pip install faiss-cpu sentence-transformers")
    
    # 3D Analysis (new)
    print("\n5. 3D Geometric Analysis:")
    print("   pip install torch networkx scipy")
    
    # Machine learning
    print("\n6. Machine Learning:")
    print("   pip install scikit-learn")
    
    # Additional dependencies
    print("\n7. Additional Dependencies:")
    print("   pip install protobuf==3.20.0 tensorflow regex")
    
    print("\nQuick Install (all dependencies):")
    print("   pip install -r requirements.txt")
    
    print("\nNote: If you encounter issues with specific dependencies, you can still run parts of the system.")
    print("For example, if pytesseract is not installed, you can still use GPT Vision API for image analysis.")
    print("Or you can skip image analysis entirely with --skip_images and use only 3D geometric analysis.")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as e:
        print(f"\nError: {e}")
        print("\nSome dependencies are missing. Run with --install_help for installation instructions:")
        print("python main.py --install_help")
    except Exception as e:
        print(f"\nError: {e}")
        print("Run with --install_help for troubleshooting information.") 