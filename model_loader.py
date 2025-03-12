import open3d as o3d
import numpy as np
import os
import sys
import traceback
from pathlib import Path
from collections import defaultdict

class EnhancedMesh:
    """
    A wrapper class for Open3D TriangleMesh that allows attaching additional information.
    This class delegates attribute access to the underlying mesh object.
    """
    
    def __init__(self, mesh, model_info=None):
        """
        Initialize the enhanced mesh with an Open3D mesh and optional model info.
        
        Args:
            mesh: The Open3D TriangleMesh object
            model_info: Dictionary containing additional information about the model
        """
        self.mesh = mesh
        self.model_info = model_info or {}
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying mesh"""
        return getattr(self.mesh, name)

def load_3d_model(file_path):
    """
    Load a 3D model and perform detailed analysis.
    
    Args:
        file_path: Path to the 3D model file
        
    Returns:
        EnhancedMesh object containing the mesh and model information, or None if loading fails
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return None
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        print(f"Loading 3D model from {file_path} (format: {file_ext})")
        
        # Load the mesh based on file extension
        mesh = None
        
        try:
            if file_ext == '.obj':
                mesh = o3d.io.read_triangle_mesh(file_path)
            elif file_ext == '.ply':
                mesh = o3d.io.read_triangle_mesh(file_path)
            elif file_ext == '.stl':
                mesh = o3d.io.read_triangle_mesh(file_path)
            elif file_ext == '.off':
                mesh = o3d.io.read_triangle_mesh(file_path)
            elif file_ext == '.fbx':
                # For FBX files, try to use trimesh as a fallback
                try:
                    import trimesh
                    try:
                        tri_mesh = trimesh.load(file_path)
                        # Convert trimesh to Open3D mesh
                        vertices = np.array(tri_mesh.vertices)
                        faces = np.array(tri_mesh.faces)
                        
                        mesh = o3d.geometry.TriangleMesh()
                        mesh.vertices = o3d.utility.Vector3dVector(vertices)
                        mesh.triangles = o3d.utility.Vector3iVector(faces)
                        
                        print("Loaded FBX file using trimesh")
                    except Exception as e:
                        print(f"Error loading FBX file with trimesh: {e}")
                        print("\nFBX files are not fully supported. Please consider converting your model to OBJ format:")
                        print("1. Use Blender, MeshLab, or other 3D software to convert FBX to OBJ")
                        print("2. Place the converted OBJ file in the models directory")
                        print("3. Run the program with --model models/YourModel.obj")
                        return None
                except ImportError:
                    print("Warning: trimesh not installed. Cannot load FBX files.")
                    print("Install with: pip install trimesh")
                    print("\nAlternatively, convert your FBX model to OBJ format:")
                    print("1. Use Blender, MeshLab, or other 3D software to convert FBX to OBJ")
                    print("2. Place the converted OBJ file in the models directory")
                    print("3. Run the program with --model models/YourModel.obj")
                    return None
            else:
                print(f"Unsupported file format: {file_ext}")
                print("Supported formats: .obj, .ply, .stl, .off, .fbx")
                return None
        except Exception as e:
            print(f"Error loading 3D model: {e}")
            print("Detailed error information:")
            traceback.print_exc()
            return None
        
        # Validate the mesh
        if mesh is None:
            print("Failed to load the mesh.")
            return None
        
        if not mesh.has_triangles():
            print("Warning: The loaded mesh has no triangles.")
            if mesh.has_vertices():
                print(f"The mesh has {len(mesh.vertices)} vertices but no triangles.")
            else:
                print("The mesh has no vertices.")
            
            # Try to create triangles if possible
            if mesh.has_vertices() and len(mesh.vertices) >= 3:
                print("Attempting to create triangles from vertices...")
                try:
                    # For point clouds, try to reconstruct a mesh
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = mesh.vertices
                    
                    # Estimate normals if needed
                    pcd.estimate_normals()
                    
                    # Create a mesh using Poisson surface reconstruction
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)[0]
                    
                    if mesh.has_triangles():
                        print(f"Successfully created a mesh with {len(mesh.triangles)} triangles.")
                    else:
                        print("Failed to create triangles from vertices.")
                        return None
                except Exception as e:
                    print(f"Error creating triangles: {e}")
                    return None
            else:
                return None
        
        # Compute properties
        print(f"Model loaded successfully with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
        
        # Compute normals if they don't exist
        if not mesh.has_vertex_normals():
            print("Computing vertex normals...")
            mesh.compute_vertex_normals()
        
        if not mesh.has_triangle_normals():
            print("Computing triangle normals...")
            mesh.compute_triangle_normals()
        
        # Extract model information
        model_info = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size_bytes": os.path.getsize(file_path),
            "vertex_count": len(mesh.vertices),
            "triangle_count": len(mesh.triangles),
            "has_vertex_normals": mesh.has_vertex_normals(),
            "has_triangle_normals": mesh.has_triangle_normals(),
            "has_vertex_colors": mesh.has_vertex_colors(),
            "has_triangle_uvs": mesh.has_triangle_uvs(),
        }
        
        # Compute bounding box
        bbox = mesh.get_axis_aligned_bounding_box()
        model_info["bounding_box_min"] = bbox.min_bound.tolist()
        model_info["bounding_box_max"] = bbox.max_bound.tolist()
        model_info["bounding_box_size"] = (bbox.max_bound - bbox.min_bound).tolist()
        
        # Compute center of mass
        model_info["center_of_mass"] = mesh.get_center().tolist()
        
        # Compute surface area
        model_info["surface_area"] = mesh.get_surface_area()
        
        # Compute volume if the mesh is watertight
        if mesh.is_watertight():
            model_info["volume"] = mesh.get_volume()
            model_info["is_watertight"] = True
        else:
            model_info["is_watertight"] = False
        
        # Create enhanced mesh
        enhanced_mesh = EnhancedMesh(mesh, model_info)
        
        return enhanced_mesh
    
    except Exception as e:
        print(f"Unexpected error loading 3D model: {e}")
        print("Detailed error information:")
        traceback.print_exc()
        return None

def check_model_directory():
    """
    Check if the models directory exists and contains any 3D model files.
    Creates the directory if it doesn't exist.
    
    Returns:
        List of model files found
    """
    models_dir = Path("models")
    
    # Create directory if it doesn't exist
    if not models_dir.exists():
        print(f"Creating models directory: {models_dir}")
        models_dir.mkdir(parents=True)
        print("Please place your 3D models in this directory.")
        return []
    
    # Check for model files
    model_extensions = ['.obj', '.ply', '.stl', '.off', '.fbx']
    model_files = []
    
    for ext in model_extensions:
        model_files.extend(list(models_dir.glob(f"*{ext}")))
    
    if not model_files:
        print("No 3D model files found in the models directory.")
        print(f"Please add 3D models with extensions: {', '.join(model_extensions)}")
    else:
        print(f"Found {len(model_files)} 3D model files:")
        for model_file in model_files:
            print(f"  - {model_file.name}")
    
    return model_files
