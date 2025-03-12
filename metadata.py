import json
import numpy as np
import os
from datetime import datetime
import hashlib

def extract_metadata(mesh, segmentation_data):
    """
    Extracts and stores comprehensive metadata about the 3D model and its semantic structure.
    Creates a rich knowledge base that can be queried by the AI system.
    """
    # Check if we're working with the enhanced mesh or the original mesh
    if hasattr(mesh, 'mesh'):
        # We're working with EnhancedMesh
        original_mesh = mesh.mesh
        model_info = mesh.model_info
    else:
        # We're working with the original mesh
        original_mesh = mesh
        # Get basic model info
        if hasattr(mesh, 'model_info'):
            model_info = mesh.model_info
        else:
            # Create basic info if not available
            vertices = np.asarray(original_mesh.vertices)
            triangles = np.asarray(original_mesh.triangles)
            
            # Calculate bounding box
            bbox = original_mesh.get_axis_aligned_bounding_box()
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()
            dimensions = max_bound - min_bound
            
            model_info = {
                "file_path": "unknown",
                "file_name": "unknown",
                "num_vertices": len(vertices),
                "num_triangles": len(triangles),
                "bounding_box": {
                    "min": min_bound.tolist(),
                    "max": max_bound.tolist(),
                    "dimensions": dimensions.tolist()
                }
            }
    
    # Create a unique model ID based on geometry
    model_hash = create_model_hash(original_mesh)
    
    # Combine all metadata
    metadata = {
        "model_id": model_hash,
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info,
        "segmentation": segmentation_data,
        
        # Architectural metrics
        "architectural_metrics": {
            "total_floor_area": segmentation_data.get("floor_area", 0),
            "total_wall_area": segmentation_data.get("wall_area", 0),
            "total_window_area": segmentation_data.get("window_area", 0),
            "total_door_area": segmentation_data.get("door_area", 0),
            "window_to_wall_ratio": calculate_ratio(
                segmentation_data.get("window_area", 0), 
                segmentation_data.get("wall_area", 0)
            ),
            "door_to_wall_ratio": calculate_ratio(
                segmentation_data.get("door_area", 0), 
                segmentation_data.get("wall_area", 0)
            ),
            "ceiling_height": calculate_ceiling_height(model_info, segmentation_data),
            "room_count": estimate_room_count(segmentation_data),
        },
        
        # Semantic descriptions
        "semantic_descriptions": generate_semantic_descriptions(model_info, segmentation_data),
        
        # Spatial relationships
        "spatial_relationships": extract_spatial_relationships(segmentation_data)
    }
    
    # Save metadata to file
    output_file = "model_metadata.json"
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Comprehensive metadata saved to {output_file}")
    
    # Create a vector database file for semantic querying
    create_vector_database(metadata)
    
    return metadata

def create_model_hash(mesh):
    """Creates a unique hash for the model based on its geometry"""
    vertices = np.asarray(mesh.vertices)
    if len(vertices) > 1000:
        # Use a subset of vertices for large models
        sample_vertices = vertices[::10]  # Take every 10th vertex
    else:
        sample_vertices = vertices
    
    # Create hash from vertex data
    vertex_bytes = sample_vertices.tobytes()
    return hashlib.md5(vertex_bytes).hexdigest()

def calculate_ratio(numerator, denominator):
    """Safely calculate ratio between two values"""
    if denominator == 0:
        return 0
    return numerator / denominator

def calculate_ceiling_height(model_info, segmentation_data):
    """Estimate average ceiling height"""
    try:
        # Get floor and ceiling centers
        floors = segmentation_data.get("segment_data", {}).get("floors", [])
        ceilings = segmentation_data.get("segment_data", {}).get("ceilings", [])
        
        if not floors or not ceilings:
            # Use bounding box if no floors/ceilings detected
            dimensions = model_info.get("bounding_box", {}).get("dimensions", [0, 0, 0])
            return dimensions[1] if len(dimensions) > 1 else 0
        
        # Calculate average heights
        floor_heights = [item["center"][1] for item in floors]
        ceiling_heights = [item["center"][1] for item in ceilings]
        
        if floor_heights and ceiling_heights:
            avg_floor_height = sum(floor_heights) / len(floor_heights)
            avg_ceiling_height = sum(ceiling_heights) / len(ceiling_heights)
            return avg_ceiling_height - avg_floor_height
    except Exception as e:
        print(f"Error calculating ceiling height: {e}")
    
    return 0

def estimate_room_count(segmentation_data):
    """Estimate number of rooms based on floor and ceiling segments"""
    try:
        floors = segmentation_data.get("segment_data", {}).get("floors", [])
        return max(1, len(floors))
    except Exception as e:
        print(f"Error estimating room count: {e}")
    
    return 1

def generate_semantic_descriptions(model_info, segmentation_data):
    """Generate natural language descriptions of the model"""
    descriptions = []
    
    # Basic model description
    dimensions = model_info.get("bounding_box", {}).get("dimensions", [0, 0, 0])
    if len(dimensions) >= 3:
        width, height, depth = dimensions
        descriptions.append(f"This is a 3D model with dimensions approximately {width:.2f} x {height:.2f} x {depth:.2f} units.")
    
    # Room description
    room_count = estimate_room_count(segmentation_data)
    if room_count == 1:
        descriptions.append("The model appears to represent a single room.")
    else:
        descriptions.append(f"The model appears to contain approximately {room_count} rooms or spaces.")
    
    # Architectural elements
    walls = segmentation_data.get("walls", 0)
    windows = segmentation_data.get("windows", 0)
    doors = segmentation_data.get("doors", 0)
    
    if walls > 0:
        descriptions.append(f"There are approximately {walls} wall segments in the model.")
    
    if windows > 0:
        descriptions.append(f"The model contains {windows} windows.")
    
    if doors > 0:
        descriptions.append(f"The model has {doors} doors.")
    
    # Floor area description
    floor_area = segmentation_data.get("floor_area", 0)
    if floor_area > 0:
        descriptions.append(f"The total floor area is approximately {floor_area:.2f} square units.")
    
    # Ceiling height
    ceiling_height = calculate_ceiling_height(model_info, segmentation_data)
    if ceiling_height > 0:
        descriptions.append(f"The ceiling height is approximately {ceiling_height:.2f} units.")
    
    return descriptions

def extract_spatial_relationships(segmentation_data):
    """Extract spatial relationships between architectural elements"""
    relationships = []
    
    segment_data = segmentation_data.get("segment_data", {})
    
    # Process windows and doors in relation to walls
    windows = segment_data.get("windows", [])
    doors = segment_data.get("doors", [])
    walls = segment_data.get("walls", [])
    
    # Window-wall relationships
    for window in windows:
        window_center = np.array(window.get("center", [0, 0, 0]))
        
        # Find closest wall
        closest_wall = None
        min_distance = float('inf')
        
        for wall in walls:
            wall_center = np.array(wall.get("center", [0, 0, 0]))
            distance = np.linalg.norm(window_center - wall_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_wall = wall
        
        if closest_wall:
            relationships.append({
                "type": "window_in_wall",
                "window_id": window.get("id", "unknown"),
                "wall_id": closest_wall.get("id", "unknown"),
                "distance": float(min_distance)
            })
    
    # Door-wall relationships
    for door in doors:
        door_center = np.array(door.get("center", [0, 0, 0]))
        
        # Find closest wall
        closest_wall = None
        min_distance = float('inf')
        
        for wall in walls:
            wall_center = np.array(wall.get("center", [0, 0, 0]))
            distance = np.linalg.norm(door_center - wall_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_wall = wall
        
        if closest_wall:
            relationships.append({
                "type": "door_in_wall",
                "door_id": door.get("id", "unknown"),
                "wall_id": closest_wall.get("id", "unknown"),
                "distance": float(min_distance)
            })
    
    return relationships

def create_vector_database(metadata):
    """Create a vector database file for semantic querying"""
    # Create a simplified version of metadata for vector database
    vector_db = {
        "model_id": metadata["model_id"],
        "descriptions": metadata["semantic_descriptions"],
        "metrics": metadata["architectural_metrics"],
        "elements": {
            "walls": metadata["segmentation"].get("walls", 0),
            "windows": metadata["segmentation"].get("windows", 0),
            "doors": metadata["segmentation"].get("doors", 0),
            "floors": metadata["segmentation"].get("floors", 0),
            "ceilings": metadata["segmentation"].get("ceilings", 0)
        },
        "relationships": metadata["spatial_relationships"]
    }
    
    # Save vector database
    with open("model_vector_db.json", "w") as f:
        json.dump(vector_db, f, indent=4)
    
    print("Vector database created for semantic querying")
