import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import math

def segment_objects(mesh):
    """
    Advanced semantic segmentation of architectural 3D models.
    Identifies and classifies architectural elements like walls, windows, doors, floors, ceilings, etc.
    Uses geometric analysis, orientation, and spatial relationships.
    """
    # Check if we're working with the enhanced mesh or the original mesh
    if hasattr(mesh, 'mesh'):
        # We're working with EnhancedMesh, get the underlying mesh
        original_mesh = mesh.mesh
    else:
        # We're working with the original mesh
        original_mesh = mesh
    
    if not original_mesh.has_triangles():
        print("Warning: Mesh has no triangles. Cannot segment objects.")
        return {"walls": 0, "windows": 0, "doors": 0, "floors": 0, "ceilings": 0, "furniture": 0}
    
    print("Starting advanced semantic segmentation...")
    
    # Ensure normals are computed
    original_mesh.compute_vertex_normals()
    original_mesh.compute_triangle_normals()
    
    # Get mesh data
    vertices = np.asarray(original_mesh.vertices)
    triangles = np.asarray(original_mesh.triangles)
    triangle_normals = np.asarray(original_mesh.triangle_normals)
    
    # Get model dimensions from model_info if available
    if hasattr(mesh, 'model_info'):
        dimensions = mesh.model_info["bounding_box"]["dimensions"]
        min_bound = mesh.model_info["bounding_box"]["min"]
        max_bound = mesh.model_info["bounding_box"]["max"]
    else:
        # Calculate if not available
        bbox = original_mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        dimensions = max_bound - min_bound
    
    # Define height thresholds for architectural elements
    floor_height_threshold = min_bound[1] + dimensions[1] * 0.1
    ceiling_height_threshold = max_bound[1] - dimensions[1] * 0.1
    
    # Initialize segmentation results
    segments = {
        "walls": [],
        "windows": [],
        "doors": [],
        "floors": [],
        "ceilings": [],
        "furniture": []
    }
    
    # Calculate triangle centers and areas
    triangle_centers = []
    triangle_areas = []
    
    for i, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        center = (v0 + v1 + v2) / 3
        triangle_centers.append(center)
        
        # Calculate triangle area
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        triangle_areas.append(area)
    
    triangle_centers = np.array(triangle_centers)
    triangle_areas = np.array(triangle_areas)
    
    # Segment by orientation
    for i, normal in enumerate(triangle_normals):
        try:
            center = triangle_centers[i]
            area = triangle_areas[i]
            
            # Skip very small triangles (likely noise)
            if area < 0.001:
                continue
                
            # Normalize normal vector
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
            else:
                continue
            
            # Classify based on normal orientation
            # Vertical surfaces (walls, doors, windows)
            if abs(normal[1]) < 0.2:  # Y-axis is up
                # Check if it's a potential window or door (holes in walls)
                # Windows and doors typically have specific dimensions and positions
                
                # Wall classification
                segments["walls"].append({
                    "triangle_idx": i,
                    "center": center.tolist(),
                    "normal": normal.tolist(),
                    "area": float(area)
                })
                
                # Check for windows (typically higher up and smaller)
                window_height_min = min_bound[1] + dimensions[1] * 0.4
                window_height_max = min_bound[1] + dimensions[1] * 0.8
                
                if (center[1] > window_height_min and 
                    center[1] < window_height_max and 
                    area < np.mean(triangle_areas) * 0.5):
                    segments["windows"].append({
                        "triangle_idx": i,
                        "center": center.tolist(),
                        "normal": normal.tolist(),
                        "area": float(area)
                    })
                
                # Check for doors (typically lower and larger than windows)
                door_height_max = min_bound[1] + dimensions[1] * 0.5
                if center[1] < door_height_max and area > np.mean(triangle_areas):
                    segments["doors"].append({
                        "triangle_idx": i,
                        "center": center.tolist(),
                        "normal": normal.tolist(),
                        "area": float(area)
                    })
                    
            # Horizontal surfaces facing up (floors, tables, etc.)
            elif normal[1] > 0.8:
                if center[1] < floor_height_threshold:
                    segments["floors"].append({
                        "triangle_idx": i,
                        "center": center.tolist(),
                        "normal": normal.tolist(),
                        "area": float(area)
                    })
                else:
                    # Horizontal surfaces above floor level are likely furniture
                    segments["furniture"].append({
                        "triangle_idx": i,
                        "center": center.tolist(),
                        "normal": normal.tolist(),
                        "area": float(area)
                    })
                    
            # Horizontal surfaces facing down (ceilings)
            elif normal[1] < -0.8:
                if center[1] > ceiling_height_threshold:
                    segments["ceilings"].append({
                        "triangle_idx": i,
                        "center": center.tolist(),
                        "normal": normal.tolist(),
                        "area": float(area)
                    })
                    
        except (IndexError, TypeError) as e:
            # Skip problematic triangles
            continue
    
    # Refine segmentation using clustering to group nearby triangles
    refined_segments = refine_segments_by_clustering(segments, triangle_centers)
    
    # Calculate statistics
    segment_stats = {
        "walls": len(refined_segments["walls"]),
        "windows": len(refined_segments["windows"]),
        "doors": len(refined_segments["doors"]),
        "floors": len(refined_segments["floors"]),
        "ceilings": len(refined_segments["ceilings"]),
        "furniture": len(refined_segments["furniture"])
    }
    
    # Calculate total areas
    segment_areas = {
        "wall_area": sum(item["area"] for item in segments["walls"]),
        "window_area": sum(item["area"] for item in segments["windows"]),
        "door_area": sum(item["area"] for item in segments["doors"]),
        "floor_area": sum(item["area"] for item in segments["floors"]),
        "ceiling_area": sum(item["area"] for item in segments["ceilings"])
    }
    
    # Combine stats and areas
    result = {**segment_stats, **segment_areas}
    
    # Add detailed segment data
    result["segment_data"] = refined_segments
    
    print(f"Segmentation complete:")
    print(f"  - Walls: {result['walls']}")
    print(f"  - Windows: {result['windows']}")
    print(f"  - Doors: {result['doors']}")
    print(f"  - Floors: {result['floors']}")
    print(f"  - Ceilings: {result['ceilings']}")
    print(f"  - Furniture: {result['furniture']}")
    
    return result

def refine_segments_by_clustering(segments, triangle_centers):
    """
    Refines segmentation by clustering nearby triangles to identify distinct objects.
    """
    refined_segments = {
        "walls": [],
        "windows": [],
        "doors": [],
        "floors": [],
        "ceilings": [],
        "furniture": []
    }
    
    # Process each segment type
    for segment_type, items in segments.items():
        if not items:
            continue
            
        # Extract triangle indices and centers
        indices = [item["triangle_idx"] for item in items]
        centers = np.array([triangle_centers[idx] for idx in indices])
        
        if len(centers) < 2:
            refined_segments[segment_type] = items
            continue
        
        try:
            # Cluster using DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(centers)
            labels = clustering.labels_
            
            # Group by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:  # Noise
                    continue
                    
                if label not in clusters:
                    clusters[label] = []
                    
                clusters[label].append(items[i])
            
            # Add each cluster as a separate object
            for label, cluster_items in clusters.items():
                # Calculate cluster properties
                avg_center = np.mean([np.array(item["center"]) for item in cluster_items], axis=0)
                total_area = sum(item["area"] for item in cluster_items)
                
                # Add as a single object
                refined_segments[segment_type].append({
                    "id": f"{segment_type}_{label}",
                    "center": avg_center.tolist(),
                    "area": total_area,
                    "triangle_count": len(cluster_items)
                })
                
        except Exception as e:
            print(f"Error clustering {segment_type}: {e}")
            refined_segments[segment_type] = items
    
    return refined_segments
