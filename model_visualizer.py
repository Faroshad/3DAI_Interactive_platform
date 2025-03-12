import open3d as o3d
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from datetime import datetime
import trimesh
import torch
import json
from pathlib import Path

class ModelVisualizer:
    """
    Analyzes 3D models using both visual and geometric approaches.
    Extracts 3D features, generates point clouds, and captures views for comprehensive analysis.
    """
    
    def __init__(self, mesh, output_dir="captured_views"):
        """Initialize the visualizer with a mesh and output directory"""
        # Store the mesh (handle both EnhancedMesh and regular mesh)
        if hasattr(mesh, 'mesh'):
            self.mesh = mesh.mesh
            self.model_info = mesh.model_info
            self.original_mesh = mesh  # Keep reference to original mesh
        else:
            self.mesh = mesh
            self.model_info = getattr(mesh, 'model_info', {})
            self.original_mesh = mesh
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for different data types
        self.views_dir = os.path.join(self.output_dir, "views")
        self.features_dir = os.path.join(self.output_dir, "features")
        self.point_cloud_dir = os.path.join(self.output_dir, "point_clouds")
        self.scene_graph_dir = os.path.join(self.output_dir, "scene_graphs")
        
        os.makedirs(self.views_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.point_cloud_dir, exist_ok=True)
        os.makedirs(self.scene_graph_dir, exist_ok=True)
        
        # Set up visualization parameters
        self.width = 1024
        self.height = 768
        self.background_color = [1, 1, 1]  # White background
        
        # Calculate model center and scale
        self.center = self.mesh.get_center()
        bbox = self.mesh.get_axis_aligned_bounding_box()
        self.scale = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        self.bbox = bbox
        
        # Initialize the visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        self.vis.add_geometry(self.mesh)
        
        # Set default rendering options
        self._set_default_rendering_options()
        
        # Convert to trimesh for additional analysis
        try:
            self.trimesh_mesh = self._convert_to_trimesh()
        except Exception as e:
            print(f"Warning: Could not convert to trimesh: {e}")
            self.trimesh_mesh = None
            
        print(f"3D Model analyzer initialized. Output directory: {self.output_dir}")
    
    def _convert_to_trimesh(self):
        """Convert Open3D mesh to trimesh for additional analysis capabilities"""
        # Extract vertices and triangles from Open3D mesh
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        
        # Create trimesh mesh
        return trimesh.Trimesh(vertices=vertices, faces=triangles)
    
    def _set_default_rendering_options(self):
        """Set default rendering options for the visualizer"""
        render_option = self.vis.get_render_option()
        render_option.background_color = self.background_color
        render_option.point_size = 5.0
        render_option.show_coordinate_frame = False
        render_option.light_on = True
        render_option.mesh_show_wireframe = False
        render_option.mesh_show_back_face = True
        
    def extract_point_cloud(self, num_points=10000, save=True):
        """
        Extract a point cloud from the mesh with uniform sampling
        
        Args:
            num_points: Number of points to sample
            save: Whether to save the point cloud to disk
            
        Returns:
            Point cloud object
        """
        print(f"Extracting point cloud with {num_points} points...")
        
        # Sample points uniformly from the mesh
        pcd = self.mesh.sample_points_uniformly(number_of_points=num_points)
        
        # Compute normals for the point cloud
        pcd.estimate_normals()
        pcd.normalize_normals()
        
        if save:
            # Save the point cloud
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.point_cloud_dir, f"point_cloud_{timestamp}.ply")
            o3d.io.write_point_cloud(filename, pcd)
            print(f"Point cloud saved to {filename}")
            
            # Also save as numpy array for easier processing
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            np_filename = os.path.join(self.point_cloud_dir, f"point_cloud_{timestamp}.npz")
            np.savez(np_filename, points=points, normals=normals)
            
        return pcd
    
    def extract_3d_features(self, point_cloud=None, feature_dim=128, save=True):
        """
        Extract 3D features from the model using geometric properties
        
        Args:
            point_cloud: Optional point cloud to use (will generate if None)
            feature_dim: Dimension of the feature vector
            save: Whether to save features to disk
            
        Returns:
            Dictionary of features
        """
        print("Extracting 3D geometric features...")
        
        if point_cloud is None:
            point_cloud = self.extract_point_cloud(save=save)
        
        # Get points and normals
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)
        
        # Calculate basic geometric features
        features = {}
        
        # 1. Bounding box properties
        bbox = point_cloud.get_axis_aligned_bounding_box()
        features['bbox_min'] = bbox.get_min_bound().tolist()
        features['bbox_max'] = bbox.get_max_bound().tolist()
        features['bbox_dimensions'] = (bbox.get_max_bound() - bbox.get_min_bound()).tolist()
        features['bbox_volume'] = np.prod(features['bbox_dimensions'])
        
        # 2. Point distribution statistics
        features['centroid'] = point_cloud.get_center().tolist()
        
        # 3. Surface area and volume (if trimesh is available)
        if self.trimesh_mesh is not None:
            features['surface_area'] = float(self.trimesh_mesh.area)
            if self.trimesh_mesh.is_watertight:
                features['volume'] = float(self.trimesh_mesh.volume)
                features['is_watertight'] = True
            else:
                features['volume'] = 0.0
                features['is_watertight'] = False
        
        # 4. Normal distribution
        normal_angles = np.arccos(np.clip(normals @ np.array([0, 1, 0]), -1.0, 1.0))
        features['normal_angle_mean'] = float(np.mean(normal_angles))
        features['normal_angle_std'] = float(np.std(normal_angles))
        
        # 5. Curvature estimation (approximate using PCA on local neighborhoods)
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
        curvatures = []
        
        # Sample points for curvature estimation (for efficiency)
        sample_indices = np.random.choice(len(points), min(1000, len(points)), replace=False)
        
        for idx in sample_indices:
            # Find nearest neighbors
            [_, indices, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[idx], 30)
            
            # Compute PCA on the neighborhood
            local_points = np.asarray(point_cloud.points)[indices]
            local_points = local_points - np.mean(local_points, axis=0)
            
            # Compute covariance matrix and its eigenvalues
            cov = local_points.T @ local_points
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)
            
            # Estimate curvature as ratio of eigenvalues
            if eigenvalues[2] != 0:
                curvature = eigenvalues[0] / eigenvalues[2]
                curvatures.append(curvature)
        
        if curvatures:
            features['mean_curvature'] = float(np.mean(curvatures))
            features['max_curvature'] = float(np.max(curvatures))
        
        # 6. Generate a compact feature vector using PCA on point coordinates
        # This creates a simplified representation of the shape
        pca_points = points - np.mean(points, axis=0)
        cov = pca_points.T @ pca_points
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project points onto principal components
        projected = pca_points @ eigenvectors
        
        # Create histogram features from the projections
        hist_features = []
        for dim in range(3):  # Use the three principal dimensions
            hist, _ = np.histogram(projected[:, dim], bins=min(32, len(points)//10), density=True)
            hist_features.extend(hist)
        
        features['pca_eigenvalues'] = eigenvalues.tolist()
        features['histogram_features'] = hist_features
        
        if save:
            # Save features to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.features_dir, f"geometric_features_{timestamp}.json")
            
            with open(filename, 'w') as f:
                json.dump(features, f, indent=2)
            
            print(f"Geometric features saved to {filename}")
        
        return features
    
    def generate_scene_graph(self, save=True):
        """
        Generate a scene graph representation of the 3D model
        
        Args:
            save: Whether to save the scene graph to disk
            
        Returns:
            Dictionary containing the scene graph
        """
        print("Generating scene graph...")
        
        scene_graph = {
            'nodes': [],
            'edges': []
        }
        
        # If we have a trimesh object, we can use it for scene graph generation
        if self.trimesh_mesh is not None:
            # 1. Add the main model as a node
            model_node = {
                'id': 'model_0',
                'type': 'model',
                'bbox': {
                    'min': self.bbox.get_min_bound().tolist(),
                    'max': self.bbox.get_max_bound().tolist()
                },
                'center': self.center.tolist()
            }
            scene_graph['nodes'].append(model_node)
            
            # 2. Try to identify components using connected components
            try:
                # Get connected components
                components = self.trimesh_mesh.split(only_watertight=False)
                
                # Add each component as a node
                for i, component in enumerate(components):
                    # Calculate component properties
                    component_center = component.centroid
                    component_bbox = component.bounds
                    
                    # Add component node
                    component_node = {
                        'id': f'component_{i}',
                        'type': 'component',
                        'bbox': {
                            'min': component_bbox[0].tolist(),
                            'max': component_bbox[1].tolist()
                        },
                        'center': component_center.tolist(),
                        'volume': float(component.volume) if component.is_watertight else 0.0,
                        'surface_area': float(component.area)
                    }
                    scene_graph['nodes'].append(component_node)
                    
                    # Add edge connecting component to model
                    edge = {
                        'source': 'model_0',
                        'target': f'component_{i}',
                        'type': 'has_component'
                    }
                    scene_graph['edges'].append(edge)
                    
                # 3. Add spatial relationships between components
                for i in range(len(components)):
                    for j in range(i+1, len(components)):
                        # Calculate distance between component centers
                        center_i = np.array(scene_graph['nodes'][i+1]['center'])
                        center_j = np.array(scene_graph['nodes'][j+1]['center'])
                        distance = np.linalg.norm(center_i - center_j)
                        
                        # Add edge for nearby components
                        if distance < self.scale * 0.5:  # Threshold based on model scale
                            edge = {
                                'source': f'component_{i}',
                                'target': f'component_{j}',
                                'type': 'near',
                                'distance': float(distance)
                            }
                            scene_graph['edges'].append(edge)
                            
                            # Determine if one component is above the other
                            if center_i[1] > center_j[1] + self.scale * 0.1:
                                edge = {
                                    'source': f'component_{i}',
                                    'target': f'component_{j}',
                                    'type': 'above'
                                }
                                scene_graph['edges'].append(edge)
                            elif center_j[1] > center_i[1] + self.scale * 0.1:
                                edge = {
                                    'source': f'component_{j}',
                                    'target': f'component_{i}',
                                    'type': 'above'
                                }
                                scene_graph['edges'].append(edge)
            
            except Exception as e:
                print(f"Warning: Could not generate component-based scene graph: {e}")
        
        # If trimesh failed or isn't available, fall back to a simpler approach
        if len(scene_graph['nodes']) <= 1:
            # Create a simple scene graph based on the bounding box
            # Divide the bounding box into regions
            min_bound = self.bbox.get_min_bound()
            max_bound = self.bbox.get_max_bound()
            dimensions = max_bound - min_bound
            
            # Create nodes for each octant of the bounding box
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # Calculate region bounds
                        region_min = min_bound + np.array([
                            i * dimensions[0] / 2,
                            j * dimensions[1] / 2,
                            k * dimensions[2] / 2
                        ])
                        region_max = region_min + dimensions / 2
                        region_center = (region_min + region_max) / 2
                        
                        # Create region node
                        region_node = {
                            'id': f'region_{i}_{j}_{k}',
                            'type': 'region',
                            'bbox': {
                                'min': region_min.tolist(),
                                'max': region_max.tolist()
                            },
                            'center': region_center.tolist()
                        }
                        scene_graph['nodes'].append(region_node)
            
            # Add spatial relationships between regions
            for idx1, node1 in enumerate(scene_graph['nodes']):
                if node1['type'] != 'region':
                    continue
                    
                for idx2, node2 in enumerate(scene_graph['nodes'][idx1+1:], idx1+1):
                    if node2['type'] != 'region':
                        continue
                        
                    # Extract region indices from IDs
                    id1_parts = node1['id'].split('_')
                    id2_parts = node2['id'].split('_')
                    
                    i1, j1, k1 = int(id1_parts[1]), int(id1_parts[2]), int(id1_parts[3])
                    i2, j2, k2 = int(id2_parts[1]), int(id2_parts[2]), int(id2_parts[3])
                    
                    # Check if regions are adjacent
                    if (abs(i1-i2) + abs(j1-j2) + abs(k1-k2)) == 1:
                        edge = {
                            'source': node1['id'],
                            'target': node2['id'],
                            'type': 'adjacent'
                        }
                        scene_graph['edges'].append(edge)
                        
                        # Determine spatial relationship
                        if i1 < i2:
                            edge = {
                                'source': node1['id'],
                                'target': node2['id'],
                                'type': 'east_of'
                            }
                            scene_graph['edges'].append(edge)
                        elif i1 > i2:
                            edge = {
                                'source': node1['id'],
                                'target': node2['id'],
                                'type': 'west_of'
                            }
                            scene_graph['edges'].append(edge)
                            
                        if j1 < j2:
                            edge = {
                                'source': node1['id'],
                                'target': node2['id'],
                                'type': 'above'
                            }
                            scene_graph['edges'].append(edge)
                        elif j1 > j2:
                            edge = {
                                'source': node1['id'],
                                'target': node2['id'],
                                'type': 'below'
                            }
                            scene_graph['edges'].append(edge)
                            
                        if k1 < k2:
                            edge = {
                                'source': node1['id'],
                                'target': node2['id'],
                                'type': 'north_of'
                            }
                            scene_graph['edges'].append(edge)
                        elif k1 > k2:
                            edge = {
                                'source': node1['id'],
                                'target': node2['id'],
                                'type': 'south_of'
                            }
                            scene_graph['edges'].append(edge)
        
        if save:
            # Save scene graph to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.scene_graph_dir, f"scene_graph_{timestamp}.json")
            
            with open(filename, 'w') as f:
                json.dump(scene_graph, f, indent=2)
            
            print(f"Scene graph saved to {filename}")
        
        return scene_graph
    
    def capture_360_views(self, num_horizontal=12, num_vertical=3, distance_factor=2.0, output_dir=None):
        """
        Capture images of the model from multiple viewpoints around a sphere.
        
        Args:
            num_horizontal: Number of horizontal viewpoints (around the equator)
            num_vertical: Number of vertical viewpoints (from top to bottom)
            distance_factor: Distance from camera to model as a factor of model scale
            output_dir: Directory to save captured images
        
        Returns:
            List of paths to the captured images
        """
        print(f"Capturing {num_horizontal * num_vertical} views around the model...")
        
        # Calculate camera distance
        distance = self.scale * distance_factor
        
        # Create a timestamp for this capture session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = output_dir or os.path.join(self.views_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Store image paths
        image_paths = []
        
        # Capture views
        view_count = 0
        
        # Loop through vertical angles (latitude)
        for v_idx in range(num_vertical):
            # Calculate vertical angle (from top to bottom)
            v_angle = (v_idx / max(1, num_vertical - 1)) * math.pi
            
            # Loop through horizontal angles (longitude)
            for h_idx in range(num_horizontal):
                # Calculate horizontal angle
                h_angle = (h_idx / num_horizontal) * 2 * math.pi
                
                # Calculate camera position on a sphere
                x = distance * math.sin(v_angle) * math.cos(h_angle)
                y = distance * math.cos(v_angle)
                z = distance * math.sin(v_angle) * math.sin(h_angle)
                
                # Set camera position
                ctr = self.vis.get_view_control()
                ctr.set_lookat(self.center)
                ctr.set_up([0, 1, 0])  # Set up direction to Y-axis
                ctr.set_front([x, y, z])
                ctr.set_zoom(0.7)
                
                # Update view
                self.vis.poll_events()
                self.vis.update_renderer()
                
                # Capture image
                image = self.vis.capture_screen_float_buffer(do_render=True)
                
                # Convert to numpy array and then to PIL Image
                image_np = np.asarray(image) * 255
                image_np = image_np.astype(np.uint8)
                
                # Create image file path
                image_filename = f"view_{view_count:03d}_h{h_idx:02d}_v{v_idx:02d}.png"
                image_path = os.path.join(session_dir, image_filename)
                
                # Save image
                plt.imsave(image_path, image_np)
                
                # Add to list of paths
                image_paths.append(image_path)
                
                view_count += 1
                print(f"Captured view {view_count}/{num_horizontal * num_vertical}: {image_filename}")
        
        print(f"Captured {view_count} views. Images saved to {session_dir}")
        
        # Create a metadata file for this session
        metadata = {
            "timestamp": timestamp,
            "num_horizontal": num_horizontal,
            "num_vertical": num_vertical,
            "distance_factor": distance_factor,
            "image_count": view_count,
            "image_width": self.width,
            "image_height": self.height,
            "model_info": self.model_info
        }
        
        metadata_path = os.path.join(session_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Close the visualizer
        self.vis.destroy_window()
        
        return image_paths, session_dir
    
    def capture_closeup_views(self, num_views=10, distance_factor=1.2, output_dir=None):
        """
        Capture closeup views of interesting parts of the model.
        
        Args:
            num_views: Number of closeup views to capture
            distance_factor: Distance from camera to model as a factor of model scale
            output_dir: Directory to save captured images
        
        Returns:
            List of paths to the captured images
        """
        print(f"Capturing {num_views} closeup views of the model...")
        
        # Calculate camera distance
        distance = self.scale * distance_factor
        
        # Create a timestamp for this capture session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = output_dir or os.path.join(self.views_dir, f"closeups_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Store image paths
        image_paths = []
        
        # Get mesh vertices to find interesting points
        vertices = np.asarray(self.mesh.vertices)
        
        # Sample random vertices as points of interest
        if len(vertices) > 0:
            indices = np.random.choice(len(vertices), min(num_views * 2, len(vertices)), replace=False)
            points_of_interest = vertices[indices]
        else:
            # Fallback if no vertices
            points_of_interest = [self.center + np.random.randn(3) * self.scale * 0.2 for _ in range(num_views * 2)]
        
        # Capture views
        view_count = 0
        
        for i, point in enumerate(points_of_interest):
            if view_count >= num_views:
                break
                
            # Calculate random view direction (looking at the point)
            view_dir = np.random.randn(3)
            view_dir = view_dir / np.linalg.norm(view_dir)
            
            # Calculate camera position
            camera_pos = point - view_dir * distance
            
            # Set camera position
            ctr = self.vis.get_view_control()
            ctr.set_lookat(point)
            ctr.set_up([0, 1, 0])  # Set up direction to Y-axis
            ctr.set_front(view_dir)
            ctr.set_zoom(0.9)
            
            # Update view
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # Capture image
            image = self.vis.capture_screen_float_buffer(do_render=True)
            
            # Convert to numpy array and then to PIL Image
            image_np = np.asarray(image) * 255
            image_np = image_np.astype(np.uint8)
            
            # Create image file path
            image_filename = f"closeup_{view_count:03d}.png"
            image_path = os.path.join(session_dir, image_filename)
            
            # Save image
            plt.imsave(image_path, image_np)
            
            # Add to list of paths
            image_paths.append(image_path)
            
            view_count += 1
            print(f"Captured closeup view {view_count}/{num_views}: {image_filename}")
        
        print(f"Captured {view_count} closeup views. Images saved to {session_dir}")
        
        # Create a metadata file for this session
        metadata = {
            "timestamp": timestamp,
            "view_type": "closeup",
            "num_views": num_views,
            "distance_factor": distance_factor,
            "image_count": view_count,
            "image_width": self.width,
            "image_height": self.height,
            "model_info": self.model_info
        }
        
        metadata_path = os.path.join(session_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Close the visualizer
        self.vis.destroy_window()
        
        return image_paths, session_dir
    
    def capture_model_views(self, horizontal_views=12, vertical_views=6, closeup_views=5, distance_factor=2.0):
        """
        Capture both 360° views and closeups of the model
        
        Args:
            horizontal_views: Number of horizontal viewpoints
            vertical_views: Number of vertical viewpoints
            closeup_views: Number of closeup views
            distance_factor: Distance from camera to model (as a factor of model scale)
            
        Returns:
            Dictionary with paths to captured images
        """
        # Create subdirectory for views
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        view_dir = os.path.join(self.views_dir, f"session_{timestamp}")
        os.makedirs(view_dir, exist_ok=True)
        
        # Capture 360° views
        view_paths, session_dir_360 = self.capture_360_views(
            num_horizontal=horizontal_views,
            num_vertical=vertical_views,
            distance_factor=distance_factor,
            output_dir=view_dir
        )
        
        # Capture closeup views
        closeup_paths, session_dir_closeup = self.capture_closeup_views(
            num_views=closeup_views,
            distance_factor=distance_factor * 0.5,
            output_dir=view_dir
        )
        
        # Combine results
        all_views = {
            'timestamp': timestamp,
            'view_directory': view_dir,
            '360_views': view_paths,
            'closeup_views': closeup_paths
        }
        
        # Create a metadata file for this session
        metadata = {
            "timestamp": timestamp,
            "num_horizontal": horizontal_views,
            "num_vertical": vertical_views,
            "distance_factor": distance_factor,
            "image_count": len(view_paths) + len(closeup_paths),
            "image_width": self.width,
            "image_height": self.height,
            "model_info": self.model_info
        }
        
        metadata_path = os.path.join(view_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Close the visualizer
        self.vis.destroy_window()
        
        return all_views
    
    def analyze_model(self, capture_views=True, extract_features=True, generate_point_cloud=True, 
                     create_scene_graph=True, horizontal_views=12, vertical_views=6):
        """
        Comprehensive analysis of the 3D model using multiple techniques
        
        Args:
            capture_views: Whether to capture 2D views of the model
            extract_features: Whether to extract 3D geometric features
            generate_point_cloud: Whether to generate a point cloud
            create_scene_graph: Whether to create a scene graph
            horizontal_views: Number of horizontal viewpoints if capturing views
            vertical_views: Number of vertical viewpoints if capturing views
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model_info': self.model_info
        }
        
        # Step 1: Generate point cloud (needed for feature extraction)
        point_cloud = None
        if generate_point_cloud:
            print("Step 1: Generating point cloud...")
            point_cloud = self.extract_point_cloud(num_points=10000, save=True)
            results['point_cloud'] = {
                'num_points': len(point_cloud.points),
                'has_normals': point_cloud.has_normals()
            }
        
        # Step 2: Extract 3D geometric features
        if extract_features:
            print("Step 2: Extracting 3D geometric features...")
            features = self.extract_3d_features(point_cloud=point_cloud, save=True)
            results['geometric_features'] = features
        
        # Step 3: Generate scene graph
        if create_scene_graph:
            print("Step 3: Generating scene graph...")
            scene_graph = self.generate_scene_graph(save=True)
            results['scene_graph'] = {
                'num_nodes': len(scene_graph['nodes']),
                'num_edges': len(scene_graph['edges'])
            }
        
        # Step 4: Capture views (optional, for visualization and backward compatibility)
        if capture_views:
            print("Step 4: Capturing model views...")
            views = self.capture_model_views(
                horizontal_views=horizontal_views,
                vertical_views=vertical_views,
                closeup_views=5
            )
            results['views'] = views
        
        # Save the complete analysis results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"model_analysis_{timestamp}.json")
        
        # Filter out large data that's already saved elsewhere
        save_results = results.copy()
        if 'geometric_features' in save_results:
            # Remove histogram features which can be large
            if 'histogram_features' in save_results['geometric_features']:
                del save_results['geometric_features']['histogram_features']
        
        with open(filename, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"Complete model analysis saved to {filename}")
        return results

# This function is kept for backward compatibility
def capture_model_views(mesh, output_dir="captured_views", num_horizontal=12, num_vertical=3, num_closeups=5):
    """
    Convenience function to capture both 360° views and closeups of a model.
    This is a wrapper around the ModelVisualizer class for backward compatibility.
    
    Args:
        mesh: The mesh to visualize
        output_dir: Directory to save output images
        num_horizontal: Number of horizontal viewpoints
        num_vertical: Number of vertical viewpoints
        num_closeups: Number of closeup views
        
    Returns:
        Tuple of (all_image_paths, session_dirs)
    """
    visualizer = ModelVisualizer(mesh, output_dir)
    return visualizer.capture_model_views(
        horizontal_views=num_horizontal,
        vertical_views=num_vertical,
        closeup_views=num_closeups
    ) 