import open3d as o3d
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from datetime import datetime

class ModelVisualizer:
    """
    Captures 360° images around a 3D model from multiple viewpoints.
    Generates a comprehensive visual dataset for further analysis.
    """
    
    def __init__(self, mesh, output_dir="captured_views"):
        """Initialize the visualizer with a mesh and output directory"""
        # Store the mesh (handle both EnhancedMesh and regular mesh)
        if hasattr(mesh, 'mesh'):
            self.mesh = mesh.mesh
            self.model_info = mesh.model_info
        else:
            self.mesh = mesh
            self.model_info = getattr(mesh, 'model_info', {})
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up visualization parameters
        self.width = 1024
        self.height = 768
        self.background_color = [1, 1, 1]  # White background
        
        # Calculate model center and scale
        self.center = self.mesh.get_center()
        bbox = self.mesh.get_axis_aligned_bounding_box()
        self.scale = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        
        # Initialize the visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        self.vis.add_geometry(self.mesh)
        
        # Set default rendering options
        self._set_default_rendering_options()
        
        print(f"Model visualizer initialized. Output directory: {self.output_dir}")
    
    def _set_default_rendering_options(self):
        """Set default rendering options for the visualizer"""
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(self.background_color)
        opt.point_size = 5.0
        opt.line_width = 2.0
        opt.mesh_show_wireframe = False
        opt.mesh_show_back_face = True
        
        # Set lighting for better visualization
        opt.light_on = True
        
        # Update view
        self.vis.update_renderer()
    
    def capture_360_views(self, num_horizontal=12, num_vertical=3, distance_factor=2.0):
        """
        Capture images of the model from multiple viewpoints around a sphere.
        
        Args:
            num_horizontal: Number of horizontal viewpoints (around the equator)
            num_vertical: Number of vertical viewpoints (from top to bottom)
            distance_factor: Distance from camera to model as a factor of model scale
        
        Returns:
            List of paths to the captured images
        """
        print(f"Capturing {num_horizontal * num_vertical} views around the model...")
        
        # Calculate camera distance
        distance = self.scale * distance_factor
        
        # Create a timestamp for this capture session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"session_{timestamp}")
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
            import json
            json.dump(metadata, f, indent=4)
        
        # Close the visualizer
        self.vis.destroy_window()
        
        return image_paths, session_dir
    
    def capture_closeup_views(self, num_views=10, distance_factor=1.2):
        """
        Capture closeup views of interesting parts of the model.
        
        Args:
            num_views: Number of closeup views to capture
            distance_factor: Distance from camera to model as a factor of model scale
        
        Returns:
            List of paths to the captured images
        """
        print(f"Capturing {num_views} closeup views of the model...")
        
        # Calculate camera distance
        distance = self.scale * distance_factor
        
        # Create a timestamp for this capture session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"closeups_{timestamp}")
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
            import json
            json.dump(metadata, f, indent=4)
        
        # Close the visualizer
        self.vis.destroy_window()
        
        return image_paths, session_dir

    def capture_model_views(self, horizontal_views=12, vertical_views=3, closeup_views=5):
        """
        Convenience function to capture both 360° views and closeups of the model.
        
        Args:
            horizontal_views: Number of horizontal viewpoints
            vertical_views: Number of vertical viewpoints
            closeup_views: Number of closeup views
            
        Returns:
            Tuple of (all_image_paths, session_dirs)
        """
        print(f"Capturing model views: {horizontal_views}x{vertical_views} 360° views and {closeup_views} closeups")
        
        # Capture 360° views
        images_360, session_dir_360 = self.capture_360_views(
            num_horizontal=horizontal_views,
            num_vertical=vertical_views
        )
        
        # Reinitialize visualizer (it was closed in capture_360_views)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        self.vis.add_geometry(self.mesh)
        self._set_default_rendering_options()
        
        # Capture closeup views
        if closeup_views > 0:
            images_closeup, session_dir_closeup = self.capture_closeup_views(
                num_views=closeup_views
            )
        else:
            images_closeup = []
            session_dir_closeup = None
        
        # Combine all images
        all_images = images_360 + images_closeup
        session_dirs = [session_dir_360]
        if session_dir_closeup:
            session_dirs.append(session_dir_closeup)
        
        print(f"Captured a total of {len(all_images)} images")
        return all_images, session_dirs

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