import json
import cv2
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import sys
import tkinter as tk
from tkinter import filedialog

def depth_map_to_point_cloud(depth_map):
    """
    Convert a depth map to a 3D point cloud.

    Args:
        depth_map (numpy.ndarray): The depth map.

    Returns:
        numpy.ndarray: The 3D point cloud.
    """
    # These parameters are assumed, might need to adjust them
    focal_length_x = depth_map.shape[1]  # Placeholder for focal length in x
    focal_length_y = depth_map.shape[0]  # Placeholder for focal length in y
    center_x = depth_map.shape[1] // 2
    center_y = depth_map.shape[0] // 2

    # Create a grid of (x, y) coordinates corresponding to each pixel
    x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
    
    # Normalize (x, y) coordinates to the camera coordinates
    x = (x - center_x)
    y = (y - center_y)
    
    # Unproject
    # z keeps the original depth values, x and y are reprojected
    z = depth_map
    #x = x * z
    #y = y * z
    
    # Stack to create 3D point cloud
    point_cloud = np.dstack((x, y, z)).reshape(-1, 3)
    
    return point_cloud

# Original JSON file path
json_file_path = 'C:/Users/crisz/Documents/ECU Classes/CSCI Graduate/Thesis/train/annos/000259.json'

# Define output directories
depth_map_dir = r"C:\Users\crisz\Documents\ECU Classes\CSCI Graduate\Thesis\Depth Map Output PNG"
ply_output_dir = r"C:\Users\crisz\Documents\ECU Classes\CSCI Graduate\Thesis\PLY output"
mesh_output_dir = r"C:\Users\crisz\Documents\ECU Classes\CSCI Graduate\Thesis\mesh output"

# Ensure directories exist
os.makedirs(depth_map_dir, exist_ok=True)
os.makedirs(ply_output_dir, exist_ok=True)
os.makedirs(mesh_output_dir, exist_ok=True)

# Extract the base filename without the extension
base_filename = os.path.splitext(os.path.basename(json_file_path))[0]

with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# Image path 
image_path = os.path.join('C:/Users/crisz/Documents/ECU Classes/CSCI Graduate/Thesis/train/image', base_filename + '.jpg') 
img = np.array(Image.open(image_path))

# Check conditions
if json_data['item1']['zoom_in'] != 1 or json_data['item1']['viewpoint'] not in [2, 3]:
    print("Conditions not met. Skipping this file.")
    sys.exit()  # Stop the process

keypoints_data = {}
for item_key, item_value in json_data.items():  # Use items() to get both key and value
    if isinstance(item_value, dict) and 'landmarks' in item_value:  # Check if it's a dictionary and has 'landmarks'
        keypoints = item_value['landmarks']
        # Keypoints are stored in triples (x-coordinate, y-coordinate, visibility)
        keypoints_data[item_value['category_name']] = [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0]

# Now, `keypoints_data` contains keypoints organized by category name, hopefully

# Load  MiDaS model
midas_model_type = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if midas_model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

# Apply MiDaS transforms to image
input_batch = transform(img).to(device)

# Get depth map
with torch.no_grad():
    depth_prediction = midas(input_batch)
    depth_prediction = torch.nn.functional.interpolate(
        depth_prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Convert depth map to a NumPy array
depth_map = depth_prediction.cpu().numpy()

# After converting the depth map to a NumPy array
print(depth_map.shape)
print(np.count_nonzero(depth_map))

# Scale the depth map
depth_map_scaled = depth_map * 30  # Adjust the scaling factor as needed

# Use the function to get the 3D points
points_3d = depth_map_to_point_cloud(depth_map_scaled)

# Create a PointCloud object from Open3D
point_cloud_o3d = o3d.geometry.PointCloud()
points_3d = np.asarray(points_3d)
points_3d[:, 0] = -points_3d[:, 0]  # Flip the x-coordinates
points_3d[:, 1] = -points_3d[:, 1]  # Flip the y-coordinates
point_cloud_o3d.points = o3d.utility.Vector3dVector(points_3d)

# After creating the point cloud
print(np.asarray(point_cloud_o3d.points).shape)

depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_visual = cv2.applyColorMap(depth_map_normalized.astype('uint8'), cv2.COLORMAP_JET)
for category, keypoints in keypoints_data.items():
    for x, y in keypoints:
        # white color for keypoints, change color if needed
        cv2.circle(depth_map_visual, (int(x), int(y)), 5, (255, 255, 255), -1)

# Show depth map with keypoints
plt.imshow(depth_map_visual)
plt.axis('off')  # Hide the axis
plt.show()

# Construct the file path for the depth map image
depth_map_file = os.path.join(depth_map_dir, f"{base_filename}_depth_map.png")

# Save the depth map image
cv2.imwrite(depth_map_file, depth_map_visual)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud_o3d])

# Estimate normals for the point cloud
point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Apply Poisson Surface Reconstruction
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud_o3d, depth=9)[0]

# Compute the vertices' normals
poisson_mesh.compute_vertex_normals()

# Set the color of the mesh (RGB values range from 0 to 1, for example, red color)
poisson_mesh.paint_uniform_color([1, 0, 0])  # Red color

# Visualize the mesh with a red color
o3d.visualization.draw_geometries([poisson_mesh])

# Extract the base name of the file and remove the extension to get the number
file_number = os.path.splitext(os.path.basename(image_path))[0]

# Use the file number when saving the output files
ply_file = os.path.join(ply_output_dir, f"{file_number}_point_cloud.ply")
mesh_file = os.path.join(mesh_output_dir, f"{file_number}_mesh.obj")

# Save the depth map visual
cv2.imwrite(depth_map_file, depth_map_visual)

# Save the point cloud
o3d.io.write_point_cloud(ply_file, point_cloud_o3d)

# Save the mesh
o3d.io.write_triangle_mesh(mesh_file, poisson_mesh)

