import json
import cv2
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
    x = (x - center_x) / focal_length_x
    y = (y - center_y) / focal_length_y
    
    # Unproject
    # z keeps the original depth values, x and y are reprojected
    z = depth_map
    x = x * z
    y = y * z
    
    # Stack to create 3D point cloud
    point_cloud = np.dstack((x, y, z)).reshape(-1, 3)
    
    return point_cloud

#use json from deepfashion training
json_file_path = 'C:/Users/crisz/Documents/ECU Classes/CSCI Graduate/Thesis/train/annos/000259.json'

with open(json_file_path, 'r') as file:
    json_data = json.load(file)

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

# Load image corresponding to the JSON file
image_path = 'C:/Users/crisz/Documents/ECU Classes/CSCI Graduate/Thesis/train/image/000259.jpg' # Replace with the correct path
img = np.array(Image.open(image_path))

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

# Use the function to get the 3D points
points_3d = depth_map_to_point_cloud(depth_map)

# Create a PointCloud object from Open3D
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(points_3d)

# Draw keypoints on depth map for visualization
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

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud_o3d])

# Create a root Tkinter window and hide it
root = tk.Tk()
root.withdraw()

# Open a save file dialog
file_path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY files", "*.ply")])

# Save the point cloud as a .ply file
o3d.io.write_point_cloud(file_path, point_cloud_o3d)


"""
# Estimate normals
o3d.geometry.estimate_normals(point_cloud_o3d, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Create a triangle mesh from the point cloud using the Ball-Pivoting Algorithm
radius = 0.02
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud_o3d, o3d.utility.DoubleVector([radius, radius * 2]))

# Visualize the mesh
o3d.visualization.draw_geometries([bpa_mesh])

# Assume `keypoints_2d` is a list of 2D keypoints from the JSON
# and `depth_map` is the depth map

keypoints_3d = []

for keypoint_2d in keypoints_2d:
    # Project the 2D keypoint onto the depth map
    depth = depth_map[int(keypoint_2d[1]), int(keypoint_2d[0])]
    
    # Convert the 2D keypoint and depth to a 3D point
    keypoint_3d = convert_2d_to_3d(keypoint_2d, depth, camera_parameters)
    
    keypoints_3d.append(keypoint_3d)

o3d.visualization.draw_geometries([keypoints])
"""