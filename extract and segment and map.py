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
    """
    image_height = depth_map.shape[0]
    image_width = depth_map.shape[1]

    # Convert FOV from degrees to radians
    fov_horizontal_rad = np.deg2rad(60)

    # Calculate focal length in pixels
    focal_length_x = image_width / (2 * np.tan(fov_horizontal_rad / 2))

    # If the aspect ratio is 1:1
    focal_length_y = focal_length_x

    # If the aspect ratio is not 1:1, calculate the vertical FOV
    # aspect_ratio = image_width / image_height
    # fov_vertical_rad = 2 * np.arctan(np.tan(fov_horizontal_rad / 2) / aspect_ratio)
    # focal_length_y = image_height / (2 * np.tan(fov_vertical_rad / 2))

    # Calculate optical centers
    center_x = image_width / 2
    center_y = image_height / 2
    """
    
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

# Allowed categories for tops and dresses
allowed_categories = ["short sleeve top", "long sleeve top", "vest dress", "long sleeve dress", "short sleeve dress", "sling", "sling dress"]

# Original JSON file path
json_file_path = 'C:/Users/crisz/Documents/ECU Classes/CSCI Graduate/Thesis/train/annos/000259.json'

# Define output directories
depth_map_dir = r"C:\Users\crisz\Documents\ECU Classes\CSCI Graduate\Thesis\Depth Map Output PNG"
ply_output_dir = r"C:\Users\crisz\Documents\ECU Classes\CSCI Graduate\Thesis\PLY output"
mesh_output_dir = r"C:\Users\crisz\Documents\ECU Classes\CSCI Graduate\Thesis\mesh output"
segmented_depth_map_dir = r"C:\Users\crisz\Documents\ECU Classes\CSCI Graduate\Thesis\segment map"


# Ensure directories exist
os.makedirs(depth_map_dir, exist_ok=True)
os.makedirs(ply_output_dir, exist_ok=True)
os.makedirs(mesh_output_dir, exist_ok=True)
os.makedirs(segmented_depth_map_dir, exist_ok=True)

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
for item_key, item_value in json_data.items():
    if isinstance(item_value, dict) and 'landmarks' in item_value and item_value['category_name'] in allowed_categories:  # Check if it's a dictionary, has 'landmarks', and is an allowed category
        keypoints = item_value['landmarks']
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

depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_visual = cv2.applyColorMap(depth_map_normalized.astype('uint8'), cv2.COLORMAP_JET)
for category, keypoints in keypoints_data.items():
    for x, y in keypoints:
        # white color for keypoints, change color if needed
        cv2.circle(depth_map_visual, (int(x), int(y)), 5, (255, 255, 255), -1)

# After converting the depth map to a NumPy array
print(depth_map.shape)
print(np.count_nonzero(depth_map))

# Extract keypoints for 'item1' which represents the shirt
shirt_keypoints = []
if 'item1' in json_data and 'landmarks' in json_data['item1']:
    keypoints = json_data['item1']['landmarks']
    shirt_keypoints = [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0]

# Check if shirt keypoints were found
if not shirt_keypoints:
    print("No shirt keypoints found. Skipping this file.")
    sys.exit()

# Create the segmentation mask for the shirt
mask = np.zeros_like(depth_map, dtype=np.uint8)  # Use the original depth map's shape for mask creation
poly = np.array([shirt_keypoints], dtype=np.int32)
cv2.fillPoly(mask, poly, 255)  # Fill with white where the shirt is

# Apply the mask to the depth map to segment the shirt
segmented_depth_map = cv2.bitwise_and(depth_map, depth_map, mask=mask)

# Now scale the segmented depth map
segmented_depth_map_scaled = segmented_depth_map * 30  # Adjust the scaling factor as needed

# Visualize the original image, original depth map, and segmented depth map for comparison
plt.figure(figsize=(15, 5))

# Original image visualization
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Original depth map visualization
plt.subplot(1, 3, 2)
plt.imshow(depth_map_visual)
plt.title('Original Depth Map')
plt.axis('off')

# Segmented and scaled depth map visualization
segmented_depth_map_visual = cv2.normalize(segmented_depth_map_scaled, None, 0, 255, cv2.NORM_MINMAX)
segmented_depth_map_visual = cv2.applyColorMap(segmented_depth_map_visual.astype('uint8'), cv2.COLORMAP_JET)
plt.subplot(1, 3, 3)
plt.imshow(segmented_depth_map_visual)
plt.title('Segmented Depth Map')
plt.axis('off')
plt.show()

# Create a PointCloud object from Open3D
points_3d = depth_map_to_point_cloud(segmented_depth_map_scaled)
point_cloud_o3d = o3d.geometry.PointCloud()
points_3d = np.asarray(points_3d)
points_3d[:, 1] = -points_3d[:, 1]  # Flip the y-coordinates
point_cloud_o3d.points = o3d.utility.Vector3dVector(points_3d)

# After creating the point cloud
print(np.asarray(point_cloud_o3d.points).shape)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud_o3d])

# Estimate normals for the point cloud
point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Apply Poisson Surface Reconstruction
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud_o3d, depth=9)[0]

# Compute the vertices' normals
poisson_mesh.compute_vertex_normals()

# Compute the distance of each vertex from the camera
distances = np.linalg.norm(np.asarray(poisson_mesh.vertices), axis=1)

# Normalize the distances to the range [0, 1]
distances_normalized = (distances - distances.min()) / (distances.max() - distances.min())

# Create a colormap based on the distances
colors = plt.cm.viridis(distances_normalized)

# Set the color of each vertex
poisson_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize the mesh
o3d.visualization.draw_geometries([poisson_mesh])

# Extract the base name of the file and remove the extension to get the number
file_number = os.path.splitext(os.path.basename(image_path))[0]

# Use the file number when saving the output files
ply_file = os.path.join(ply_output_dir, f"{file_number}_point_cloud.ply")
mesh_file = os.path.join(mesh_output_dir, f"{file_number}_mesh.obj")
segment_file = os.path.join(segmented_depth_map_dir, f"{file_number}_segment.png")
depth_map_file = os.path.join(depth_map_dir, f"{base_filename}_depth_map.png")

# Save the depth map image
cv2.imwrite(depth_map_file, depth_map_visual)

# Save the segment map visual
cv2.imwrite(segment_file, segmented_depth_map_visual)

# Save the point cloud
o3d.io.write_point_cloud(ply_file, point_cloud_o3d)

# Save the mesh
o3d.io.write_triangle_mesh(mesh_file, poisson_mesh)