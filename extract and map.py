import json
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

json_file_path = 'C:/Users/crisz/Documents/ECU Classes/CSCI Graduate/Thesis/train/annos/000102.json'

with open(json_file_path, 'r') as file:
    json_data = json.load(file)

keypoints_data = {}
for item_key, item_value in json_data.items():  # Use items() to get both key and value
    if isinstance(item_value, dict) and 'landmarks' in item_value:  # Check if it's a dictionary and has 'landmarks'
        keypoints = item_value['landmarks']
        # Keypoints are stored in triples (x-coordinate, y-coordinate, visibility)
        keypoints_data[item_value['category_name']] = [(keypoints[i], keypoints[i + 1])
                                                       for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0]

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
image_path = 'C:/Users/crisz/Documents/ECU Classes/CSCI Graduate/Thesis/train/image/000102.jpg' # Replace with the correct path
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

# Map keypoints to depth map
keypoints_with_depth = {}
for category, keypoints in keypoints_data.items():
    keypoints_with_depth[category] = [(x, y, depth_map[y, x]) for x, y in keypoints]

# keypoints_with_depth now contains the keypoints with their depth value!!!!!!!

# Normalize depth map for visualization
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_visual = cv2.applyColorMap(depth_map_normalized.astype('uint8'), cv2.COLORMAP_JET)

# Draw keypoints on depth map
for category, keypoints in keypoints_with_depth.items():
    for x, y, depth in keypoints:
        # white color for keypoints, change color if needed
        cv2.circle(depth_map_visual, (int(x), int(y)), 5, (255, 255, 255), -1)

# Show depth map with keypoints
plt.imshow(depth_map_visual)
plt.axis('off')  # Hide the axis
plt.show()