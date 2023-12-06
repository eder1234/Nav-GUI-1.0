import tkinter as tk
from tkinter import filedialog, Label, Button, Entry, Frame, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy

# Global variables to hold the loaded images
current_color_image = None
target_color_image = None
kp1 = None
kp2 = None
matches = None
points1 = None
points2 = None
colors1 = None
colors2= None

# Camera intrinsic parameters (assumed to be globally known for this example)
K = np.array([
    [929.994628906, 0.0, 643.123168945],
    [0.0, 929.571960449, 356.984924316],
    [0.0, 0.0, 1.0]
])


# Function to browse for the root path
def browse_files():
    folder_selected = filedialog.askdirectory()
    entry_root_file.delete(0, tk.END)  # Remove current text in entry
    entry_root_file.insert(0, folder_selected)  # Insert the selected folder path

# Function to load and display color images
# Function to load and display color images
def load_images():
    global current_color_image, target_color_image  # Declare as globals
        
    root_path = entry_root_file.get()
    current_id = entry_current_id.get()
    target_id = entry_target_id.get()

    # Validate IDs
    if not current_id.isdigit() or not target_id.isdigit():
        messagebox.showerror("Error", "Please enter a valid numeric ID.")
        return

    # Construct file paths for color images
    current_color_path = os.path.join(root_path, f'color/frame_{int(current_id):04d}.png')
    target_color_path = os.path.join(root_path, f'color/frame_{int(target_id):04d}.png')

    # Validate file existence
    if not os.path.isfile(current_color_path) or not os.path.isfile(target_color_path):
        messagebox.showerror("Error", "Files do not exist in the specified path.")
        return

    # Load color images
    current_color_image = cv2.cvtColor(cv2.imread(current_color_path), cv2.COLOR_BGR2RGB)
    target_color_image = cv2.cvtColor(cv2.imread(target_color_path), cv2.COLOR_BGR2RGB)

    # Resize for display
    display_size = (320, 240)  # Adjust this as needed
    current_color_image_display = cv2.resize(current_color_image, display_size)
    target_color_image_display = cv2.resize(target_color_image, display_size)

    # Convert to PhotoImage
    current_color_photo = ImageTk.PhotoImage(image=Image.fromarray(current_color_image_display))
    target_color_photo = ImageTk.PhotoImage(image=Image.fromarray(target_color_image_display))

    # Update the image labels
    label_current_color_image.configure(image=current_color_photo)
    label_current_color_image.image = current_color_photo  # Keep a reference

    label_target_color_image.configure(image=target_color_photo)
    label_target_color_image.image = target_color_photo  # Keep a reference
    
    print(f"Current color image loaded with shape: {current_color_image.shape}")
    print(f"Target color image loaded with shape: {target_color_image.shape}")


def match_orb_keypoints(img1, img2):
    global kp1, kp2, matches
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

def extract_and_match():
    global current_color_image, target_color_image  # Declare as globals
    
    # Validate if images are loaded
    if 'current_color_image' not in globals() or 'target_color_image' not in globals():
        messagebox.showerror("Error", "Please load the images first.")
        return
    
    # Proceed with feature matching using ORB
    kp1, kp2, matches = match_orb_keypoints(current_color_image, target_color_image)
    
    # Draw matches on the images
    matched_img = cv2.drawMatches(current_color_image, kp1, target_color_image, kp2, matches, None, flags=2)
    
    # Resize the matched image for display
    matched_img_resized = cv2.resize(matched_img, (640, 240))    
    
    # Convert to PhotoImage
    matched_photo = ImageTk.PhotoImage(image=Image.fromarray(matched_img_resized))
    
    # Update the label to display the matches
    matched_point_images.configure(image=matched_photo)
    matched_point_images.image = matched_photo  # Keep a reference
    
    print(f"Number of keypoints in current image: {len(kp1)}")
    print(f"Number of keypoints in target image: {len(kp2)}")
    print(f"Number of matches: {len(matches)}")


# Function to generate 3D points from matched points and depth images
def generate_point_clouds():
    print("Entered generate_point_clouds function")
    global current_color_image, target_color_image, kp1, kp2, matches, points1, points2, colors1, colors2

    # Validate if keypoints and matches are computed
    if kp1 is None or kp2 is None or matches is None:
        messagebox.showerror("Error", "Please extract and match keypoints first.")
        return
    
    # Construct file paths for depth images
    root_path = entry_root_file.get()
    current_id = entry_current_id.get()
    target_id = entry_target_id.get()
    current_depth_path = os.path.join(root_path, f'depth/frame_{int(current_id):04d}.png')
    target_depth_path = os.path.join(root_path, f'depth/frame_{int(target_id):04d}.png')

    # Load depth images
    current_depth_image = cv2.imread(current_depth_path, -1)
    target_depth_image = cv2.imread(target_depth_path, -1)

    if current_depth_image is None or target_depth_image is None:
        messagebox.showerror("Error", "Unable to load depth images.")
        return

    # Generate 3D points from the depth images and matched keypoints
    points1, colors1, points2, colors2 = get_3d_points(kp1, kp2, matches, current_depth_image, target_depth_image, K)

    # Here you would continue with creating the point clouds and processing them
    # For now, we can just print the number of points to verify
    print(f"Generated point cloud for current image with {len(points1)} points")
    print(f"Generated point cloud for target image with {len(points2)} points")

def get_3d_points(kp1, kp2, matches, depth_img1, depth_img2, K):
    global current_color_image, target_color_image
    
    points1 = []
    points2 = []
    colors1 = []
    colors2 = []
    for m in matches:
        # Query image
        u1, v1 = kp1[m.queryIdx].pt
        u1, v1 = int(u1), int(v1)
        z1 = depth_img1[v1, u1] / 1000.0  # Convert to meters
        if z1 == 0: continue  # Skip if depth is invalid
        x1 = (u1 - K[0][2]) * z1 / K[0][0]
        y1 = (v1 - K[1][2]) * z1 / K[1][1]
        points1.append((x1, y1, z1))
        colors1.append(current_color_image[v1, u1])

        # Train image
        u2, v2 = kp2[m.trainIdx].pt
        u2, v2 = int(u2), int(v2)
        z2 = depth_img2[v2, u2] / 1000.0  # Convert to meters
        if z2 == 0: continue  # Skip if depth is invalid
        x2 = (u2 - K[0][2]) * z2 / K[0][0]
        y2 = (v2 - K[1][2]) * z2 / K[1][1]
        points2.append((x2, y2, z2))
        colors2.append(target_color_image[v2, u2])

    return np.array(points1), np.array(colors1), np.array(points2), np.array(colors2)

def execute_global_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, distance_threshold):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh, distance_threshold*2, #distance doubled
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def refine_registration(source_pcd, target_pcd, distance_threshold, result_ransac):
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, distance_threshold*1.5, result_ransac.transformation, #increasing distance
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def compute_action():
    global points1, colors1, points2, colors2, action_label_text, transformed_pcd

    # Ensure point clouds are available
    if points1 is None or points2 is None:
        messagebox.showerror("Error", "Point clouds not available.")
        return

    # Convert numpy arrays to Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(points1)
    source_pcd.colors = o3d.utility.Vector3dVector(colors1 / 255.0)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(points2)
    target_pcd.colors = o3d.utility.Vector3dVector(colors2 / 255.0)

    # Downsample, remove outliers, and compute normals for source and target point clouds
    voxel_size = 0.05  # Define the voxel size for downsampling
    source_processed = compute_normals(remove_outliers(downsample_point_cloud(source_pcd, voxel_size)))
    target_processed = compute_normals(remove_outliers(downsample_point_cloud(target_pcd, voxel_size)))

    # Compute FPFH features for RANSAC
    source_fpfh = compute_fpfh_feature(source_processed, voxel_size)
    target_fpfh = compute_fpfh_feature(target_processed, voxel_size)

    # Execute global registration (RANSAC)
    distance_threshold = voxel_size * 1.5
    result_ransac = execute_global_registration(source_processed, target_processed, source_fpfh, target_fpfh, distance_threshold)

    # Check if RANSAC registration was successful
    if not result_ransac:
        messagebox.showerror("Error", "RANSAC registration failed.")
        return

    # Refine registration with ICP based on the rough alignment from RANSAC
    result_icp = refine_registration(source_processed, target_processed, distance_threshold, result_ransac)
    print(result_icp.fitness)
    # Check if ICP refinement was successful
    if result_icp.fitness < 0.1:  # Fitness threshold can be adjusted
        messagebox.showerror("Error", "ICP registration failed.")
        
        transformed_pcd = copy.deepcopy(source_pcd)
        transformed_pcd.transform(result_icp.transformation)
        # Determine action based on the transformation matrix
        action = determine_bot_action(result_icp.transformation)
        print(f'Action would be: {action}')
        return
    
    transformed_pcd = copy.deepcopy(source_pcd)
    transformed_pcd.transform(result_icp.transformation)
    # Determine action based on the transformation matrix
    action = determine_bot_action(result_icp.transformation)
    print(f'Action: {action}')

    # Optionally, display the result or update the GUI with the action
    #messagebox.showinfo("Registration Result", f"Action: {action}")
    action_label_text.set(f"Action: {action}")


def downsample_point_cloud(pcd, voxel_size=0.05):
    return pcd.voxel_down_sample(voxel_size)

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    return pcd.remove_statistical_outlier(nb_neighbors, std_ratio)[0]

def compute_normals(pcd, radius=0.1, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    return pcd

def compute_fpfh_feature(pcd, voxel_size):
    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

def determine_bot_action(T_result, verbose=True):
    """
    Determine the action a bot should take based on the transformation matrix.
    Args:
    T (np.array): A 4x4 transformation matrix containing rotation and translation.
    Returns:
    str: The action the bot should take: 'Move Forward', 'Turn Right', 'Turn Left', or 'Stop'.
    """
    # Extract the translation vector and Euler angles
    print('Processing action')
    T = np.copy(T_result)
    translation = T[0:3, 3]
    rotation_matrix = T[0:3, 0:3]
    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    
    if verbose:
        print(f'Translation: {translation}')
        print(f'Angles (xyz): {euler_angles}')

    # Define thresholds
    forward_threshold = 0.5  # meters
    lateral_threshold = 0.2  # meters
    yaw_threshold = 10       # degrees

    # Check translation for forward/backward movement
    if translation[0] < -forward_threshold:
        action_forward = 'Move Forward'
    else:
        action_forward = 'Stop'  # If the bot is close enough to the target

    # Check lateral translation and yaw angle for turning
    if translation[1] < -lateral_threshold or euler_angles[2] < -yaw_threshold:
        action_turn = 'Turn Right'
    elif translation[1] > lateral_threshold or euler_angles[2] > yaw_threshold:
        action_turn = 'Turn Left'
    else:
        action_turn = None  # No turn is needed if within thresholds

    # Combine actions: prioritize turning over moving forward
    if action_turn:
        return action_turn
    else:
        return action_forward
    
def save_initial_pc():
    global points1, colors1, points2, colors2
    if points1 is None or points2 is None:
        messagebox.showerror("Error", "Initial point clouds are not available.")
        return

    # Create Open3D point cloud objects for both clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.colors = o3d.utility.Vector3dVector(np.array(colors1) / 255.0)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.colors = o3d.utility.Vector3dVector(np.array(colors2) / 255.0)

    # Combine both point clouds
    combined_pcd = pcd1 + pcd2

    # Save the combined point cloud in PLY format
    o3d.io.write_point_cloud("initial_combined_point_cloud.ply", combined_pcd)
    messagebox.showinfo("Save Point Cloud", "Initial combined point cloud saved as 'initial_combined_point_cloud.ply'")


def save_aligned_pc():
    global transformed_pcd
    if not hasattr(transformed_pcd, "points"):
        messagebox.showerror("Error", "Aligned point cloud is not available.")
        return

    # Save the aligned point cloud in PLY format
    o3d.io.write_point_cloud("aligned_point_cloud.ply", transformed_pcd)
    messagebox.showinfo("Save Point Cloud", "Aligned point cloud saved as 'aligned_point_cloud.ply'")


def save_point_cloud(filename, points, colors):
    # Create an Open3D point cloud object
    pc = o3d.geometry.PointCloud()
    
    # Assign points and colors to the point cloud
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizing colors
    
    # Save the point cloud in PLY format
    o3d.io.write_point_cloud(filename, pc)
    print(f"Point cloud saved as '{filename}'")

def save_current_pc():
    global points1, colors1
    if points1 is None or colors1 is None:
        messagebox.showerror("Error", "Current point cloud is not available.")
        return
    save_point_cloud("current_point_cloud.ply", points1, colors1)

def save_target_pc():
    global points2, colors2
    if points2 is None or colors2 is None:
        messagebox.showerror("Error", "Target point cloud is not available.")
        return
    save_point_cloud("target_point_cloud.ply", points2, colors2)

def reset_application():
    global current_color_image, target_color_image, kp1, kp2, matches, points1, points2, colors1, colors2, transformed_pcd

    # Reset global variables
    current_color_image = None
    target_color_image = None
    kp1 = None
    kp2 = None
    matches = None
    points1 = None
    points2 = None
    colors1 = None
    colors2 = None
    transformed_pcd = None

    # Clear GUI elements (update labels, entries, etc.)
    entry_root_file.delete(0, tk.END)
    entry_current_id.delete(0, tk.END)
    entry_target_id.delete(0, tk.END)
    action_label_text.set("Action: None")
    # Reset image labels if you have them
    # label_current_color_image.config(image='')
    # label_target_color_image.config(image='')

    # Add any additional reset logic as needed


# Initialize the main application window
root = tk.Tk()
root.title('Navigation Algorithm Evaluation GUI')

# Define the layout frames
top_frame = Frame(root)
middle_frame = Frame(root)
bottom_frame = Frame(root)

# Global variable to update the label text
action_label_text = tk.StringVar()
action_label_text.set("Action: None")

# Create a label for displaying the action
action_label = Label(middle_frame, textvariable=action_label_text)
action_label.grid(row=3, column=0, columnspan=2)

# Layout for the top frame
Label(top_frame, text='root_file').grid(row=0, column=0, sticky='ew')
entry_root_file = Entry(top_frame)
entry_root_file.grid(row=0, column=1, sticky='ew')
Button(top_frame, text='Browse', command=browse_files).grid(row=0, column=2, sticky='ew')

Label(top_frame, text='current_id').grid(row=1, column=0, sticky='ew')
entry_current_id = Entry(top_frame)
entry_current_id.grid(row=1, column=1, sticky='ew')

Label(top_frame, text='target_id').grid(row=1, column=2, sticky='ew')
entry_target_id = Entry(top_frame)
entry_target_id.grid(row=1, column=3, sticky='ew')

Button(top_frame, text='Load Images', command=load_images).grid(row=1, column=4, sticky='ew')

# Layout for the middle frame
label_current_color_image = Label(middle_frame, text='current_color_image')
label_current_color_image.grid(row=0, column=0, padx=10, pady=10)

label_target_color_image = Label(middle_frame, text='target_color_image')
label_target_color_image.grid(row=0, column=1, padx=10, pady=10)

Button(middle_frame, text='Extract and Match', command=extract_and_match).grid(row=1, column=0, columnspan=1, sticky='ew')# Add a button for generating point clouds
Button(middle_frame, text='Generate Point Clouds', command=generate_point_clouds).grid(row=1, column=1, columnspan=1, sticky='ew')
Button(middle_frame, text='Compute Action', command=compute_action).grid(row=2, column=0, columnspan=2)
#Label(middle_frame, text='action').grid(row=3, column=0, columnspan=2)

matched_point_images = Label(middle_frame, text='matched_point_images')
matched_point_images.grid(row=4, column=0, columnspan=2)

# Layout for the bottom frame
Button(bottom_frame, text='Save Initial PC', command=save_initial_pc).grid(row=0, column=0, sticky='ew')
Button(bottom_frame, text='Save Aligned PC', command=save_aligned_pc).grid(row=0, column=1, sticky='ew')

# Save Current Point Cloud Button
Button(bottom_frame, text='Save Current PC', command=save_current_pc).grid(row=0, column=2, sticky='ew')

# Save Target Point Cloud Button
Button(bottom_frame, text='Save Target PC', command=save_target_pc).grid(row=0, column=3, sticky='ew')
Button(bottom_frame, text='Reset', command=reset_application).grid(row=0, column=4, sticky='ew')

#Button(bottom_frame, text='Compute Estimate Pose', command=perform_point_cloud_registration).grid(row=0, column=4, sticky='ew')

# Pack the frames onto the root window
top_frame.pack(fill='x')
middle_frame.pack(fill='both', expand=True)
bottom_frame.pack(fill='x')

# Start the main application loop
root.mainloop()
