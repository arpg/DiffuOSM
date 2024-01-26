import os
import numpy as np
import open3d as o3d
from open3d.io import write_point_cloud

# Internal
from labels import labels
from convert_oxts_pose import *

'''
view_frame_semantics.py
'''
def read_label_bin_file(file_path):
    """
    Reads a .bin file containing point cloud labels.
    """
    labels = np.fromfile(file_path, dtype=np.int16)
    return labels.reshape(-1)

def read_bin_file(file_path):
    """
    Reads a .bin file containing point cloud data.
    Assumes each point is represented by four float32 values (x, y, z, intensity).
    """
    point_cloud = np.fromfile(file_path, dtype=np.float32)
    return point_cloud.reshape(-1, 4)

# Create a dictionary for label colors
labels_dict = {label.id: label.color for label in labels}

def color_point_cloud(points, labels):
    """
    Colors the point cloud based on the labels.
    Each point in the point cloud is assigned the color of its label.
    """
    colored_points = np.zeros_like(points[:, :3])  # Initialize with zeros, only xyz
    for i, label in enumerate(labels):
        if np.isnan(label):
            continue  # Skip NaN labels
        if label == -1:
            continue  # Skip invalid labels

        color = labels_dict.get(label, (0, 0, 0))  # Default color is black

        colored_points[i] = np.array(color) / 255.0  # Normalize to [0, 1] for Open3D
    return colored_points

def read_poses(file_path):
    transformation_matrices = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual elements
            elements = line.strip().split()
            frame_index = int(elements[0])  # Frame index is the first element

            # Check if the line contains 16 elements for a 4x4 matrix
            if len(elements[1:]) == 16:
                # Convert elements to float and reshape into 4x4 matrix
                matrix_4x4 = np.array(elements[1:], dtype=float).reshape((4, 4))
            else:
                # Otherwise, assume it is a 3x4 matrix and append a homogeneous row
                matrix_3x4 = np.array(elements[1:], dtype=float).reshape((3, 4))
                matrix_4x4 = np.vstack([matrix_3x4, np.array([0, 0, 0, 1])])

            # Store the matrix using the frame index as the key
            transformation_matrices[frame_index] = matrix_4x4

    return transformation_matrices

def transform_point_cloud(pc, transformation_matrices, frame_number):
    if frame_number >= len(transformation_matrices):
        print(f"Frame number {frame_number} is out of range.")
        return None

    # Get the transformation matrix for the current frame
    transformation_matrix = transformation_matrices.get(frame_number)

    # Separate the XYZ coordinates and intensity values
    xyz = pc[:, :3]
    intensity = pc[:, 3].reshape(-1, 1)

    # Convert the XYZ coordinates to homogeneous coordinates
    xyz_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])

    # Apply the transformation to each XYZ coordinate
    transformed_xyz = np.dot(xyz_homogeneous, transformation_matrix.T)[:, :3]

    return transformed_xyz

poses_file = '/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/vel_poses.txt'
transformation_matrices = read_poses(poses_file)
def load_and_visualize(frame_number, last_min):
    # Adjust file paths based on frame number
    pc_filepath = f'/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/KITTI360/data_3d_raw/2013_05_28_drive_0005_sync/velodyne_points/data/{frame_number:010d}.bin'
    label_filepath = f'/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/recovered/2013_05_28_drive_0005_sync/labels/{frame_number:010d}.bin'
    
    if not os.path.exists(pc_filepath) or not os.path.exists(label_filepath):
        print(f"File not found for frame number {frame_number}")
        return None, None

    # read pointcloud bin files and label bin files
    pc = read_bin_file(pc_filepath)
    pc = transform_point_cloud(pc, transformation_matrices, frame_number)
    labels_np = read_label_bin_file(label_filepath)

    # boolean mask where True represents the labels to keep
    label_mask = (labels_np == 11) | (labels_np == 0)

    # mask to filter the point cloud and labels
    pc = pc[label_mask]
    labels_np = labels_np[label_mask]

    # color the point cloud
    colored_points = color_point_cloud(pc, labels_np)
    colored_pcd = o3d.geometry.PointCloud()
    
    # Reshape pointcloud to fit in convertPoseToOxts function
    pc_reshaped = np.array([np.eye(4) for _ in range(pc.shape[0])])
    pc_reshaped[:, 0:3, 3] = pc[:, :3]

    # Convert to lat-lon-alt
    pc_reshaped = np.asarray(postprocessPoses(pc_reshaped))
    pc_lla = np.asarray(convertPoseToOxts(pc_reshaped))

    min_alt = 226.60675 #(np.min(pc_lla[:, 2]) + last_min)/2
    pc_lla[:, 2] = (pc_lla[:, 2] - min_alt)*0.00001*0

    colored_pcd.points = o3d.utility.Vector3dVector(pc_lla[:, :3])  # Only use lat, lon, alt for geometry
    colored_pcd.colors = o3d.utility.Vector3dVector(colored_points) # Set colors

    return colored_pcd, min_alt

def load_xyz_positions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    xyz_positions = []
    for line in lines:
        matrix_elements = np.array(line.split(), dtype=float)
        x, y = matrix_elements[0], matrix_elements[1]
        xyz_positions.append([x, y, 0])
    return xyz_positions

def create_point_clouds_from_xyz(xyz_positions):
    point_clouds = []
    for xyz in xyz_positions:
        # Create a point cloud with a single point
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([xyz])
        # pcd.paint_uniform_color([1, 0, 0])  # Red color for poses
        point_clouds.append(pcd)
    return point_clouds

file_path = "/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/2013_05_28_drive_0005_sync_pose2oxts.txt"
xyz_positions = load_xyz_positions(file_path)

# Create point clouds from XYZ positions
xyz_point_clouds = create_point_clouds_from_xyz(xyz_positions)

# List to hold all point cloud geometries
pcd_geometries = []

# Iterate through frame numbers and load each point cloud
frame_num = 30  # Initial frame number
last_min = 0
while frame_num < 6255:  # Assuming you want to load frames from 0 to 6255
    frame_num += 1
    pcd, new_min = load_and_visualize(frame_num, last_min)
    # print(frame_num)
    if pcd is not None:
        last_min = new_min
        frame_num += 5
        # voxel_size = 0.0000001  # example voxel size
        # pcd_ds = pcd.voxel_down_sample(voxel_size)
        pcd_geometries.append(pcd)

# Merge all point clouds in pcd_geometries into a single point cloud
merged_pcd = o3d.geometry.PointCloud()
for pcd in pcd_geometries:
    merged_pcd += pcd

# Save the merged point cloud to a PLY file
output_file_path = './output.ply'  # Specify your output file path here
o3d.io.write_point_cloud(output_file_path, merged_pcd)

print(f"Saved merged point cloud to {output_file_path}")