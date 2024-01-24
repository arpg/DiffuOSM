import os
import numpy as np
import open3d as o3d
import osmnx as ox
from collections import namedtuple

# Internal
from labels import labels
from convert_oxts_pose import *

osm_file_path = '/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/map_0005.osm'

# Filter features for buildings and sidewalks
buildings = ox.features_from_xml(osm_file_path, tags={'building': True})
sidewalks = ox.features_from_xml(osm_file_path, tags={'highway': 'footway', 'footway': 'sidewalk'})

# Process Sidewalks
sidewalk_lines = []
for _, sidewalk in sidewalks.iterrows():
    if sidewalk.geometry.geom_type == 'LineString':
        y, x = sidewalk.geometry.xy
        for i in range(len(x) - 1):
            sidewalk_lines.append([[x[i], y[i], 0], [x[i + 1], y[i + 1], 0]])

sidewalk_line_set = o3d.geometry.LineSet()
sidewalk_points = [point for line in sidewalk_lines for point in line]
sidewalk_lines_idx = [[i, i + 1] for i in range(0, len(sidewalk_points), 2)]
sidewalk_line_set.points = o3d.utility.Vector3dVector(sidewalk_points)
sidewalk_line_set.lines = o3d.utility.Vector2iVector(sidewalk_lines_idx)
sidewalk_line_set.paint_uniform_color([0, 1, 0])  # Green color for sidewalks

# Process Buildings as LineSets
building_lines = []
for _, building in buildings.iterrows():
    if building.geometry.geom_type == 'Polygon':
        exterior_coords = building.geometry.exterior.coords
        for i in range(len(exterior_coords) - 1):
            start_point = [exterior_coords[i][1], exterior_coords[i][0], 0]
            end_point = [exterior_coords[i + 1][1], exterior_coords[i + 1][0], 0]
            building_lines.append([start_point, end_point])

building_line_set = o3d.geometry.LineSet()
building_points = [point for line in building_lines for point in line]
building_lines_idx = [[i, i + 1] for i in range(0, len(building_points), 2)]
building_line_set.points = o3d.utility.Vector3dVector(building_points)
building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings

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

poses_file = 'kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/vel_poses.txt'
transformation_matrices = read_poses(poses_file)
def load_and_visualize(frame_number, last_min):
    # Adjust file paths based on frame number
    pc_filepath = f'/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/KITTI360/data_3d_raw/2013_05_28_drive_0005_sync/velodyne_points/data/{frame_number:010d}.bin'
    label_filepath = f'/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/recovered/2013_05_28_drive_0005_sync/labels/{frame_number:010d}.bin'
    
    if not os.path.exists(pc_filepath) or not os.path.exists(label_filepath):
        print(f"File not found for frame number {frame_number}")
        return None, None

    # Step 1: Read pointcloud bin files and label bin files
    pc = read_bin_file(pc_filepath)
    pc = transform_point_cloud(pc, transformation_matrices, frame_number)
    labels_np = read_label_bin_file(label_filepath)

    # Step 2: Color the point cloud
    colored_points = color_point_cloud(pc, labels_np)
    colored_pcd = o3d.geometry.PointCloud()
    
    # Reshape pointcloud to fit in convertPoseToOxts function
    pc_reshaped = np.array([np.eye(4) for _ in range(pc.shape[0])])
    pc_reshaped[:, 0:3, 3] = pc[:, :3]

    # Convert to lat-lon-alt
    pc_reshaped = np.asarray(postprocessPoses(pc_reshaped))
    pc_lla = np.asarray(convertPoseToOxts(pc_reshaped))

    min_alt = 226.60675 #(np.min(pc_lla[:, 2]) + last_min)/2
    pc_lla[:, 2] = (pc_lla[:, 2] - min_alt)*0.00001

    colored_pcd.points = o3d.utility.Vector3dVector(pc_lla[:, :3])  # Only use lat, lon, alt for geometry
    colored_pcd.colors = o3d.utility.Vector3dVector(colored_points) # Set colors

    return colored_pcd, min_alt

# def change_frame(vis, key_code):
#     global frame_number
#     if key_code == ord('N'):
#         frame_number += 10
#     elif key_code == ord('P'):
#         frame_number -= 10
#     else:
#         return False
#     new_pcd = load_and_visualize(frame_number)

#     if new_pcd is not None:
#         voxel_size = 0.00001  # example voxel size
#         new_pcd_ds = new_pcd.voxel_down_sample(voxel_size)
#         # vis.clear_geometries()
#         # vis.add_geometry(sidewalk_line_set)
#         # vis.add_geometry(building_line_set)
#         vis.add_geometry(new_pcd_ds)
#     return True

# frame_number = 30  # starting frame number
# initial_pcd = load_and_visualize(frame_number)

# if initial_pcd:
#     key_to_callback = {
#         ord('N'): lambda vis: change_frame(vis, ord('N')),
#         ord('P'): lambda vis: change_frame(vis, ord('P'))
#     }
#     o3d.visualization.draw_geometries_with_key_callbacks([initial_pcd], key_to_callback)


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

file_path = "/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/OSM_merge/2013_05_28_drive_0005_sync_pose2oxts.txt"
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
        frame_num += 100
        # voxel_size = 0.0000001  # example voxel size
        # pcd_ds = pcd.voxel_down_sample(voxel_size)
        pcd_geometries.append(pcd)

# print(last_min)

# # pcd_geometries.extend(xyz_point_clouds)
pcd_geometries.append(building_line_set)
o3d.visualization.draw_geometries(pcd_geometries)