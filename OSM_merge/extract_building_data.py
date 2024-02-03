'''
Doncey Albin

'''

import os
import open3d as o3d
from open3d.visualization import gui
import numpy as np
from collections import namedtuple
import osmnx as ox

# Internal
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *


'''
    - See number #1 below and integrate
    - See number #2 below and integrate
'''


oxts_pose_file_path = "/Users/donceykong/Desktop/kitti360Scripts/data/2013_05_28_drive_0005_sync_pose2oxts.txt"
xyz_point_clouds, xyz_positions = get_pointcloud_from_txt(oxts_pose_file_path) # Create point clouds from XYZ positions


ply_file_path = '/Users/donceykong/Desktop/kitti360Scripts/data/output3D.ply'
point_cloud_3D = o3d.io.read_point_cloud(ply_file_path)
points_3D = np.asarray(point_cloud_3D.points)
points_2D = points_3D.copy()
points_2D[:, 2] = 0
point_cloud_2D = o3d.geometry.PointCloud()
point_cloud_2D.points = o3d.utility.Vector3dVector(points_2D)


osm_file_path = '/Users/donceykong/Desktop/kitti360Scripts/data/map_0005.osm'
building_features = ox.features_from_xml(osm_file_path, tags={'building': True})
print(f"\nlen(buildings): {len(building_features)}")
threshold_dist = 0.0008 
building_list, building_line_set = get_buildings_near_poses(building_features, xyz_positions, threshold_dist)


num_points_per_edge = 100
discretize_all_building_edges(building_list, num_points_per_edge)


radius = 0.000008
# TODO: Maybe here would be a good point to do some sort of scan-matching so that the buildings and OSM-polygons are better aligned
calc_points_on_building_edges(building_list, point_cloud_3D, point_cloud_2D, radius)


hit_building_list, hit_building_line_set = get_building_hit_list(building_list)


def extract_and_save_building_points(new_pcd_3D, hit_building_list):
    # print("\n\n-   -   -   -   -   extract_and_save_points     -   -   -   -   -")
    new_pcd_2D = np.copy(np.asarray(new_pcd_3D.points))
    new_pcd_2D[:, 2] = 0

    # Create a new 2D point cloud from the modified points
    point_cloud_2D = o3d.geometry.PointCloud()
    point_cloud_2D.points = o3d.utility.Vector3dVector(new_pcd_2D)

    len_hit_building_list = len(hit_building_list)
    point_cloud_2D_kdtree = KDTree(np.asarray(point_cloud_2D.points))
    for iter, hit_building in enumerate(hit_building_list):
        iter += 1
        # print(f"    - Hit Building: {iter} / {len_hit_building_list}")
        masked_points_building = []
        for edge in hit_building.expanded_edges:
            distances, indices = point_cloud_2D_kdtree.query([edge])
            # Use a mask to filter 3D points that are within the XY radius from the edge point
            mask = abs(distances) <= radius
            masked_points = np.asarray(new_pcd_3D.points)[indices[mask]]
            masked_points_building.extend(masked_points)
            # Update building statistics based on the number of points within the radius

        if len(masked_points_building) > 0:
            hit_building.scan_num += 1
            # hit_building.points.extend(masked_points_building)

            # Save hit_building.points as .bin file

            # TODO: Inlcude frame number??????????????????????????????
            file_name = f"/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_3d_extracted/2013_05_28_drive_0005_sync/buildings/hitbuilding_{iter+1}_scan_{hit_building.scan_num}.bin"
            with open(file_name, 'wb') as bin_file:
                np.array(masked_points_building).tofile(bin_file)


# Create a dictionary for label colors
labels_dict = {label.id: label.color for label in labels}

poses_file = '/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/vel_poses.txt'
transformation_matrices = get_transform_matrices(poses_file)

frame_number = 30  # starting frame number
MAX_FRAME_NUMBER = 6000

while True:
    new_pcd = load_and_visualize(frame_number, transformation_matrices, labels_dict)
    
    if new_pcd:
        print(f"frame: {frame_number}")
        extract_and_save_building_points(new_pcd, hit_building_list)
    
    frame_number += 1  # Increment the frame number

    # Exit the loop if you've processed all frames
    if frame_number > MAX_FRAME_NUMBER:  # Define the maximum frame number you want to process
        break



'''
Number 1: Create Velodyne Poses

'''

# import os
# import numpy as np

# def read_poses(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     matrices = []
#     for line in lines:
#         elements = line.split()
#         matrix_3x4 = np.array(elements[1:], dtype=float).reshape((3, 4))
#         matrix_4x4 = np.vstack([matrix_3x4, np.array([0, 0, 0, 1])])
#         matrices.append(matrix_4x4)
#     transformation_matrices = np.stack(matrices)
#     return transformation_matrices

# def transform_imu_to_lidar(transformation_matrices, translation_vector, rotation_matrix):
#     # Create the transformation matrix from IMU to LiDAR
#     imu_to_lidar_matrix = np.identity(4)
#     imu_to_lidar_matrix[:3, :3] = rotation_matrix
#     imu_to_lidar_matrix[:3, 3] = translation_vector

#     # Apply the transformation to each pose
#     lidar_poses = np.matmul(transformation_matrices, imu_to_lidar_matrix)
#     return lidar_poses

# def write_poses(file_path, transformation_matrices, frame_indices):
#     with open(file_path, 'w') as file:
#         for idx, matrix in zip(frame_indices, transformation_matrices):
#             # Flatten the matrix to a 1D array, convert to strings, and join with spaces
#             matrix_string = ' '.join(map(str, matrix.flatten()))
#             # Write the frame index followed by the flattened matrix
#             file.write(f"{idx} {matrix_string}\n")

# # Define the translation vector from IMU to LiDAR
# translation_vector = np.array([0.81, 0.32, -0.83])

# # Define the rotation matrix for a 180-degree rotation about the X-axis
# rotation_matrix = np.array([[1, 0, 0],
#                             [0, -1, 0],
#                             [0, 0, -1]])

# imu_poses_file = 'kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/poses.txt'
# vel_poses_file = 'kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/vel_poses.txt'

# # Read the IMU poses
# transformation_matrices = read_poses(imu_poses_file)

# # Extract frame indices (assuming they are the first element in each line)
# frame_indices = []
# with open(imu_poses_file, 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         frame_indices.append(line.split()[0])

# # Transform IMU poses to LiDAR poses
# lidar_poses = transform_imu_to_lidar(transformation_matrices, translation_vector, rotation_matrix)

# # Write the LiDAR poses to file
# write_poses(vel_poses_file, lidar_poses, frame_indices)




'''
Number 2: Save "building" and "unlabeled" points

'''

# import os
# import numpy as np
# import open3d as o3d
# from open3d.io import write_point_cloud

# # Internal
# from tools.labels import labels
# from tools.convert_oxts_pose import *

# '''
# view_frame_semantics.py
# '''
# def read_label_bin_file(file_path):
#     """
#     Reads a .bin file containing point cloud labels.
#     """
#     labels = np.fromfile(file_path, dtype=np.int16)
#     return labels.reshape(-1)

# def read_bin_file(file_path):
#     """
#     Reads a .bin file containing point cloud data.
#     Assumes each point is represented by four float32 values (x, y, z, intensity).
#     """
#     point_cloud = np.fromfile(file_path, dtype=np.float32)
#     return point_cloud.reshape(-1, 4)

# # Create a dictionary for label colors
# labels_dict = {label.id: label.color for label in labels}

# def color_point_cloud(points, labels):
#     """
#     Colors the point cloud based on the labels.
#     Each point in the point cloud is assigned the color of its label.
#     """
#     colored_points = np.zeros_like(points[:, :3])  # Initialize with zeros, only xyz
#     for i, label in enumerate(labels):
#         if np.isnan(label):
#             continue  # Skip NaN labels
#         if label == -1:
#             continue  # Skip invalid labels

#         color = labels_dict.get(label, (0, 0, 0))  # Default color is black

#         colored_points[i] = np.array(color) / 255.0  # Normalize to [0, 1] for Open3D
#     return colored_points

# def read_poses(file_path):
#     transformation_matrices = {}

#     with open(file_path, 'r') as file:
#         for line in file:
#             # Split the line into individual elements
#             elements = line.strip().split()
#             frame_index = int(elements[0])  # Frame index is the first element

#             # Check if the line contains 16 elements for a 4x4 matrix
#             if len(elements[1:]) == 16:
#                 # Convert elements to float and reshape into 4x4 matrix
#                 matrix_4x4 = np.array(elements[1:], dtype=float).reshape((4, 4))
#             else:
#                 # Otherwise, assume it is a 3x4 matrix and append a homogeneous row
#                 matrix_3x4 = np.array(elements[1:], dtype=float).reshape((3, 4))
#                 matrix_4x4 = np.vstack([matrix_3x4, np.array([0, 0, 0, 1])])

#             # Store the matrix using the frame index as the key
#             transformation_matrices[frame_index] = matrix_4x4

#     return transformation_matrices

# def transform_point_cloud(pc, transformation_matrices, frame_number):
#     if frame_number >= len(transformation_matrices):
#         print(f"Frame number {frame_number} is out of range.")
#         return None

#     # Get the transformation matrix for the current frame
#     transformation_matrix = transformation_matrices.get(frame_number)

#     # Separate the XYZ coordinates and intensity values
#     xyz = pc[:, :3]
#     intensity = pc[:, 3].reshape(-1, 1)

#     # Convert the XYZ coordinates to homogeneous coordinates
#     xyz_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])

#     # Apply the transformation to each XYZ coordinate
#     transformed_xyz = np.dot(xyz_homogeneous, transformation_matrix.T)[:, :3]

#     return transformed_xyz

# poses_file = '/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/vel_poses.txt'
# transformation_matrices = read_poses(poses_file)
# def load_and_visualize(frame_number, last_min):
#     # Adjust file paths based on frame number
#     pc_filepath = f'/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_3d_raw/2013_05_28_drive_0005_sync/velodyne_points/data/{frame_number:010d}.bin'
#     label_filepath = f'/Users/donceykong/Desktop/kitti360Scripts/data/recovered/2013_05_28_drive_0005_sync/labels/{frame_number:010d}.bin'
    
#     if not os.path.exists(pc_filepath) or not os.path.exists(label_filepath):
#         print(f"File not found for frame number {frame_number}")
#         return None, None

#     # read pointcloud bin files and label bin files
#     pc = read_bin_file(pc_filepath)
#     pc = transform_point_cloud(pc, transformation_matrices, frame_number)
#     labels_np = read_label_bin_file(label_filepath)

#     # boolean mask where True represents the labels to keep
#     label_mask = (labels_np == 11) | (labels_np == 0)

#     # mask to filter the point cloud and labels
#     pc = pc[label_mask]
#     labels_np = labels_np[label_mask]

#     # color the point cloud
#     colored_points = color_point_cloud(pc, labels_np)
#     colored_pcd = o3d.geometry.PointCloud()
    
#     # Reshape pointcloud to fit in convertPoseToOxts function
#     pc_reshaped = np.array([np.eye(4) for _ in range(pc.shape[0])])
#     pc_reshaped[:, 0:3, 3] = pc[:, :3]

#     # Convert to lat-lon-alt
#     pc_reshaped = np.asarray(postprocessPoses(pc_reshaped))
#     pc_lla = np.asarray(convertPoseToOxts(pc_reshaped))

#     min_alt = 226.60675 #(np.min(pc_lla[:, 2]) + last_min)/2
#     pc_lla[:, 2] = (pc_lla[:, 2] - min_alt)*0.00001

#     colored_pcd.points = o3d.utility.Vector3dVector(pc_lla[:, :3])  # Only use lat, lon, alt for geometry
#     colored_pcd.colors = o3d.utility.Vector3dVector(colored_points) # Set colors

#     return colored_pcd, min_alt

# def load_xyz_positions(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     xyz_positions = []
#     for line in lines:
#         matrix_elements = np.array(line.split(), dtype=float)
#         x, y = matrix_elements[0], matrix_elements[1]
#         xyz_positions.append([x, y, 0])
#     return xyz_positions

# def create_point_clouds_from_xyz(xyz_positions):
#     point_clouds = []
#     for xyz in xyz_positions:
#         # Create a point cloud with a single point
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector([xyz])
#         # pcd.paint_uniform_color([1, 0, 0])  # Red color for poses
#         point_clouds.append(pcd)
#     return point_clouds

# file_path = "/Users/donceykong/Desktop/kitti360Scripts/data/2013_05_28_drive_0005_sync_pose2oxts.txt"
# xyz_positions = load_xyz_positions(file_path)

# # Create point clouds from XYZ positions
# xyz_point_clouds = create_point_clouds_from_xyz(xyz_positions)

# # List to hold all point cloud geometries
# pcd_geometries = []

# # Iterate through frame numbers and load each point cloud
# frame_num = 30  # Initial frame number
# last_min = 0
# total_labels = 6255
# while frame_num <= total_labels:
#     # print(f"frame_num: {frame_num}")
#     pcd, new_min = load_and_visualize(frame_num, last_min)
#     frame_num += 1
#     if pcd is not None:
#         last_min = new_min
#         # frame_num += 5
#         # voxel_size = 0.0000001  # example voxel size
#         # pcd_ds = pcd.voxel_down_sample(voxel_size)
#         pcd_geometries.append(pcd)

# # Merge all point clouds in pcd_geometries into a single point cloud
# merged_pcd = o3d.geometry.PointCloud()
# for pcd in pcd_geometries:
#     merged_pcd += pcd

# # Save the merged point cloud to a PLY file
# output_file_path = '/Users/donceykong/Desktop/kitti360Scripts/data/output3D.ply'  # Specify your output file path here
# o3d.io.write_point_cloud(output_file_path, merged_pcd)

# print(f"Saved merged point cloud to {output_file_path}")