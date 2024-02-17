'''
By: Doncey Albin


Refactoring of kitti360scripts and recoverKitti repositories was made in order to create this pipeline.
I couldn't have done it without them.
    - kitti360scripts:
    - recoverKitti:


Genral utilities for running code.

'''
import os
import numpy as np
import open3d as o3d
import osmnx as ox
from sklearn.neighbors import KDTree

# Internal
import tools.osm_building as osm_building
from tools.convert_oxts_pose import *

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

def get_pointcloud_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    xyz_positions = []
    for line in lines:
        matrix_elements = np.array(line.split(), dtype=float)
        x, y = matrix_elements[0], matrix_elements[1]
        xyz_positions.append([x, y, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_positions)
    return pcd, xyz_positions

def create_circle(center, radius, num_points=30):
    """
    Create a circle at a given center point.

    :param center: Center of the circle (x, y, z).
    :param radius: Radius of the circle.
    :param num_points: Number of points to approximate the circle.
    :return: Open3D point cloud representing the circle.
    """
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = 0 #center[2]
        points.append([x, y, z])
    
    circle_pcd = o3d.geometry.PointCloud()
    circle_pcd.points = o3d.utility.Vector3dVector(points)
    return circle_pcd












def visualize_point_cloud(points):
    """
    Visualizes the point cloud data using Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Using only x, y, z for point cloud
    return pcd

def discretize_edge(start_point, end_point, num_points):
    """
    Generate linearly spaced points between start_point and end_point.
    """
    return np.linspace(start_point, end_point, num_points, axis=0)

def is_point_near_edge(point_cloud, edge_points, threshold):
    """
    Check if any point in point_cloud is within 'threshold' distance of any point in edge_points.
    """
    for point in edge_points:
        # Compute distances from the current edge point to all points in the point cloud
        distances = np.linalg.norm(np.asarray(point_cloud.points) - point, axis=1)
        if np.any(distances < threshold):
            print("     point is near")
            return True
    return False

def check_building_proximity(building_coords, point_cloud, threshold):
    """
    Check if any point of the building is near the point cloud.
    """
    for coord in building_coords:
        if is_point_near_edge(point_cloud, coord, threshold):
            return True
    return False

min_vert_list = []
def building_within_bounds(building_vertex, xyz_positions, threshold):
    vert_dist_arr = []
    building_vertex = np.array(building_vertex)
    # print(f"building_vertex[0]: {building_vertex[0]}")
    for pos in xyz_positions:
        # pos[0] = pos[1]
        # pos[1] = pos[0]
        vert_dist = np.sqrt((pos[1] - building_vertex[0])*(pos[1] - building_vertex[0])+(pos[0] - building_vertex[1])*(pos[0] - building_vertex[1]))
        # print(f"pos: {pos[:2]}")
        # print(f"building_vertex: {building_vertex}")
        # vert_dist = np.linalg.norm(building_vertex - pos[:2])
        # print(f"vert dist: {vert_dist}")
        vert_dist_arr.append(vert_dist)
    min_vert_dist = np.min(vert_dist_arr)
    min_vert_list.append(min_vert_dist)
    # print(f"min vert dist: {min_vert_dist}")
    return min_vert_dist <= threshold

def get_all_buildings(osm_file_path):
    buildings = ox.geometries_from_xml(osm_file_path, tags={'building': True})
    # Process Buildings as LineSets
    building_lines = []
    for _, building in buildings.iterrows():
        if building.geometry.type == 'Polygon':
            exterior_coords = building.geometry.exterior.coords
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                end_point = [exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0]
                building_lines.append([start_point, end_point])

    building_line_set = o3d.geometry.LineSet()
    building_points = [point for line in building_lines for point in line]
    building_lines_idx = [[i, i + 1] for i in range(0, len(building_points), 2)]
    building_line_set.points = o3d.utility.Vector3dVector(building_points)
    building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
    building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings
    return building_lines, building_line_set

def get_buildings_near_poses(osm_file_path, xyz_positions, threshold_dist):
    building_features = ox.features_from_xml(osm_file_path, tags={'building': True})
    building_list = []
    building_lines = []
    for _, building in building_features.iterrows():
        if building.geometry.geom_type == 'Polygon':
            exterior_coords = building.geometry.exterior.coords
            # Check if first building vertex is within path
            if True:#building_within_bounds(exterior_coords[0], xyz_positions, threshold_dist): 
                per_building_lines = []
                for i in range(len(exterior_coords) - 1):
                    start_point = [exterior_coords[i][1], exterior_coords[i][0], 0]
                    end_point = [exterior_coords[i + 1][1], exterior_coords[i + 1][0], 0]
                    per_building_lines.append([start_point, end_point])
                    building_lines.append([start_point, end_point])
                new_building = osm_building.OSMBuilding(per_building_lines)
                building_list.append(new_building)

    building_line_set = o3d.geometry.LineSet()
    building_points = [point for line in building_lines for point in line]
    building_lines_idx = [[i, i + 1] for i in range(0, len(building_points), 2)]
    building_line_set.points = o3d.utility.Vector3dVector(building_points)
    building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
    building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings

    return building_list, building_line_set

def discretize_all_building_edges(building_list, num_points_per_edge):
    for building in building_list:
        for edge in building.edges:
            edge_points = np.linspace(edge[0], edge[1], num_points_per_edge, axis=0)
            for edge_point in edge_points:
                building.expanded_edges.append(edge_point)
    
def get_building_edge_bounds(building_list, radius=0.000008):
    all_edge_circles = o3d.geometry.PointCloud()
    for building in building_list:
            edge_points = building.expanded_edges
            edge_circles = [create_circle(edge_point, radius) for edge_point in edge_points]  # Adjust radius as needed
            for edge_circle in edge_circles:
                all_edge_circles += edge_circle

    return all_edge_circles

def calc_points_on_building_edges(building_list, point_cloud_3D, point_cloud_2D, label_filepath, radius):
    # Filter buildings only hit by "building" points here
    # labels_np = read_label_bin_file(label_filepath)
    # label_mask = (labels_np == 11) | (labels_np == 0)
    # pc = np.asarray(point_cloud_3D.points)
    # pc = pc[label_mask]
    # labels_np = labels_np[label_mask]

    len_building_list = len(building_list)
    point_cloud_2D_kdtree = KDTree(np.asarray(point_cloud_2D.points))
    for iter, building in enumerate(building_list):
        iter += 1
        print(f"        --> Building: {iter} / {len_building_list}")
        for edge in building.expanded_edges:
            distances, indices = point_cloud_2D_kdtree.query([edge])
            
            # Use a mask to filter 3D points that are within the XY radius from the edge point
            mask = abs(distances) <= radius
            masked_points = np.asarray(point_cloud_3D.points)[indices[mask]]
            
            # Update building statistics based on the number of points within the radius
            building.times_hit += len(masked_points)
            building.accum_points.extend(masked_points)
                
def get_building_hit_list(building_list, min_scans): 
    hit_building_list = []
    for build_iter, building in enumerate(building_list):
        if building.times_hit < min_scans:
            building.times_hit = 0  # reset
            hit_building_list.append(building)

    hit_building_line_set = o3d.geometry.LineSet()
    hit_building_points = [point for building in hit_building_list for line in building.edges for point in line]
    hit_building_lines_idx = []
    hit_building_lines_idx = [[i, i + 1] for i in range(0, len(hit_building_points) - 1, 2)]
    hit_building_line_set.points = o3d.utility.Vector3dVector(hit_building_points)
    hit_building_line_set.lines = o3d.utility.Vector2iVector(hit_building_lines_idx)
    hit_building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings

    return hit_building_list, hit_building_line_set








'''
view_frame_semantics2.py
'''

def color_point_cloud(points, labels, labels_dict):
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

        # if (pc[i, 0] < 0 and color != (0,0,0)):
        #     print(f"{pc[i, :3]} iter: {i}, color: {color}")
        #     color = (255, 0, 0)

        colored_points[i] = np.array(color) / 255.0  # Normalize to [0, 1] for Open3D
    return colored_points

def get_transform_matrices(file_path):
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

def get_transformed_point_cloud(pc, transformation_matrices, frame_number):
    # if frame_number >= len(transformation_matrices):
    #     print(f"Frame number {frame_number} is out of range.")
    #     return None

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

def load_and_visualize(pc_filepath, label_filepath, velodyne_poses, frame_number, labels_dict):
    if not os.path.exists(pc_filepath) or not os.path.exists(label_filepath):
        print(f"        --> File not found for frame number {frame_number}!")
        return None

    # read pointcloud bin files and label bin files
    pc = read_bin_file(pc_filepath)
    pc = get_transformed_point_cloud(pc, velodyne_poses, frame_number)
    labels_np = read_label_bin_file(label_filepath)

    # # boolean mask where True represents the labels to keep
    # label_mask = (labels_np == 11) | (labels_np == 0)

    # # mask to filter the point cloud and labels
    # pc = pc[label_mask]
    # labels_np = labels_np[label_mask]

    # color the point cloud
    colored_points = color_point_cloud(pc, labels_np, labels_dict)
    colored_pcd = o3d.geometry.PointCloud()
    
    # Reshape pointcloud to fit in convertPointsToOxts function
    pc_reshaped = np.array([np.eye(4) for _ in range(pc.shape[0])])
    pc_reshaped[:, 0:3, 3] = pc[:, :3]

    # Convert to lat-lon-alt
    pc_reshaped = np.asarray(postprocessPoses(pc_reshaped))
    pc_lla = np.asarray(convertPointsToOxts(pc_reshaped))

    # ave_alt = 226.60675 # Average altitude
    pc_lla[:, 2] *= 0.00002 #(pc_lla[:, 2] - ave_alt)*0.00001

    colored_pcd.points = o3d.utility.Vector3dVector(pc_lla[:, :3])  # Only use lat, lon, alt for geometry
    colored_pcd.colors = o3d.utility.Vector3dVector(colored_points) # Set colors

    return colored_pcd

# # TODO: Remove and replace with above
# def load_and_visualize_OG(frame_number, transformation_matrices, labels_dict):
#     # Adjust file paths based on frame number
#     pc_filepath = f'/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_3d_raw/2013_05_28_drive_0005_sync/velodyne_points/data/{frame_number:010d}.bin'
#     label_filepath = f'/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_3d_semantics/train/2013_05_28_drive_0005_sync/labels/{frame_number:010d}.bin'
    
#     if not os.path.exists(pc_filepath) or not os.path.exists(label_filepath):
#         print(f"File not found for frame number {frame_number}")
#         return None

#     # read pointcloud bin files and label bin files
#     pc = read_bin_file(pc_filepath)
#     pc = get_transformed_point_cloud(pc, transformation_matrices, frame_number)
#     labels_np = read_label_bin_file(label_filepath)

#     # boolean mask where True represents the labels to keep
#     label_mask = (labels_np == 11) | (labels_np == 0)

#     # mask to filter the point cloud and labels
#     # pc = pc[label_mask]
#     # labels_np = labels_np[label_mask]

#     # color the point cloud
#     colored_points = color_point_cloud(pc, labels_np, labels_dict)
#     colored_pcd = o3d.geometry.PointCloud()
    
#     # Reshape pointcloud to fit in convertPointsToOxts function
#     pc_reshaped = np.array([np.eye(4) for _ in range(pc.shape[0])])
#     pc_reshaped[:, 0:3, 3] = pc[:, :3]

#     # Convert to lat-lon-alt
#     pc_reshaped = np.asarray(postprocessPoses(pc_reshaped))
#     pc_lla = np.asarray(convertPointsToOxts(pc_reshaped))

#     ave_alt = 226.60675 # Average altitude
#     pc_lla[:, 2] = (pc_lla[:, 2] - ave_alt)*0.00001

#     colored_pcd.points = o3d.utility.Vector3dVector(pc_lla[:, :3])  # Only use lat, lon, alt for geometry
#     colored_pcd.colors = o3d.utility.Vector3dVector(colored_points) # Set colors

#     return colored_pcd

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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([xyz])
        point_clouds.append(pcd)
    return point_clouds







# From create_velodyne_poses.py

def read_poses(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrices = []
    for line in lines:
        elements = line.split()
        matrix_3x4 = np.array(elements[1:], dtype=float).reshape((3, 4))
        matrix_4x4 = np.vstack([matrix_3x4, np.array([0, 0, 0, 1])])
        matrices.append(matrix_4x4)
    transformation_matrices = np.stack(matrices)
    return transformation_matrices

def loadPoses (pos_file):
  ''' load system poses '''
  data = np.loadtxt(pos_file)
  ts = data[:, 0].astype(np.int64)
  poses = np.reshape(data[:, 1:], (-1, 3, 4))
  poses = np.concatenate((poses, np.tile(np.array([0, 0, 0, 1]).reshape(1,1,4),(poses.shape[0],1,1))), 1)
  return ts, poses

def transform_imu_to_lidar(transformation_matrices, translation_vector, rotation_matrix):
    # Create the transformation matrix from IMU to LiDAR
    imu_to_lidar_matrix = np.identity(4)
    imu_to_lidar_matrix[:3, :3] = rotation_matrix
    imu_to_lidar_matrix[:3, 3] = translation_vector

    # Apply the transformation to each pose
    lidar_poses = np.matmul(transformation_matrices, imu_to_lidar_matrix)
    return lidar_poses

def write_poses(file_path, transformation_matrices, frame_indices):
    with open(file_path, 'w') as file:
        for idx, matrix in zip(frame_indices, transformation_matrices):
            # Flatten the matrix to a 1D array, convert to strings, and join with spaces
            matrix_string = ' '.join(map(str, matrix.flatten()))
            # Write the frame index followed by the flattened matrix
            file.write(f"{idx} {matrix_string}\n")




def get_trans_poses_from_imu_to_velodyne(imu_poses_file, vel_poses_file, save_to_file=False):
    # Define the translation vector from IMU to LiDAR
    translation_vector = np.array([0.81, 0.32, -0.83])

    # Define the rotation matrix for a 180-degree rotation about the X-axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])

    # Read the IMU poses
    transformation_matrices = read_poses(imu_poses_file)

    # Extract frame indices (assuming they are the first element in each line)
    frame_indices = []
    with open(imu_poses_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            frame_indices.append(line.split()[0])

    # Transform IMU poses to LiDAR poses
    lidar_poses = transform_imu_to_lidar(transformation_matrices, translation_vector, rotation_matrix)

    # Save velodyne poses to file
    if (save_to_file):
        write_poses(vel_poses_file, lidar_poses, frame_indices)

    return lidar_poses


def get_accum_colored_pc(init_frame, fin_frame, inc_frame, raw_pc_path, label_path, velodyne_poses, labels_dict, accum_ply_path):
    # List to hold all point cloud geometries
    pcd_geometries = []

    # Iterate through frame numbers and load each point cloud
    frame_num = init_frame         # TODO: Retrieve initial frame number of label
    while frame_num <= fin_frame:
        raw_pc_frame_path = os.path.join(raw_pc_path, f'{frame_num:010d}.bin')
        pc_frame_label_path = os.path.join(label_path, f'{frame_num:010d}.bin')

        # print(f"frame_num: {frame_num}")
        pcd = load_and_visualize(raw_pc_frame_path, pc_frame_label_path, velodyne_poses, frame_num, labels_dict)
        if pcd is not None:
            # last_min = new_min    # TODO: remove?
            # voxel_size = 0.0000001  # example voxel size
            # pcd_ds = pcd.voxel_down_sample(voxel_size)
            pcd_geometries.append(pcd)
        frame_num += inc_frame

    # Merge all point clouds in pcd_geometries into a single point cloud
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in pcd_geometries:
        merged_pcd += pcd

    # Save the merged point cloud to a PLY file
    o3d.io.write_point_cloud(accum_ply_path, merged_pcd)
    print(f"        --> Saved merged point cloud to {accum_ply_path}")
        
    return merged_pcd

def read_vel_poses(file_path):
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