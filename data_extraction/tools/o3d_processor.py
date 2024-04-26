

import os
import open3d as o3d
import numpy as np
from tqdm import tqdm

# Internal
from tools.file_processor import *
from tools.convert_oxts_pose import *

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

def get_circle_pcd(center, radius, num_points=30):
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

def building_list_to_o3d_lineset(building_list):
    building_line_set = o3d.geometry.LineSet()
    building_points = [point for building in building_list for edge in building.edges for point in edge.edge_vertices]
    building_lines_idx = [[i, i + 1] for i in range(0, len(building_points) - 1, 2)]
    building_line_set.points = o3d.utility.Vector3dVector(building_points)
    building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
    building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings
    return building_line_set

def vertex_list_to_o3d_lineset(vertices):
    try:
        building_line_set = o3d.geometry.LineSet()
        building_points = vertices
        building_lines_idx = [[i, (i + 1) % len(building_points)] for i in range(len(building_points))]
        building_line_set.points = o3d.utility.Vector3dVector(building_points)
        building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
        building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings
        return building_line_set
    except Exception as e:
        print("Error:", e)
        print("Vertices:", vertices)
        raise

def vis_total_accum_points(build_list):
    build_total_accum_points_frame = []
    for build in build_list:
        if len(build.get_total_accum_obs_points()) == 0:
            continue
        else:
            total_accum_obs_points = build.get_total_accum_obs_points() #np.asarray(build.get_total_accum_obs_points()).reshape(-1, 3))
            total_accum_obs_points[:,2] = total_accum_obs_points[:,2] - np.min(total_accum_obs_points[:,2])
            build_total_accum_points_frame.extend(total_accum_obs_points) 

    # build_total_accum_points_frame = np.asarray(build_total_accum_points_frame)
    # build_total_accum_points_frame[:,2] = build_total_accum_points_frame[:,2] - np.mean(build_total_accum_points_frame[:,2])

    build_line_set = building_list_to_o3d_lineset(build_list)
    build_accum_points_pcd = o3d.geometry.PointCloud()
    print(f"points length: {len(build_total_accum_points_frame)}")
    build_accum_points_pcd.points = o3d.utility.Vector3dVector(build_total_accum_points_frame)
    o3d.visualization.draw_geometries([build_line_set, build_accum_points_pcd])


def create_point_clouds_from_xyz(xyz_positions):
    point_clouds = []
    for xyz in xyz_positions:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([xyz])
        point_clouds.append(pcd)
    return point_clouds

def load_and_visualize(raw_pc_path, label_path, velodyne_poses, frame_num, labels_dict):
    raw_pc_frame_path = os.path.join(raw_pc_path, f'{frame_num:010d}.bin')
    pc_frame_label_path = os.path.join(label_path, f'{frame_num:010d}.bin')

    if not os.path.exists(raw_pc_frame_path) or not os.path.exists(pc_frame_label_path):
        # print(f"        --> File not found for frame number {frame_num}!")
        return None

    # print(f"        --> File exists for frame number {frame_num}!")
    # read pointcloud bin files and label bin files
    pc = read_bin_file(raw_pc_frame_path)
    pc = get_transformed_point_cloud(pc, velodyne_poses, frame_num)
    labels_np = read_label_bin_file(pc_frame_label_path)

    # boolean mask where True represents the labels to keep
    label_mask = (labels_np == 11) | (labels_np == 0)
    building_label_mask = (labels_np == 11)

    # find min point labeled as "building"
    pc_buildings = pc[building_label_mask]
    if (len(pc_buildings) > 0):
        min_building_z_point = np.min(pc_buildings[:, 2])
    else: 
        return None
    
    # mask to filter the point cloud and labels
    pc = pc[label_mask] # TODO: also remove any points with a z-position below min_building_z_point
    labels_np = labels_np[label_mask]

    # Also remove any points with a z-position below min_building_z_point
    z_position_mask = pc[:, 2] >= min_building_z_point
    pc = pc[z_position_mask]
    labels_np = labels_np[z_position_mask]

    # color the point cloud
    colored_points = color_point_cloud(pc, labels_np, labels_dict)
    colored_pcd = o3d.geometry.PointCloud()
    
    # Reshape pointcloud to fit in convertPointsToOxts function
    pc_reshaped = np.array([np.eye(4) for _ in range(pc.shape[0])])
    pc_reshaped[:, 0:3, 3] = pc[:, :3]

    # Convert to lat-lon-alt
    pc_reshaped = np.asarray(postprocessPoses(pc_reshaped))
    pc_lla = np.asarray(convertPointsToOxts(pc_reshaped))
    pc_lla[:, 2] *= 0.00002 # TODO: Remove this and only use for visualization

    colored_pcd.points = o3d.utility.Vector3dVector(pc_lla[:, :3])  # Only use lat, lon, alt for geometry
    colored_pcd.colors = o3d.utility.Vector3dVector(colored_points) # Set colors

    return colored_pcd

def get_accum_pc(init_frame, fin_frame, inc_frame, raw_pc_path, label_path, velodyne_poses, labels_dict, accum_ply_path):
    # List to hold all point cloud o3d-based pcds
    pcd_geometries = []

    # Initialize tqdm progress bar
    num_frames = len(range(init_frame, fin_frame + 1, inc_frame))
    progress_bar = tqdm(total=num_frames, desc="            ")

    # Iterate through frame numbers and load each point cloud
    for frame_num in range(init_frame, fin_frame + 1, inc_frame):
        pcd = load_and_visualize(raw_pc_path, label_path, velodyne_poses, frame_num, labels_dict)
        if pcd is not None:
            pcd_geometries.append(pcd)
        
        # Update progress bar
        progress_bar.update(1)
    
    # Merge all point clouds in pcd_geometries into a single point cloud
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in pcd_geometries:
        merged_pcd += pcd

    if len(merged_pcd.points):
        # Save the merged point cloud to a PLY file
        o3d.io.write_point_cloud(accum_ply_path, merged_pcd)
        print(f"\n        --> Saved merged point cloud to {accum_ply_path}")
        
    return merged_pcd

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