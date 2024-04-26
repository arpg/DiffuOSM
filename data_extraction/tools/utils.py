'''
Doncey Albin

'''

import os
import glob
import numpy as np
import open3d as o3d
import osmnx as ox
from sklearn.neighbors import KDTree as sklearnKDTree
from scipy.spatial import KDTree as scipyKDTree
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
import pickle
from tqdm import tqdm

import time
from contextlib import contextmanager

# Internal
import tools.osm_building as osm_building
from tools.convert_oxts_pose import *

min_vert_list = []
def building_within_bounds(building_vertex, velodyne_poses_latlon, threshold):
    vert_dist_arr = []
    building_vertex = np.array(building_vertex)

    for pos in velodyne_poses_latlon.values():
        pos = pos[:3]       # Only use lat-lon-alt
        # pos[0] = pos[1]
        # pos[1] = pos[0]
        vert_dist = np.sqrt((pos[1] - building_vertex[0])*(pos[1] - building_vertex[0])+(pos[0] - building_vertex[1])*(pos[0] - building_vertex[1]))
        vert_dist_arr.append(vert_dist)
    min_vert_dist = np.min(vert_dist_arr)
    min_vert_list.append(min_vert_dist)

    return min_vert_dist <= threshold

def find_file_max_value(dir_path):
    pattern = os.path.join(dir_path, '*.msgpack')
    files = glob.glob(pattern)

    max_frame_values = []
    for file in files:
        try:
            filename = os.path.basename(file)
            components = filename.split('_')
            max_frame = int(components[3])      # 'maxframe' value is the fourth component in the filename
            max_frame_values.append(max_frame)
        except (IndexError, ValueError) as e:
            print(f"Skipping file {filename} due to error: {e}")
    
    if not max_frame_values:
        return 0
    
    # Find the maximum value
    max_frame_max_value = max(max_frame_values)

    return max_frame_max_value
    
def get_all_osm_buildings(osm_file_path):
    building_features = ox.features_from_xml(osm_file_path, tags={'building': True})
    building_list = []
    building_lines = []
    building_id = 0
    for _, building in building_features.iterrows():
        if building.geometry.geom_type == 'Polygon':
            exterior_coords = building.geometry.exterior.coords
            per_building_lines = []
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][1], exterior_coords[i][0], 0]
                end_point = [exterior_coords[i + 1][1], exterior_coords[i + 1][0], 0]
                per_building_lines.append([start_point, end_point])
                building_lines.append([start_point, end_point])
            new_building = osm_building.OSMBuilding(per_building_lines, building_id)
            building_list.append(new_building)
            building_id += 1
    return building_list

def get_buildings_near_poses(osm_file_path, velodyne_poses_latlon, threshold_dist):
    building_features = ox.features_from_xml(osm_file_path, tags={'building': True})
    building_list = []
    building_lines = []
    building_id = 0
    for _, building in building_features.iterrows():
        if building.geometry.geom_type == 'Polygon':
            exterior_coords = building.geometry.exterior.coords
            # Check if first building vertex is within path
            if building_within_bounds(exterior_coords[0], velodyne_poses_latlon, threshold_dist): 
                per_building_lines = []
                for i in range(len(exterior_coords) - 1):
                    start_point = [exterior_coords[i][1], exterior_coords[i][0], 0]
                    end_point = [exterior_coords[i + 1][1], exterior_coords[i + 1][0], 0]
                    per_building_lines.append([start_point, end_point])
                    # per_building_lines.append(convertOSMToPose([start_point, end_point]))
                    building_lines.append([start_point, end_point])
                new_building = osm_building.OSMBuilding(per_building_lines, building_id)
                building_list.append(new_building)
                building_id += 1

    building_line_set = o3d.geometry.LineSet()
    building_points = [point for line in building_lines for point in line]
    building_lines_idx = [[i, i + 1] for i in range(0, len(building_points), 2)]
    building_line_set.points = o3d.utility.Vector3dVector(building_points)
    building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
    building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings

    return building_list

def building_offset_to_o3d_lineset(building_list):
    building_line_set = o3d.geometry.LineSet()
    building_points = [point for building in building_list for edge in building.edges for point in edge.edge_vertices]
    building_lines_idx = [[i, i + 1] for i in range(0, len(building_points) - 1, 2)]
    building_line_set.points = o3d.utility.Vector3dVector(building_points)
    building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
    building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings
    return building_line_set

@contextmanager
def time_block(method_name):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration_seconds = end_time - start_time
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = duration_seconds % 60
        print(f"{method_name} took {hours}:{minutes:02d}:{seconds:.2f}")
        
def calc_points_within_build_poly(frame_num, building_list, point_cloud_3D, pos_latlong, near_path_threshold):
    # Get 2D representation of accumulated_color_pc
    points_2D = np.asarray(np.copy(point_cloud_3D.points))
    points_2D[:, 2] = 0

    # Preprocess building offset vertices to Shapely Polygon objects
    building_polygons = [Polygon(building.offset_vertices) for building in building_list]
    point_cloud_2D_kdtree = cKDTree(points_2D) # cKDTree
    points_3d = np.asarray(point_cloud_3D.points)
    
    for building, building_polygon in zip(building_list, building_polygons):
            distance = np.linalg.norm(pos_latlong[:2] - building.center[:2])
            if distance <= near_path_threshold:
                # Filter points within a threshold distance of the building center using KDTree
                indices = point_cloud_2D_kdtree.query_ball_point(building.center, building.max_dist_vertex_from_center)
                
                # Convert indices to numpy array
                indices = np.array(indices)

                # Filter points within the polygon
                points_within_polygon = [
                    points_3d[idx]
                    for idx in indices
                    if building_polygon.contains(Point(points_3d[idx, :2]))
                ]
                
                if len(points_within_polygon) > 0:
                    building.set_curr_obs_points(frame_num, points_within_polygon)

def get_building_hit_list(building_list, min_num_points): 
    hit_building_list = []

    for building in building_list:
        if len(building.get_total_accum_obs_points()) >= min_num_points:
            hit_building_list.append(building)

    return hit_building_list

# Remove?
# def update_per_frame_data(hit_building, building_edges_frame, observed_points_frame, curr_accum_points_frame):
#     """
#     """
    
#     building_edges_frame.extend(edge.edge_vertices for edge in hit_building.edges)
#     observed_points_frame.extend(hit_building.per_scan_obs_points)
#     curr_accum_points_frame.extend(hit_building.curr_accum_obs_points)






'''
Create:

File_processor.py

'''

def save_pkl_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_building_hit_list_from_files(building_edgeaccum_dir):
    hit_building_list = []
    hit_build_edges_list = []
    max_build_index = get_max_build_index(building_edgeaccum_dir)

    for build_index in range(max_build_index-1):
        build_index += 1
        if build_index > max_build_index-1:
            break

        per_build_edges_file = os.path.join(building_edgeaccum_dir, f'build_{build_index}_edges.bin')
        per_building_edges = read_building_edges_file(per_build_edges_file)
        new_building = osm_building.OSMBuilding(per_building_edges)
        hit_build_edges_list.append(per_building_edges.reshape(-1, 3))

        per_build_accum_file = os.path.join(building_edgeaccum_dir, f'build_{build_index}_accum.bin')
        per_building_accum_points = read_building_pc_file(per_build_accum_file)
        new_building.accum_points = per_building_accum_points

        hit_building_list.append(new_building)
    
    return hit_building_list

def get_max_build_index(edges_accum_dir):
    build_index = 1
    while True:
        build_pc_file = os.path.join(edges_accum_dir, f'build_{build_index}_accum.bin')
        if os.path.exists(build_pc_file):
            build_index += 1
        else:
            break
    return build_index

def read_building_edges_file(building_edges_file):
    with open(building_edges_file, 'rb') as bin_file:
        edges_array = np.fromfile(bin_file, dtype=float).reshape(-1, 2, 3)  # Reshape to 3D array
    return edges_array

def read_building_pc_file(file_path):
    point_cloud = np.fromfile(file_path)
    return point_cloud.reshape(-1, 3)
    
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

def convert_and_save_oxts_poses(imu_poses_file, output_path):
    """
    """
    [timestamps, poses] = loadPoses(imu_poses_file)
    poses = postprocessPoses(poses)
    oxts = convertPoseToOxts(poses)  # Convert to lat/lon coordinate
    with open(output_path, 'w') as f:
        for oxts_ in oxts:
            oxts_line = ' '.join(['%.6f' % x for x in oxts_])
            f.write(f'{oxts_line}\n')

def save_per_scan_data(extracted_per_frame_dir, frame_num, building_edges_frame, curr_accum_points_frame, unobserved_curr_accum_points_frame):
#def save_per_scan_obs_data(extracted_per_frame_dir, frame_num, building_edges_frame, observed_points_frame, curr_accum_points_frame, total_accum_points_frame):
    """
    """
    # Save all edges from buildings that were observed in current scan
    # frame_build_edges_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_build_edges.bin')
    # with open(frame_build_edges_file, 'wb') as bin_file:
        # np.array(building_edges_frame).tofile(bin_file)
    frame_build_edges_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_build_edges')
    np.save(frame_build_edges_file, building_edges_frame)  # Saving

    # # Save total accumulated points for all buildings that have been observed by current scan
    # frame_totalbuildaccum_scan_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_total_accum_points.bin')
    # with open(frame_totalbuildaccum_scan_file, 'wb') as bin_file:
    #     np.array(total_accum_points_frame).tofile(bin_file)
        
    # Save observed_points_frame
    #frame_obs_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_obs_points.bin')
    #with open(frame_obs_points_file, 'wb') as bin_file:
    #    np.array(observed_points_frame).tofile(bin_file)

    # Save the current accumulation of points of buildings that were observed in this scan
    # frame_obs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_curr_accum_points.bin')
    # with open(frame_obs_curr_accum_points_file, 'wb') as bin_file:
    #     np.array(curr_accum_points_frame).tofile(bin_file)
    frame_obs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_curr_accum_points')
    np.save(frame_obs_curr_accum_points_file, curr_accum_points_frame)

    # Save current accumulated difference from total
    # if len(unobserved_curr_accum_points_frame)>0:
    # frame_unobs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_unobs_curr_accum_points.bin')
    # with open(frame_unobs_curr_accum_points_file, 'wb') as bin_file:
    #     np.array(unobserved_curr_accum_points_frame).tofile(bin_file)
    frame_unobs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_unobs_curr_accum_points')
    np.save(frame_unobs_curr_accum_points_file, unobserved_curr_accum_points_frame)

def save_per_scan_obs_data(extracted_per_frame_dir, frame_num, building_edges_frame, curr_accum_points_frame, total_accum_points_frame):
    """
    """
    building_edges_frame = np.asarray(building_edges_frame)
    curr_accum_points_frame = np.asarray(curr_accum_points_frame)
    total_accum_points_frame = np.asarray(total_accum_points_frame)

    # Save all edges from buildings that were observed in current scan
    frame_build_edges_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_build_edges')
    np.save(frame_build_edges_file, building_edges_frame)  # Saving

    # Save total accumulated points for all buildings that have been observed by current scan
    frame_totalbuildaccum_scan_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_total_accum_points')
    np.save(frame_totalbuildaccum_scan_file, total_accum_points_frame)

    # Save the current accumulation of points of buildings that were observed in this scan
    frame_obs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_curr_accum_points')
    np.save(frame_obs_curr_accum_points_file, curr_accum_points_frame)

# TODO: Remove?
def save_per_scan_unobs_data(extracted_per_frame_dir, frame_num, unobserved_curr_accum_points_frame):
#def save_per_scan_unobs_data(extracted_per_frame_dir, frame_num, unobserved_points_frame, unobserved_curr_accum_points_frame):
    """
    """
    # Save current scan difference from total
    #if len(unobserved_points_frame)>0:
    #    frame_unobs_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_unobs_points.bin')
    #    with open(frame_unobs_points_file, 'wb') as bin_file:
    #        np.array(unobserved_points_frame).tofile(bin_file)

    # Save current accumulated difference from total
    if len(unobserved_curr_accum_points_frame)>0:
        frame_unobs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_unobs_curr_accum_points.bin')
        np.save(frame_unobs_curr_accum_points_file, unobserved_curr_accum_points_frame)
        # with open(frame_unobs_curr_accum_points_file, 'wb') as bin_file:
        #     np.array(unobserved_curr_accum_points_frame).tofile(bin_file)


# TODO: Probably delete
# def save_building_edges_and_accum(extracted_building_data_dir, hit_building_list):
#     '''
#     Save building edges and accumulated scan as np .bin file for each building that is hit by points during seq.
#     '''
#     for iter, hit_building in enumerate(hit_building_list):
#         iter += 1
        
#         hit_building_edges = []
#         for edge in hit_building.edges:
#             hit_building_edges.append(edge.edge_vertices)
#         hit_building_edges = np.array(hit_building_edges)

#         building_edges_file = os.path.join(extracted_building_data_dir, 'per_building', 'edges_accum', f'build_{iter}_edges.bin')
#         with open(building_edges_file, 'wb') as bin_file:
#             np.array(hit_building_edges).tofile(bin_file)

#         building_accum_scan_file = os.path.join(extracted_building_data_dir, 'per_building', 'edges_accum', f'build_{iter}_accum.bin')
#         with open(building_accum_scan_file, 'wb') as bin_file:
#             np.array(hit_building.accum_points).tofile(bin_file)







'''
Create:

o3d_processor.py

'''

# def get_pointcloud_from_txt(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     xyz_positions = []
#     for line in lines:
#         matrix_elements = np.array(line.split(), dtype=float)
#         x, y = matrix_elements[0], matrix_elements[1]
#         xyz_positions.append([x, y, 0])
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz_positions)
#     return pcd, xyz_positions

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

# def vertex_list_to_o3d_lineset(vertices):
#     building_line_set = o3d.geometry.LineSet()
#     building_points = vertices
#     building_lines_idx = [[i, i + 1] for i in range(len(building_points) - 1)]
#     building_lines_idx.append([len(building_points) - 1, 0])  # Closing the loop
#     building_line_set.points = o3d.utility.Vector3dVector(building_points)
#     building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
#     building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings
#     return building_line_set

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

# def load_xyz_positions(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     xyz_positions = []
#     for line in lines:
#         matrix_elements = np.array(line.split(), dtype=float)
#         x, y = matrix_elements[0], matrix_elements[1]
#         xyz_positions.append([x, y, 0])
#     return xyz_positions

def create_point_clouds_from_xyz(xyz_positions):
    point_clouds = []
    for xyz in xyz_positions:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([xyz])
        point_clouds.append(pcd)
    return point_clouds


















"""
Create new file name for these...?
"""

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

def get_velo_poses_list(init_frame, fin_frame, inc_frame, velodyne_poses, label_path):
    pos_latlong_list = []
    for frame_num in range(init_frame, fin_frame + 1, inc_frame):
        pc_frame_label_path = os.path.join(label_path, f'{frame_num:010d}.bin')
        if os.path.exists(pc_frame_label_path):
            transformation_matrix = velodyne_poses.get(frame_num)
            trans_matrix_oxts = np.asarray(convertPoseToOxts(transformation_matrix))
            pos_latlong = trans_matrix_oxts[:3]
            pos_latlong_list.append(pos_latlong)
    return pos_latlong_list

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

def read_vel_poses(file_path):
    velodyne_poses_xyz = {}
    velodyne_poses_latlon = {}

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
            velodyne_poses_xyz[frame_index] = matrix_4x4
            velodyne_poses_latlon[frame_index] = np.asarray(convertPoseToOxts(matrix_4x4))
    return velodyne_poses_xyz, velodyne_poses_latlon

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
