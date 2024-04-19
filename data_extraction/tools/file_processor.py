import os
import numpy as np

# Internal
import tools.osm_building as osm_building
from tools.convert_oxts_pose import *

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

def save_per_scan_obs_data(extracted_per_frame_dir, frame_num, building_edges_frame, observed_points_frame, curr_accum_points_frame, total_accum_points_frame):
    """
    """
    # Save all edges from buildings that were observed in current scan
    frame_build_edges_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_build_edges.bin')
    with open(frame_build_edges_file, 'wb') as bin_file:
        np.array(building_edges_frame).tofile(bin_file)

    # Save total accumulated points for all buildings that have been observed by current scan
    frame_totalbuildaccum_scan_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_total_accum_points.bin')
    with open(frame_totalbuildaccum_scan_file, 'wb') as bin_file:
        np.array(total_accum_points_frame).tofile(bin_file)
        
    # Save observed_points_frame
    frame_obs_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_obs_points.bin')
    with open(frame_obs_points_file, 'wb') as bin_file:
        np.array(observed_points_frame).tofile(bin_file)

    # Save the current accumulation of points of buildings that were observed in this scan
    frame_obs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_curr_accum_points.bin')
    with open(frame_obs_curr_accum_points_file, 'wb') as bin_file:
        np.array(curr_accum_points_frame).tofile(bin_file)

def save_per_scan_unobs_data(extracted_per_frame_dir, frame_num, unobserved_points_frame, unobserved_curr_accum_points_frame):
    """
    """
    # Save current scan difference from total
    if len(unobserved_points_frame)>0:
        frame_unobs_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_unobs_points.bin')
        with open(frame_unobs_points_file, 'wb') as bin_file:
            np.array(unobserved_points_frame).tofile(bin_file)

    # Save current accumulated difference from total
    if len(unobserved_curr_accum_points_frame)>0:
        frame_unobs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_unobs_curr_accum_points.bin')
        with open(frame_unobs_curr_accum_points_file, 'wb') as bin_file:
            np.array(unobserved_curr_accum_points_frame).tofile(bin_file)


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
            
def load_xyz_positions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    xyz_positions = []
    for line in lines:
        matrix_elements = np.array(line.split(), dtype=float)
        x, y = matrix_elements[0], matrix_elements[1]
        xyz_positions.append([x, y, 0])
    return xyz_positions

def write_poses(file_path, transformation_matrices, frame_indices):
    with open(file_path, 'w') as file:
        for idx, matrix in zip(frame_indices, transformation_matrices):
            # Flatten the matrix to a 1D array, convert to strings, and join with spaces
            matrix_string = ' '.join(map(str, matrix.flatten()))
            # Write the frame index followed by the flattened matrix
            file.write(f"{idx} {matrix_string}\n")

# TODO: Only need one of two below functions
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