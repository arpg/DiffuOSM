'''
Doncey Albin

Readframes.py

'''

import os
import glob
import numpy as np
import open3d as o3d

# Internal
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *

seq = 0

if 'KITTI360_DATASET' in os.environ:
    kitti360Path = os.environ['KITTI360_DATASET']
else:
    kitti360Path = os.path.join(os.path.dirname(
                        os.path.realpath(__file__)), '..','data/KITTI-360')

sequence = '2013_05_28_drive_%04d_sync' % seq
per_frame_build = os.path.join(kitti360Path, 'data_3d_extracted', sequence, 'buildings', 'per_frame')

imu_poses_file = os.path.join(kitti360Path, 'data_poses', sequence, 'poses.txt')
velodyne_poses_file = os.path.join(kitti360Path, 'data_poses', sequence, 'velodyne_poses.txt')
#velodyne_poses = get_trans_poses_from_imu_to_velodyne(imu_poses_file, velodyne_poses_file, save_to_file=True)
velodyne_poses = read_vel_poses(velodyne_poses_file)

def read_bin_file(file_path):
    point_cloud = np.fromfile(file_path)
    return point_cloud.reshape(-1, 3)

def read_edges_file(building_edges_file):
    with open(building_edges_file, 'rb') as bin_file:
        edges_array = np.fromfile(bin_file, dtype=float).reshape(-1, 2, 3)  # Reshape to 3D array
    build_edges_points = edges_array.reshape(-1, 3)
    build_edges_lines_idx = [[i, i + 1] for i in range(0, len(build_edges_points) - 1, 2)]
    return build_edges_points, build_edges_lines_idx

def get_pcds(frame):
    build_edges_file = os.path.join(per_frame_build, f'{frame:010d}_build_edges.bin', )
    obs_curr_accum_points_file = os.path.join(per_frame_build, f'{frame:010d}_curr_accum_points.bin', )
    obs_points_file = os.path.join(per_frame_build, f'{frame:010d}_obs_points.bin', )
    total_accum_points_file = os.path.join(per_frame_build, f'{frame:010d}_total_accum_points.bin', )
    unobs_points_file = os.path.join(per_frame_build, f'{frame:010d}_unobs_points.bin', )
    unobs_curr_accum_points_file = os.path.join(per_frame_build, f'{frame:010d}_unobs_curr_accum_points.bin', )

    files_exist = False
    if not os.path.exists(unobs_curr_accum_points_file) or not os.path.exists(obs_points_file):
        print(f"File not found: {unobs_curr_accum_points_file}")
        return files_exist, None, None, None, None
    else:
        files_exist = True

    # total_accum_points = read_bin_file(total_accum_points_file)
    # # total_accum_points[:, 2] = 0 # Set accum points to 2D grid
    # total_accum_points_file[:, 2] -= np.min(total_accum_points_file[:, 2])

    obs_curr_accum_points = read_bin_file(obs_curr_accum_points_file)
    obs_curr_accum_points[:, 2] -= np.min(obs_curr_accum_points[:, 2])

    unobs_curr_accum_points = read_bin_file(unobs_curr_accum_points_file)
    unobs_curr_accum_points[:, 2] -= np.min(unobs_curr_accum_points[:, 2])

    build_edges_points, build_edges_lines = read_edges_file(build_edges_file)

    transformation_matrix = velodyne_poses.get(frame)
    trans_matrix_oxts = np.asarray(convertPoseToOxts(transformation_matrix))
    pos_latlong = trans_matrix_oxts[:3]
    pos_latlong[2] = 0

    obs_curr_accum_points_pcd = o3d.geometry.PointCloud()
    unobs_curr_accum_points_pcd = o3d.geometry.PointCloud()
    build_edges_pcd = o3d.geometry.LineSet()

    obs_curr_accum_points_pcd.points = o3d.utility.Vector3dVector(obs_curr_accum_points)
    unobs_curr_accum_points_pcd.points = o3d.utility.Vector3dVector(unobs_curr_accum_points)
    build_edges_pcd.points = o3d.utility.Vector3dVector(build_edges_points)
    build_edges_pcd.lines = o3d.utility.Vector2iVector(build_edges_lines)

    obs_curr_accum_points_pcd.paint_uniform_color([0, 0, 0])    # Black color for accum frame points
    unobs_curr_accum_points_pcd.paint_uniform_color([0, 1, 0])  # Green color for unobs frame points
    build_edges_pcd.paint_uniform_color([0, 0, 1])              # Blue color for OSM build edges

    return files_exist, obs_curr_accum_points_pcd, unobs_curr_accum_points_pcd, build_edges_pcd, pos_latlong

def plot_pcds(accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd):
    o3d.visualization.draw_geometries([accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd])

# def get_accum_pcds(): 
#     global frame_min
#     global frame_max
#     global frame_inc
#     global frame

#     frame = frame_min
#     files_exist, accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd = get_pcds(frame)

#     # print(f"files_exist: {files_exist} for frame {frame}")
#     all_accum_frame_pcds = accum_frame_pcd
#     all_edge_pcds = (obs_edges_pcd + unobs_edges_pcd)

#     frame += frame_inc
#     while frame < frame_max:
#         files_exist, accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd = get_pcds(frame)
#         if (files_exist):
#             all_accum_frame_pcds += accum_frame_pcd
#             all_edge_pcds += (obs_edges_pcd + unobs_edges_pcd)
#         frame += frame_inc

#     all_accum_frame_pcds.paint_uniform_color([0, 0, 0])  # Black color for accum frame points
#     all_edge_pcds.paint_uniform_color([0, 0, 1])  # Black color for accum frame points

#     frame = frame_min

#     return all_accum_frame_pcds, all_edge_pcds

def change_frame(vis, key_code):
    global frame_min
    global frame_max
    global frame_inc
    global frame
    global ds_accum_points
    global ds_accum_points_pcd

    if key_code == ord('N') and frame < frame_max:
        frame += frame_inc
    elif key_code == ord('P') and frame > frame_min:
        frame -= frame_inc
        
    files_exist, obs_curr_accum_points_pcd, unobs_curr_accum_points_pcd, build_edges_pcd, pos_latlong = get_pcds(frame)

    if (files_exist):
        voxel_size = 0.00001  # Define the voxel size, adjust this value based on your needs
        ds_accum = obs_curr_accum_points_pcd.voxel_down_sample(voxel_size)
        ds_accum_points.extend(obs_curr_accum_points_pcd.points)#ds_accum.points)
        ds_accum_points_pcd.points = o3d.utility.Vector3dVector(ds_accum_points)
        ds_accum_points_pcd.paint_uniform_color([0.2, 0.2, 0.4])  # RED color for accum frame points

        center = obs_curr_accum_points_pcd.get_center()
        
        # Create a coordinate frame at the center of the point cloud
        axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.0001, origin=pos_latlong)

        # extrinsic = np.eye(4)
        # extrinsic[:3, :3] = np.eye(3)
        # extrinsic[:3, 3] = center

        vis.clear_geometries()
        # vis.add_geometry(ds_accum_points_pcd)
        vis.add_geometry(obs_curr_accum_points_pcd)
        vis.add_geometry(unobs_curr_accum_points_pcd)
        vis.add_geometry(build_edges_pcd)
        vis.add_geometry(axis_frame)
        
        # o3d.visualization.draw_geometries([obs_curr_accum_points_pcd, unobs_curr_accum_points_pcd])#, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd])
        # # Control where the visualizer looks at
        # vis.get_view_control().set_lookat(center)
        # vis.get_view_control().set_front([-0.5, -0.3, 1])
        # zoom = 0.00005
        # vis.get_view_control().set_zoom(zoom)  
        vis.get_view_control().set_lookat(center)

        # Update the visualization window
        # vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

    return True

def find_min_max_file_names(label_path):
    # Pattern to match all .bin files in the directory
    pattern = os.path.join(label_path, '*_total_accum_points.bin')

    # List all .bin files
    files = glob.glob(pattern)
    # print(f"files[0]: {os.path.basename(files[0]).split('_accum_points')}")

    # Extract the integer part of the file names
    file_numbers = [int(os.path.basename(file).split('_total_accum_points')[0]) for file in files]
    # Find and return min and max
    if file_numbers:  # Check if list is not empty
        min_file, max_file = min(file_numbers), max(file_numbers)
        return min_file, max_file
    else:
        return None, None

frame_min, frame_max = find_min_max_file_names(per_frame_build)
# print(f"frame_min: {frame_min}, frame_max: {frame_max}")
frame_inc = 10
frame = frame_min
ds_accum_points = []
ds_accum_points_pcd = o3d.geometry.PointCloud()
def main():
    # all_accum_frame_pcds, all_edge_pcds = get_accum_pcds()
    # voxel_size = 0.00001  # Define the voxel size, adjust this value based on your needs
    # all_accum_frame_pcds = all_accum_frame_pcds.voxel_down_sample(voxel_size)

    key_to_callback = {
        ord('N'): lambda vis: change_frame(vis, ord('N')),
        ord('P'): lambda vis: change_frame(vis, ord('P'))
    }
    o3d.visualization.draw_geometries_with_key_callbacks([], key_to_callback)

if __name__=="__main__": 
    main() 