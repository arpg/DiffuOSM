'''
Doncey Albin

Readframes.py

'''

import os
import numpy as np
import open3d as o3d


seq = 0

if 'KITTI360_DATASET' in os.environ:
    kitti360Path = os.environ['KITTI360_DATASET']
else:
    kitti360Path = os.path.join(os.path.dirname(
                        os.path.realpath(__file__)), '..','data/KITTI-360')

sequence = '2013_05_28_drive_%04d_sync' % seq
kitti360Path = kitti360Path
    
per_frame_build = os.path.join(kitti360Path, 'data_3d_extracted', sequence, 'buildings', 'per_frame')

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
    accum_points_file = os.path.join(per_frame_build, f'{frame:010d}_accum_points.bin', )
    obs_points_file = os.path.join(per_frame_build, f'{frame:010d}_obs_points.bin', )
    unobs_points_file = os.path.join(per_frame_build, f'{frame:010d}_unobs_points.bin', )
    obs_edges_file = os.path.join(per_frame_build, f'{frame:010d}_obs_edges.bin', )
    unobs_edges_file = os.path.join(per_frame_build, f'{frame:010d}_unobs_edges.bin', )

    files_exist = False
    if not os.path.exists(accum_points_file) or not os.path.exists(obs_points_file):
        return files_exist, None, None, None, None, None
    else:
        files_exist = True

    accum_points = read_bin_file(accum_points_file)
    # accum_points[:, 2] = 0 # Set accum points to 2D grid
    accum_points[:, 2] -= np.min(accum_points[:, 2])
    obs_points = read_bin_file(obs_points_file)
    obs_points[:, 2] -= np.min(obs_points[:, 2])
    unobs_points = read_bin_file(unobs_points_file)
    unobs_points[:, 2] -= np.min(unobs_points[:, 2])
    obs_edges_points, obs_edges_lines = read_edges_file(obs_edges_file)
    unobs_edges_points, unobs_edges_lines = read_edges_file(unobs_edges_file)

    accum_frame_pcd = o3d.geometry.PointCloud()
    obs_points_pcd = o3d.geometry.PointCloud()
    unobs_points_pcd = o3d.geometry.PointCloud()
    obs_edges_pcd = o3d.geometry.LineSet()
    unobs_edges_pcd = o3d.geometry.LineSet()

    accum_frame_pcd.points = o3d.utility.Vector3dVector(accum_points)
    obs_points_pcd.points = o3d.utility.Vector3dVector(obs_points)
    unobs_points_pcd.points = o3d.utility.Vector3dVector(unobs_points)
    obs_edges_pcd.points = o3d.utility.Vector3dVector(obs_edges_points)
    obs_edges_pcd.lines = o3d.utility.Vector2iVector(obs_edges_lines)
    unobs_edges_pcd.points = o3d.utility.Vector3dVector(unobs_edges_points)
    unobs_edges_pcd.lines = o3d.utility.Vector2iVector(unobs_edges_lines)

    accum_frame_pcd.paint_uniform_color([0, 0, 0])  # Black color for accum frame points
    obs_points_pcd.paint_uniform_color([0, 1, 0]) # Red color for frame building points
    unobs_points_pcd.paint_uniform_color([1, 0, 0])   # Blue color for diff frame points
    obs_edges_pcd.paint_uniform_color([0, 1, 0]) # Red color for frame building points
    unobs_edges_pcd.paint_uniform_color([1, 0, 0]) # Red color for frame building points

    return files_exist, accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd

def plot_pcds(accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd):
    o3d.visualization.draw_geometries([accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd])

def get_accum_pcds(): 
    global frame_min
    global frame_max
    global frame_inc
    global frame

    frame = frame_min

    files_exist, accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd = get_pcds(frame)

    all_accum_frame_pcds = accum_frame_pcd
    all_edge_pcds = (obs_edges_pcd + unobs_edges_pcd)

    frame += frame_inc
    while frame < frame_max:
        files_exist, accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd = get_pcds(frame)
        if (files_exist):
            all_accum_frame_pcds += accum_frame_pcd
            all_edge_pcds += (obs_edges_pcd + unobs_edges_pcd)
        frame += frame_inc

    all_accum_frame_pcds.paint_uniform_color([0, 0, 0])  # Black color for accum frame points
    all_edge_pcds.paint_uniform_color([0, 0, 1])  # Black color for accum frame points

    frame = frame_min

    return all_accum_frame_pcds, all_edge_pcds

def change_frame(vis, key_code, all_accum_frame_pcds, all_edge_pcds):
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
    
    files_exist, accum_frame_pcd, obs_points_pcd, unobs_points_pcd, obs_edges_pcd, unobs_edges_pcd = get_pcds(frame)

    if (files_exist):
        voxel_size = 0.00005  # Define the voxel size, adjust this value based on your needs
        ds_accum = accum_frame_pcd.voxel_down_sample(voxel_size)
        ds_accum_points.extend(ds_accum.points)
        ds_accum_points_pcd.points = o3d.utility.Vector3dVector(ds_accum_points)
        ds_accum_points_pcd.paint_uniform_color([0, 0, 1])  # Blue color for accum frame points

        center = obs_points_pcd.get_center()
        axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.0001, origin=center)
        # extrinsic = np.eye(4)
        # extrinsic[:3, :3] = np.eye(3)
        # extrinsic[:3, 3] = center

        vis.clear_geometries()
        # vis.add_geometry(all_accum_frame_pcds)
        # vis.add_geometry(all_edge_pcds)
        vis.add_geometry(ds_accum_points_pcd)
        vis.add_geometry(unobs_edges_pcd)
        vis.add_geometry(obs_edges_pcd)
        vis.add_geometry(unobs_points_pcd)
        vis.add_geometry(obs_points_pcd)
        vis.add_geometry(axis_frame)
        
        # Control where the visualizer looks at
        vis.get_view_control().set_lookat(center)
        vis.get_view_control().set_front([-0.5, -0.3, 1])
        zoom = 0.00005
        vis.get_view_control().set_zoom(zoom)  

    return True

frame_min = 30
frame_max = 4000
frame_inc = 25
frame = frame_min
ds_accum_points = []
ds_accum_points_pcd = o3d.geometry.PointCloud()
def main(): 
    all_accum_frame_pcds, all_edge_pcds = get_accum_pcds()
    voxel_size = 0.00001  # Define the voxel size, adjust this value based on your needs
    all_accum_frame_pcds = all_accum_frame_pcds.voxel_down_sample(voxel_size)

    key_to_callback = {
        ord('N'): lambda vis: change_frame(vis, ord('N'), all_accum_frame_pcds, all_edge_pcds),
        ord('P'): lambda vis: change_frame(vis, ord('P'), all_accum_frame_pcds, all_edge_pcds)
    }
    o3d.visualization.draw_geometries_with_key_callbacks([all_edge_pcds], key_to_callback)

if __name__=="__main__": 
    main() 