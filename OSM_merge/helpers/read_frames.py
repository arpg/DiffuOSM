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

train_test = 'train'
if (seq==8 or seq==18): train_test = 'test'
    
per_frame_build = os.path.join(kitti360Path, 'data_3d_extracted', sequence, 'buildings/per_frame', )

def plot_frame(per_frame_file, per_frame_accum_file, per_frame_diff_file):
    masked_points_frame = read_bin_file(per_frame_file)
    accum_points_frame = read_bin_file(per_frame_accum_file)
    diff_points_frame = read_bin_file(per_frame_diff_file)
    
    masked_frame_pcd = o3d.geometry.PointCloud()
    accum_frame_pcd = o3d.geometry.PointCloud()
    diff_frame_pcd = o3d.geometry.PointCloud()
    accum_points_frame = np.array(accum_points_frame)
    accum_points_frame[:, 2] = 0

    masked_frame_pcd.points = o3d.utility.Vector3dVector(masked_points_frame)
    accum_frame_pcd.points = o3d.utility.Vector3dVector(accum_points_frame)
    diff_frame_pcd.points = o3d.utility.Vector3dVector(diff_points_frame)

    masked_frame_pcd.paint_uniform_color([1, 0, 0]) # Red color for frame building points
    accum_frame_pcd.paint_uniform_color([0, 0, 0])  # Black color for accum frame points
    diff_frame_pcd.paint_uniform_color([0, 0, 1])   # Blue color for diff frame points

    o3d.visualization.draw_geometries([accum_frame_pcd, masked_frame_pcd, diff_frame_pcd])

def read_bin_file(file_path):
    """
    Reads a .bin file containing point cloud data.
    Assumes each point is represented by four float32 values (x, y, z, intensity).
    """
    point_cloud = np.fromfile(file_path)
    return point_cloud.reshape(-1, 3)


def main(): 
    frame = 30
    while frame < 4000:
        per_frame_file = os.path.join(per_frame_build, f'{frame:010d}.bin', )
        per_frame_accum_file = os.path.join(per_frame_build, f'{frame:010d}_accum.bin', )
        per_frame_diff_file = os.path.join(per_frame_build, f'{frame:010d}_diff.bin', )

        plot_frame(per_frame_file, per_frame_accum_file, per_frame_diff_file)

        frame += 100

  
if __name__=="__main__": 
    main() 