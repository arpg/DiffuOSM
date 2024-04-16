'''
Doncey Albin


'''

import os
from collections import namedtuple
from typing import List

import open3d as o3d
import numpy as np
import numpy.typing as npt


# Internal
from tools.labels import labels
from OSM_merge.tools.utils import * ## TODO remove * import


# Create a dictionary for label colors
labels_dict = {label.id: label.color for label in labels} ## TODO globals should be uppercase snake case. Ex: LABELS_DICT

def color_point_cloud(points: npt.ArrayLike, labels: List[int]) -> npt.NDArray:
    '''
    Colors the point cloud based on the labels.
    Each point in the point cloud is assigned the color of its label.

    params:
        points (npt.ArrayLike): points to color
        labels (List[int]): Labels to color points

    returns:
        colored_points (npt.NDArray): points wiht colors mapped on
    '''
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

def load_and_visualize(frame_number: int) -> o3d.Vector3dVector:
    '''
    Loads a point cloud, colors it, visualizes it, then returns it

    params:
        frame_number (int): frame to process

    returns:
        colored_pcd (o3d.Vector3dVector): Point cloud corresponding to frame number
    '''
    # Adjust file paths based on frame number
    pc_filepath = f'/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_3d_raw/2013_05_28_drive_0005_sync/velodyne_points/data/{frame_number:010d}.bin'
    label_filepath = f'/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_3d_semantics/train/2013_05_28_drive_0005_sync/labels/{frame_number:010d}.bin'

    if not os.path.exists(pc_filepath) or not os.path.exists(label_filepath):
        print(f"File not found for frame number {frame_number}")
        return None, None

    # Step 1: Read pointcloud bin files and label bin files
    pc = read_bin_file(pc_filepath)
    pcd = visualize_point_cloud(pc)
    labels_np = read_label_bin_file(label_filepath)

    # print(f"len_labels: {labels_np.shape}, len_points: {pc.shape}")

    # Step 2: Color the point cloud
    colored_points = color_point_cloud(pc, labels_np)
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(pc[:, :3])  # Only use xyz for geometry
    colored_pcd.colors = o3d.utility.Vector3dVector(colored_points)  # Set colors

    return colored_pcd

def change_frame(vis, key_code):
    '''
    Changes the data currently visualized to the next frame

    params:
        vis (): o3d visualization object
        key_code (): key code? ## TODO not sure about this

    returns:
        bool: whether the frame is valid or not
    '''
    global frame_number
    if key_code == ord('N'):
        frame_number += 1
    elif key_code == ord('P'):
        frame_number -= 1
    else:
        return False
    new_pcd = load_and_visualize(frame_number)
    if new_pcd:
        vis.clear_geometries()
        vis.add_geometry(new_pcd)
    return True

if __name__ == '__main__':

    frame_number = 100  # starting frame number
    initial_pcd = load_and_visualize(frame_number)

    if initial_pcd:
        key_to_callback = {
            ord('N'): lambda vis: change_frame(vis, ord('N')),
            ord('P'): lambda vis: change_frame(vis, ord('P'))
        }
        o3d.visualization.draw_geometries_with_key_callbacks([initial_pcd], key_to_callback)