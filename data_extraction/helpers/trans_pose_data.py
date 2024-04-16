'''
Doncey Albin

Utility for loading pose data from KITTI360 dataset
'''

# Standard Library
import os

# Third party
import numpy as np
import osmnx as ox
import open3d as o3d
from open3d.visualization import gui
from collections import namedtuple

# Internal
from tools.labels import labels
from tools.utils import * ## TODO remove * imports
from tools.convert_oxts_pose import * ## TODO Remove * imports


if __name__ == '__main__'
    # Define the translation vector from IMU to LiDAR
    translation_vector = np.array([0.81, 0.32, -0.83])

    # Define the rotation matrix for a 180-degree rotation about the X-axis (imu -> lidar)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])

    imu_poses_file = 'kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/poses.txt'
    vel_poses_file = 'kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/vel_poses.txt'

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

    # Write the LiDAR poses to file
    write_poses(vel_poses_file, lidar_poses, frame_indices)