import os
import numpy as np

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

# Define the translation vector from IMU to LiDAR
translation_vector = np.array([0.81, 0.32, -0.83])

# Define the rotation matrix for a 180-degree rotation about the X-axis
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
