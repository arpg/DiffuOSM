'''
Doncey Albin

'''

import numpy as np
from scipy.spatial import cKDTree, KDTree as scipyKDTree
from sklearn.neighbors import KDTree as sklearnKDTree

class PointCloudProcessor:
    def __init__(self):
        # Initialize point cloud processor
        ...

    # # TODO: use ball point kdtree query
    # def remove_overlapping_points(self, accum_points, frame_points):
    #     total_accum_points_array = np.asarray(accum_points)
    #     frame_points_array = np.asarray(frame_points)

    #     filtered_frame_points = []

    #     # Query each accum_point in the KDTree of frame_points to check if there's an exact match (distance = 0)
    #     for accum_point in total_accum_points_array:
    #         distance, _ = frame_points_kdtree.query(accum_point)
    #         if distance > 0:
    #             filtered_frame_points.append(accum_point)
    #     return np.array(filtered_frame_points)
    

    def remove_overlapping_points(self, total_accum_points, frame_points):
        total_accum_points_array = np.asarray(total_accum_points)
        frame_points_array = np.asarray(frame_points)

        frame_points_kdtree = scipyKDTree(frame_points_array) # Use KD-tree for faster searching

        # Query the KDTree with all accum_points at once to find nearest neighbors
        distances, _ = frame_points_kdtree.query(total_accum_points_array)
        filtered_accum_points = total_accum_points_array[distances > 0]

        return filtered_accum_points

    # def remove_overlapping_points_efficient(self, larger_points, smaller_points):
    #     # Convert lists of points to sets of tuples
    #     larger_set = set(map(tuple, larger_points))
    #     smaller_set = set(map(tuple, smaller_points))

    #     # Find the difference between the two sets
    #     unique_to_larger_set = larger_set - smaller_set

    #     # (Optional) Convert back to a list of numpy arrays or keep as tuples based on your needs
    #     unique_points_list = np.array(list(unique_to_larger_set))

    #     return unique_points_list

    def remove_overlapping_points_efficient(self, A, B):
        # Ensure A and B are numpy arrays (if they aren't already)
        A = np.asarray(A)
        B = np.asarray(B)
        
        # Convert points to a structured array with an additional 'z' dimension for 3D points
        dtype = [('x', A.dtype), ('y', A.dtype), ('z', A.dtype)]  # Adjusted for 3D
        A_struct = A.view(dtype).reshape(-1)
        B_struct = B.view(dtype).reshape(-1)
        
        # Use setdiff1d to find unique points in A that are not in B
        unique_struct = np.setdiff1d(A_struct, B_struct)
        
        # Convert back to original shape/format
        # The shape is set to -1, 3 to ensure it's reshaped as 3D points
        unique_points = unique_struct.view(A.dtype).reshape(-1, 3)
        
        return unique_points