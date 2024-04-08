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

    def remove_overlapping_points(self, total_accum_points, frame_points):
        total_accum_points_array = np.asarray(total_accum_points)
        frame_points_array = np.asarray(frame_points)

        frame_points_kdtree = cKDTree(frame_points_array)

        # Query the KDTree with all accum_points at once to find nearest neighbors
        distances, _ = frame_points_kdtree.query(total_accum_points_array)
        filtered_accum_points = total_accum_points_array[distances > 0]

        return filtered_accum_points