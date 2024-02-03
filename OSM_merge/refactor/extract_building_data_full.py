'''
Doncey Albin

'''

import os
import open3d as o3d
from open3d.visualization import gui
import numpy as np
from collections import namedtuple
import osmnx as ox

# Internal
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *

'''
Step 1: Load poses.

'''
oxts_pose_file_path = "/Users/donceykong/Desktop/kitti360Scripts/data/2013_05_28_drive_0005_sync_pose2oxts.txt"
xyz_point_clouds, xyz_positions = get_pointcloud_from_txt(oxts_pose_file_path) # Create point clouds from XYZ positions


'''
Step 2: Load pointcloud with "building" and "unlabeled" points.

'''
ply_file_path = '/Users/donceykong/Desktop/kitti360Scripts/data/output3D.ply'
point_cloud_3D = o3d.io.read_point_cloud(ply_file_path)

# Get the points as a numpy array
points_3D = np.asarray(point_cloud_3D.points)

# Set all z-components to zero to create a 2D point cloud
points_2D = points_3D.copy()
points_2D[:, 2] = 0

# Create a new 2D point cloud from the modified points
point_cloud_2D = o3d.geometry.PointCloud()
point_cloud_2D.points = o3d.utility.Vector3dVector(points_2D)

'''
Step 3: Filter buildings within bounds near pose path. This makes cycling through to see if edges were hit much faster.

'''
osm_file_path = '/Users/donceykong/Desktop/kitti360Scripts/data/map_0005.osm'

# Filter features for buildings
building_features = ox.features_from_xml(osm_file_path, tags={'building': True})
print(f"\nlen(buildings): {len(building_features)}")

threshold_dist = 0.0008 
building_list, building_line_set = get_buildings_near_poses(building_features, xyz_positions, threshold_dist)

'''
Step 4: Create edge_points_list via discretizing by a set number.

'''
num_points_per_edge = 100
discretize_all_building_edges(building_list, num_points_per_edge)


'''
Step 5: Filter lidar points that are a +/- radius distance away from and of the edge points in the edge_points list.

'''
radius = 0.000008
# TODO: Maybe here would be a good point to do some sort of scan-matching so that the buildings and OSM-polygons are better aligned
calc_points_on_building_edges(building_list, point_cloud_3D, point_cloud_2D, radius)

'''
Step 6: Filter buildings with no internal points.

'''
hit_building_list, hit_building_line_set = get_building_hit_list(building_list)





def extract_and_save_points(new_pcd_3D, hit_building_list):
    # print("\n\n-   -   -   -   -   extract_and_save_points     -   -   -   -   -")
    new_pcd_2D = np.copy(np.asarray(new_pcd_3D.points))
    new_pcd_2D[:, 2] = 0

    # Create a new 2D point cloud from the modified points
    point_cloud_2D = o3d.geometry.PointCloud()
    point_cloud_2D.points = o3d.utility.Vector3dVector(new_pcd_2D)

    len_hit_building_list = len(hit_building_list)
    point_cloud_2D_kdtree = KDTree(np.asarray(point_cloud_2D.points))
    for iter, hit_building in enumerate(hit_building_list):
        iter += 1
        # print(f"    - Hit Building: {iter} / {len_hit_building_list}")
        masked_points_building = []
        for edge in hit_building.expanded_edges:
            distances, indices = point_cloud_2D_kdtree.query([edge])
            # Use a mask to filter 3D points that are within the XY radius from the edge point
            mask = abs(distances) <= radius
            masked_points = np.asarray(new_pcd_3D.points)[indices[mask]]
            masked_points_building.extend(masked_points)
            # Update building statistics based on the number of points within the radius

        if len(masked_points_building) > 0:
            hit_building.scan_num += 1
            hit_building.points.extend(masked_points_building)

            # Save hit_building.points as .bin file
            file_name = f"/Users/donceykong/Desktop/kitti360Scripts/OSM_merge/scans_step_1/hitbuilding_{iter+1}_scan_{hit_building.scan_num}.bin"
            with open(file_name, 'wb') as bin_file:
                np.array(masked_points_building).tofile(bin_file)

            # # Save hit_building.points as .bin file
            # file_name = f"/Users/donceykong/Desktop/kitti360Scripts/OSM_merge/scans_step_1/hitbuilding_{iter}_scan_{hit_building.scan_num}.bin"
            # with open(file_name, 'wb') as bin_file:
            #     np.array(hit_building.points).tofile(bin_file)






# Create a dictionary for label colors
labels_dict = {label.id: label.color for label in labels}

poses_file = '/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/vel_poses.txt'
transformation_matrices = get_transform_matrices(poses_file)

frame_number = 30  # starting frame number
MAX_FRAME_NUMBER = 6000

while True:
    new_pcd = load_and_visualize(frame_number, transformation_matrices, labels_dict)
    
    if new_pcd:
        print(f"frame: {frame_number}")
        extract_and_save_points(new_pcd, hit_building_list)
    
    frame_number += 1  # Increment the frame number

    # Exit the loop if you've processed all frames
    if frame_number > MAX_FRAME_NUMBER:  # Define the maximum frame number you want to process
        break