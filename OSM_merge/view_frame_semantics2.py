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










# Create a dictionary for label colors
labels_dict = {label.id: label.color for label in labels}

poses_file = '/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/vel_poses.txt'
transformation_matrices = get_transform_matrices(poses_file)

def change_frame(vis, key_code):
    global frame_number
    if key_code == ord('N'):
        frame_number += 50
    elif key_code == ord('P'):
        frame_number -= 50
    else:
        return False
    new_pcd = load_and_visualize(frame_number, transformation_matrices, labels_dict)
    # label_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    # label = gui.Label3D(f"frame: {frame_number}", label_position)
    if new_pcd:
        # vis.clear_geometries()
        vis.add_geometry(hit_building_line_set)
        vis.add_geometry(new_pcd)
        # vis.add_geometry(label)
    return True

frame_number = 30  # starting frame number
initial_pcd = load_and_visualize(frame_number, transformation_matrices, labels_dict)

if initial_pcd:
    key_to_callback = {
        ord('N'): lambda vis: change_frame(vis, ord('N')),
        ord('P'): lambda vis: change_frame(vis, ord('P'))
    }
    o3d.visualization.draw_geometries_with_key_callbacks([hit_building_line_set, initial_pcd], key_to_callback)