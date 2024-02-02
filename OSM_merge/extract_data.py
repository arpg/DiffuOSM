import os
import numpy as np
import open3d as o3d
import osmnx as ox
import pickle

# Internal
from tools.labels import labels
from tools.convert_oxts_pose import *
import tools.osm_building as osm_building
from tools.utils import *


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
o3d.visualization.draw_geometries([building_line_set])

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
# Save hit_building_list to a file
with open('hit_building_list_seq_0005.pkl', 'wb') as file:
    pickle.dump(hit_building_list, file)

'''
Step 7: Save filtered building points.

'''
filtered_points = [point for building in hit_building_list for point in building.points]
filtered_point_cloud = o3d.geometry.PointCloud()
filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
filtered_point_cloud.paint_uniform_color([1, 0, 0])  # Blue color for buildings

'''
Optional: View bounds on building edges which points were included

'''
all_edge_circles = get_building_edge_bounds(hit_building_list, radius)
all_edge_circles.paint_uniform_color([0, 1, 0])

'''
Step 8: Visualize the filtered point cloud

'''
# # Visualize building points on OG point cloud
# o3d.visualization.draw_geometries([point_cloud, filtered_point_cloud, hit_building_line_set])

# Visualize buildings that have been hit by laser and the clouds
o3d.visualization.draw_geometries([all_edge_circles, filtered_point_cloud, hit_building_line_set])







































# import os
# import numpy as np
# import open3d as o3d
# from open3d.io import write_point_cloud

# # Internal
# from OSM_merge.tools.labels import labels
# from OSM_merge.tools.convert_oxts_pose import *


# # Create a dictionary for label colors
# labels_dict = {label.id: label.color for label in labels}

# poses_file = '/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_poses/2013_05_28_drive_0005_sync/vel_poses.txt'
# transformation_matrices = get_transform_matrices(poses_file)

# file_path = "/Users/donceykong/Desktop/kitti360Scripts/data/2013_05_28_drive_0005_sync_pose2oxts.txt"
# xyz_positions = load_xyz_positions(file_path)

# # Create point clouds from XYZ positions
# xyz_point_clouds = create_point_clouds_from_xyz(xyz_positions)

# # List to hold all point cloud geometries
# pcd_geometries = []

# # Iterate through frame numbers and load each point cloud
# frame_num = 30  # Initial frame number
# total_labels = 6255
# while frame_num <= total_labels:
#     # print(f"frame_num: {frame_num}")
#     pcd, new_min = load_and_visualize(frame_num, transformation_matrices)
#     frame_num += 1
#     if pcd is not None:
#         pcd_geometries.append(pcd)

# # Merge all point clouds in pcd_geometries into a single point cloud
# merged_pcd = o3d.geometry.PointCloud()
# for pcd in pcd_geometries:
#     merged_pcd += pcd

# # Save the merged point cloud to a PLY file
# output_file_path = '/Users/donceykong/Desktop/kitti360Scripts/data/output3D.ply'  # Specify your output file path here
# o3d.io.write_point_cloud(output_file_path, merged_pcd)

# print(f"Saved merged point cloud to {output_file_path}")