import os
import numpy as np
import open3d as o3d
import osmnx as ox
import pickle

# Internal
from OSM_merge.tools.labels import labels
from OSM_merge.tools.convert_oxts_pose import *
import OSM_merge.tools.osm_building as osm_building


'''
View path of poses

'''
def create_point_clouds_from_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    xyz_positions = []
    for line in lines:
        matrix_elements = np.array(line.split(), dtype=float)
        x, y = matrix_elements[0], matrix_elements[1]
        xyz_positions.append([x, y, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_positions)
    return pcd, xyz_positions
file_path = "/Users/donceykong/Desktop/kitti360Scripts/data/2013_05_28_drive_0005_sync_pose2oxts.txt"
xyz_point_clouds, xyz_positions = create_point_clouds_from_xyz(file_path) # Create point clouds from XYZ positions


'''
Point cloud with "building" and "unlabeled" points

'''
ply_file_path = '/Users/donceykong/Desktop/kitti360Scripts/data/output.ply'
point_cloud = o3d.io.read_point_cloud(ply_file_path)
# o3d.visualization.draw_geometries([point_cloud])


'''
Cycle through saved OSM buildings that are hit by points.

'''
with open('hit_building_list.pkl', 'rb') as file:
    hit_building_list = pickle.load(file)

hit_building_line_set = o3d.geometry.LineSet()
hit_building_points = [point for building in hit_building_list for line in building.edges for point in line]
hit_building_lines_idx = []
hit_building_lines_idx = [[i, i + 1] for i in range(0, len(hit_building_points) - 1, 2)]
hit_building_line_set.points = o3d.utility.Vector3dVector(hit_building_points)
hit_building_line_set.lines = o3d.utility.Vector2iVector(hit_building_lines_idx)
hit_building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings

'''
 Now, only use lidar points that are a +/- radius distance away from and of the edge points in the edge_points list.

'''
filtered_points = []  # To store filtered lidar points
edge_points_hit = []  # Save list of edges hit by points
radius = 0.000008
# Iterate through each lidar point
print(f"np.asarray(point_cloud.points).shape: {np.asarray(point_cloud.points).shape}")
iter = 0
tot_iter = 0
for lidar_point in np.asarray(point_cloud.points):
    iter += 1
    tot_iter += 1
    if iter < 5:
        continue
    iter = 0

    min_distance = float('inf')

    # if tot_iter > 1000000:
    #     break

    # Iterate through edge points in edge_points_list
    for building in hit_building_list:
        edge_points = building.expanded_edges
        # Calculate distances from the current lidar point to all edge points in the current edge
        distances = np.linalg.norm(edge_points - lidar_point, axis=1)
        min_distance = min(min_distance, np.min(distances))
    
        # Check if the lidar point is within the radius of any edge point
        if min_distance <= radius:
            # print("             ----> Min found")
            print(f"tot_iter: {tot_iter}")
            building.times_hit += 1
            filtered_points.append(lidar_point)
            break

# TODO: Save lidar points here
# with open('filtered_points.pkl', 'wb') as file:
#     pickle.dump(filtered_points, file)

# Create a point cloud from the filtered points
filtered_point_cloud = o3d.geometry.PointCloud()
filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

# Visualize the filtered point cloud
o3d.visualization.draw_geometries([filtered_point_cloud, hit_building_line_set])


'''
 
 2) Save points in an array with shape (num_points, num_buildings, 3), make sure it corresponds to above. --> save as npy file.

 4) Color points as "labeled" and "unlabled".
 5) Remove iter limit.

'''