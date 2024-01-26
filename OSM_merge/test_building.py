import os
import numpy as np
import open3d as o3d
import osmnx as ox

# Internal
from labels import labels
from convert_oxts_pose import *

def discretize_edge(start_point, end_point, num_points):
    """
    Generate linearly spaced points between start_point and end_point.
    """
    return np.linspace(start_point, end_point, num_points, axis=0)

def is_point_near_edge(point_cloud, edge_points, threshold):
    """
    Check if any point in point_cloud is within 'threshold' distance of any point in edge_points.
    """
    for point in edge_points:
        # Compute distances from the current edge point to all points in the point cloud
        distances = np.linalg.norm(np.asarray(point_cloud.points) - point, axis=1)
        if np.any(distances < threshold):
            print("     point is near")
            return True
    return False

def create_circle(center, radius, num_points=30):
    """
    Create a circle at a given center point.

    :param center: Center of the circle (x, y, z).
    :param radius: Radius of the circle.
    :param num_points: Number of points to approximate the circle.
    :return: Open3D point cloud representing the circle.
    """
    print(f"center: {center}")
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = 0 #center[2]
        points.append([x, y, z])
    
    circle_pcd = o3d.geometry.PointCloud()
    circle_pcd.points = o3d.utility.Vector3dVector(points)
    return circle_pcd

# Combine all circles into one point cloud
all_edge_circles = o3d.geometry.PointCloud()
    
def check_building_proximity(building_coords, point_cloud, threshold):
    """
    Check if any point of the building is near the point cloud.
    """
    for coord in building_coords:
        if is_point_near_edge(point_cloud, coord, threshold):
            return True
    return False


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

file_path = "/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/2013_05_28_drive_0005_sync_pose2oxts.txt"
xyz_point_clouds, xyz_positions = create_point_clouds_from_xyz(file_path) # Create point clouds from XYZ positions


'''
Circle boundary to see cuttoff of building structures

'''
# radius = 0.0008
# circles = [create_circle(pos, radius) for pos in xyz_positions]  # Adjust radius as needed

# # Combine all circles into one point cloud
# all_circles = o3d.geometry.PointCloud()
# # all_circles += circles[0]
# circle_iter = 1
# for circle in circles:
#     if circle_iter == 10:
#         all_circles += circle
#         circle_iter = 1
#     circle_iter += 1
# all_circles.paint_uniform_color([1, 0, 0])  # Red color for poses

'''
Point cloud with building and unlabeled points

'''
# Path to your PLY file
ply_file_path = '/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/output.ply'  # Replace with your PLY file path

# Read the PLY file
point_cloud = o3d.io.read_point_cloud(ply_file_path)


'''
Filter buildings within bounds near pose path

'''
min_vert_list = []
def building_within_bounds(building_vertex, threshold):
    vert_dist_arr = []
    building_vertex = np.array(building_vertex)
    print(f"building_vertex[0]: {building_vertex[0]}")
    for pos in xyz_positions:
        # pos[0] = pos[1]
        # pos[1] = pos[0]
        vert_dist = np.sqrt((pos[1] - building_vertex[0])*(pos[1] - building_vertex[0])+(pos[0] - building_vertex[1])*(pos[0] - building_vertex[1]))
        # print(f"pos: {pos[:2]}")
        # print(f"building_vertex: {building_vertex}")
        # vert_dist = np.linalg.norm(building_vertex - pos[:2])
        # print(f"vert dist: {vert_dist}")
        vert_dist_arr.append(vert_dist)
    min_vert_dist = np.min(vert_dist_arr)
    min_vert_list.append(min_vert_dist)
    print(f"min vert dist: {min_vert_dist}")
    return min_vert_dist <= threshold

osm_file_path = '/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/map_0005.osm'

# Filter features for buildings
buildings = ox.features_from_xml(osm_file_path, tags={'building': True})
print(f"len(buildings: {len(buildings)})")
building_lines = []
iter = 1
for _, building in buildings.iterrows():
    print(f"building number: {iter}")
    iter += 1
    if building.geometry.geom_type == 'Polygon':
        exterior_coords = building.geometry.exterior.coords
        # Check if first building vertex is within path
        if building_within_bounds(exterior_coords[0], 0.0008): 
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][1], exterior_coords[i][0], 0]
                end_point = [exterior_coords[i + 1][1], exterior_coords[i + 1][0], 0]
                building_lines.append([start_point, end_point])

building_line_set = o3d.geometry.LineSet()
building_points = [point for line in building_lines for point in line]
building_lines_idx = [[i, i + 1] for i in range(0, len(building_points), 2)]
building_line_set.points = o3d.utility.Vector3dVector(building_points)
building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings


'''
Last step: Check if building is hit by point

Save these buildings in a file, as they will be later accessed.

They will be indexed and for each time a building is hit by a point during mapping, save the pc for that building as 
an incrementing list. 

Initially choose some arbitrary building from the list that will be saved below.
'''
# Process Buildings as LineSets
# Parameters
num_points_per_edge = 10  # Adjust this based on desired density
threshold_distance = 0.0001  # Adjust this based on your specific requirements
building_lines = []
iter = 1
for _, building in buildings.iterrows():
    print(f"building number: {iter}")
    iter += 1
    # if iter > 1:
    #     break
    if building.geometry.geom_type == 'Polygon':
        exterior_coords = building.geometry.exterior.coords
        # print(f"exterior coords: {exterior_coords[:]}")
        for coord in building_lines:
            print(f"Coord: {coord}")
            coord = np.array(coord)
            edge_points = np.linspace(coord[0], coord[-1], num_points_per_edge, axis=0)
            print(f"edge_points: {edge_points}")
            radius = 0.0008
            circles = [create_circle(edge_point, radius) for edge_point in edge_points]  # Adjust radius as needed

            # all_circles += circles[0]
            circle_iter = 1
            for circle in circles:
                if circle_iter == 10:
                    all_edge_circles += circle
                    circle_iter = 1
                circle_iter += 1
            all_edge_circles.paint_uniform_color([1, 0, 0])  # Red color for poses
        if True: check_building_proximity(exterior_coords, point_cloud, threshold_distance):
            print(" building is near")
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][1], exterior_coords[i][0], 0]
                end_point = [exterior_coords[i + 1][1], exterior_coords[i + 1][0], 0]
                building_lines.append([start_point, end_point])
        break
building_line_set = o3d.geometry.LineSet()
building_points = [point for line in building_lines for point in line]
building_lines_idx = [[i, i + 1] for i in range(0, len(building_points), 2)]
building_line_set.points = o3d.utility.Vector3dVector(building_points)
building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings

# Visualize the point cloud with both sets of building lines
# o3d.visualization.draw_geometries([building_line_set, xyz_point_clouds, all_circles])
o3d.visualization.draw_geometries([building_line_set, all_edge_circles])
print(np.mean(min_vert_list))