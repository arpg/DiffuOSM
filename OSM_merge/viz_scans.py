import os
import open3d as o3d
import numpy as np
import time
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









accum_scan_points = []
def get_scan(hit_build_num, scan_number):
    file_name = f"/Users/donceykong/Desktop/kitti360Scripts/OSM_merge/scans_step_1/hitbuilding_{hit_build_num}_scan_{scan_number}.bin"
    scan_points = np.fromfile(file_name)
    scan_points = scan_points.reshape(-1, 3)
    scan_points[:, 2] = 0

    accum_scan_points.extend(scan_points)

    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = o3d.utility.Vector3dVector(scan_points)

    accum_scan_pcd = o3d.geometry.PointCloud()
    accum_scan_pcd.points = o3d.utility.Vector3dVector(np.array(accum_scan_points))

    return scan_pcd, accum_scan_pcd

def get_max_scan_number(hit_build_num):
    MAX_SCAN_NUMBER = 0
    scan_number = 1
    while True:
        file_name = f"/Users/donceykong/Desktop/kitti360Scripts/OSM_merge/scans_step_1/hitbuilding_{hit_build_num}_scan_{scan_number}.bin"
        if not os.path.exists(file_name):
            break
        scan_number += 1
        MAX_SCAN_NUMBER += 1
        
    return MAX_SCAN_NUMBER





hit_build_num = 329 #469
scan_number = 1
MAX_SCAN_NUMBER = get_max_scan_number(hit_build_num)
initial_pcd, accum_scan_pcd = get_scan(hit_build_num, scan_number)

def change_frame(vis, key_code):
    global scan_number
    if key_code == ord('N') and scan_number < MAX_SCAN_NUMBER:
        scan_number += 1
    elif key_code == ord('P') and scan_number > 1:
        scan_number -= 1
    else:
        return False
    new_pcd, accum_scan_pcd = get_scan(hit_build_num, scan_number)
    new_pcd.paint_uniform_color([0, 1, 0])          # Green color for each new scan of building
    accum_scan_pcd.paint_uniform_color([0, 0, 0])   # Black color for accumulated scan of building scans
    if new_pcd:
        vis.clear_geometries()
        vis.add_geometry(hit_building_line_set)
        vis.add_geometry(accum_scan_pcd)
        vis.add_geometry(new_pcd)
        
        lookat_point = np.mean(accum_scan_points, axis=0)
        vis.get_view_control().set_lookat(lookat_point)
        vis.get_view_control().set_front([0, 0, -1])   
    return True

if initial_pcd:
    key_to_callback = {
        ord('N'): lambda vis: change_frame(vis, ord('N')),
        ord('P'): lambda vis: change_frame(vis, ord('P'))
    }
    o3d.visualization.draw_geometries_with_key_callbacks([initial_pcd, accum_scan_pcd, hit_building_line_set], key_to_callback)