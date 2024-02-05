'''
By: Doncey Albin


Refactoring of kitti360scripts and recoverKitti repositories was made in order to create this pipeline.
I couldn't have done it without them.
    - kitti360scripts:
    - recoverKitti:


Extract building points for each frame in each sequence, as well as save them.

*) Extract points and save each building (from osm) that is actually hit by points.
    --> Save building semantic info?
    --> Make sure to extract complete building, not just subsections (see get_target_osm_building.py)
    --> saved in KITTI360/data_3d_extracted/2013_05_28_drive_0005_sync/hit_building_list.npy


'''

import os
import open3d as o3d
from open3d.visualization import gui
import numpy as np
from collections import namedtuple
# import osmnx as ox

# Internal
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *

def extract_and_save_building_points(new_pcd_3D, hit_building_list, radius):
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
            # hit_building.points.extend(masked_points_building)

            # TODO: remove or comment out below
            masked_building_pcd = o3d.geometry.PointCloud()
            masked_building_pcd.points = o3d.utility.Vector3dVector(masked_points_building)
            o3d.visualization.draw_geometries([masked_building_pcd])

            # Save hit_building.points as .bin file
            # # TODO: Inlcude frame number??????????????????????????????
            # file_name = f"/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360/data_3d_extracted/2013_05_28_drive_0005_sync/buildings/hitbuilding_{iter+1}_scan_{hit_building.scan_num}.bin"
            # with open(file_name, 'wb') as bin_file:
            #     np.array(masked_points_building).tofile(bin_file)



class extractBuildingData(object):
    # Constructor
    def __init__(self, seq=5):

        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')
        
        train_test = 'train'
        if (seq>5): train_test = 'test'

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.kitti360Path = kitti360Path
        self.raw_pc_path  = os.path.join(kitti360Path, 'data_3d_raw', sequence, 'velodyne_points', 'data')

        # 1) Create velodyne poses in world frame
        self.imu_poses_file = os.path.join(kitti360Path, 'data_poses', sequence, 'poses.txt')
        self.velodyne_poses_file = os.path.join(kitti360Path, 'data_poses', sequence, 'velodyne_poses.txt')
        self.velodyne_poses = get_trans_poses_from_imu_to_velodyne(self.imu_poses_file, self.velodyne_poses_file, save_to_file=True)
        self.velodyne_poses = read_vel_poses(self.velodyne_poses_file) # This is okay for now ...
        # TODO: Why is read_vel_poses different from read_poses? 
        # see get() in utils/get_transformed_point_cloud -> would like to use read_poses() instead of read_vel_poses()

        # 2) Get accumulated points with labels "building" and "unlabeled" in lat-long frame
        self.raw_pc_path  = os.path.join(kitti360Path, 'data_3d_raw', sequence, 'velodyne_points', 'data')
        self.label_path = os.path.join(kitti360Path, 'data_3d_semantics', train_test, sequence, 'labels')
        self.labels_dict = {label.id: label.color for label in labels}         # Create a dictionary for label colors
        self.accumulated_color_pc = get_accum_colored_pc(self.raw_pc_path, self.label_path, self.velodyne_poses, self.labels_dict)
        # o3d.visualization.draw_geometries([self.accumulated_color_pc])

        # 3) Get 2D representation of accumulated_color_pc
        accumulated_pc_3D = np.asarray(self.accumulated_color_pc.points)
        points_2D = accumulated_pc_3D.copy()
        points_2D[:, 2] = 0
        self.accumulated_pc_2D = o3d.geometry.PointCloud()
        self.accumulated_pc_2D.points = o3d.utility.Vector3dVector(points_2D)

        # 3) Get imu in lat-long frame
        # TODO: clean up below
        [ts, poses] = loadPoses(self.imu_poses_file)
        poses = postprocessPoses(poses)
        oxts = convertPoseToOxts(poses) # convert to lat/lon coordinate
        oxts_pose_file_path = os.path.join(kitti360Path, 'data_poses', sequence, 'poses.txt')
        with open(oxts_pose_file_path, 'w') as f:
            for oxts_ in oxts:
                oxts_ = ' '.join(['%.6f'%x for x in oxts_])
                f.write('%s\n'%oxts_)
        print('Output written to %s' % oxts_pose_file_path)
        xyz_point_clouds, xyz_positions = get_pointcloud_from_txt(oxts_pose_file_path) # Create point clouds from XYZ positions

        # 5) Filter buildings to be within threshold_dist of path
        threshold_dist = 0.0008
        osm_file = 'map_%04d.osm' % seq
        self.osm_file_path = os.path.join(kitti360Path, 'data_osm', osm_file) 
        self.building_list, building_line_set = get_buildings_near_poses(self.osm_file_path, xyz_positions, threshold_dist)

        # 6) Extract and save points corresponding to OSM building edges
        self.num_points_per_edge = 100
        self.radius = 0.000008
        self.init_frame = 30
        self.inc_frame = 100
        self.fin_frame = 6000
        self.extract_per_frame_building_edge_points()

    def extract_per_frame_building_edge_points(self):
        discretize_all_building_edges(self.building_list, self.num_points_per_edge)
        # TODO: Maybe here would be a good point to do some sort of scan-matching so that the buildings and OSM-polygons are better aligned
        calc_points_on_building_edges(self.building_list, self.accumulated_color_pc, self.accumulated_pc_2D, self.radius)
        hit_building_list, hit_building_line_set = get_building_hit_list(self.building_list)
        frame_num = self.init_frame
        while True:
            raw_pc_frame_path = os.path.join(self.raw_pc_path, f'{frame_num:010d}.bin')
            pc_frame_label_path = os.path.join(self.label_path, f'{frame_num:010d}.bin')
            new_pcd = load_and_visualize(raw_pc_frame_path, pc_frame_label_path, self.velodyne_poses, frame_num, self.labels_dict)
            
            if new_pcd is not None:
                extract_and_save_building_points(new_pcd, hit_building_list, self.radius)
                print(f"Extracted points from frame: {frame_num} that hit OSM building edges.")
            frame_num += self.inc_frame

            # Exit the loop if you've processed all frames
            if frame_num > self.fin_frame:  # Define the maximum frame number you want to process
                break