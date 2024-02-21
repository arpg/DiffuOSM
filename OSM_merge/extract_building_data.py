'''
By: Doncey Albin

Extract building points for each frame in each sequence, as well as save them.

- Save per building scan/accumscan to KITTI-360/data_3d_extracted/2013_05_28_drive_{sequence}_sync/buildings/per_building/
    - build_{build_num}_scan_{scan_num}.bin
    - build_{build_num}_accumscan.bin
    - build_{build_num}_diffscan_{scan_num}.bin     (build_{build_num}_accumscan.bin - build_{build_num}_scan_{scan_num}.bin)

Save per frame scan of building edges to KITTI-360/data_3d_extracted/2013_05_28_drive_0005_sync/buildings/per_frame/
    - frame_{frame_num}.bin
    - frame_{frame_num}_accumscan.bin
    - frame_{frame_num}_diffscan.bin                (frame_{frame_num}_accumscan.bin - frame_{frame_num}.bin)
'''

import os
import glob
import open3d as o3d
from open3d.visualization import gui
import numpy as np
from collections import namedtuple
# import osmnx as ox
from sklearn.neighbors import KDTree
from scipy.spatial import KDTree

# Internal
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *

def remove_overlapping_points(accum_points, frame_points):
    # Convert lists to numpy arrays for efficient computation
    accum_points_array = np.asarray(accum_points)
    frame_points_array = np.asarray(frame_points)
    
    # Initialize an empty list to hold filtered points
    filtered_frame_points = []
    
    # Use a KDTree for efficient nearest neighbor search
    from scipy.spatial import KDTree
    frame_points_kdtree = KDTree(frame_points_array)
    
    # Query each accum_point in the KDTree of frame_points
    # to check if there's an exact match (distance = 0)
    for accum_point in accum_points_array:
        distance, _ = frame_points_kdtree.query(accum_point)
        if distance > 0:  # Use a small threshold instead of 0 for floating-point precision
            filtered_frame_points.append(accum_point)
    
    return np.array(filtered_frame_points)

def extract_and_save_building_points(new_pcd_3D, hit_building_list, radius, frame_num, extracted_building_data_dir):
    # o3d.visualization.draw_geometries([new_pcd_3D])     # Visualize current scan

    new_pcd_2D = np.copy(np.asarray(new_pcd_3D.points))
    new_pcd_2D[:, 2] = 0

    # Create a new 2D point cloud from the modified points
    point_cloud_2D = o3d.geometry.PointCloud()
    point_cloud_2D.points = o3d.utility.Vector3dVector(new_pcd_2D)

    len_hit_building_list = len(hit_building_list)

    point_cloud_2D_kdtree = KDTree(np.asarray(point_cloud_2D.points))
    masked_points_frame = []
    accum_points_frame = []

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

        # Only if current building is hit by current scan
        if len(masked_points_building) > 0:
            hit_building.scan_num += 1
            # hit_building.points.extend(masked_points_building)

            # Save hit_building.points as .bin file
            building_scan_file = os.path.join(extracted_building_data_dir, 'per_building', f'build_{iter}_scan_{hit_building.scan_num}.bin')
            with open(building_scan_file, 'wb') as bin_file:
                np.array(masked_points_building).tofile(bin_file)

            diff_points_build = remove_overlapping_points(hit_building.accum_points, masked_points_building)
            building_diff_scan_file = os.path.join(extracted_building_data_dir, 'per_building', f'build_{iter}_diffscan_{hit_building.scan_num}.bin')
            with open(building_diff_scan_file, 'wb') as bin_file:
                np.array(diff_points_build).tofile(bin_file)

            masked_points_frame.extend(masked_points_building)
            accum_points_frame.extend(hit_building.accum_points)
    
    if len(masked_points_frame)>0:
        diff_points_frame = remove_overlapping_points(accum_points_frame, masked_points_frame)

    masked_frame_pcd = o3d.geometry.PointCloud()
    accum_frame_pcd = o3d.geometry.PointCloud()
    diff_frame_pcd = o3d.geometry.PointCloud()
    accum_points_frame = np.array(accum_points_frame)
    # accum_points_frame[:, 2] = 0

    # if len(masked_points_frame)>0:
    #     masked_frame_pcd.points = o3d.utility.Vector3dVector(masked_points_frame)
    
    # if len(accum_points_frame)>0:
    #     accum_frame_pcd.points = o3d.utility.Vector3dVector(accum_points_frame)

    # if len(diff_points_frame)>0:
    #     diff_frame_pcd.points = o3d.utility.Vector3dVector(diff_points_frame)

    # masked_frame_pcd.paint_uniform_color([1, 0, 0]) # Red color for frame building points
    # accum_frame_pcd.paint_uniform_color([0, 0, 0])  # Black color for accum frame points
    # diff_frame_pcd.paint_uniform_color([0, 0, 1])   # Blue color for diff frame points

    # o3d.visualization.draw_geometries([accum_frame_pcd, masked_frame_pcd, diff_frame_pcd])

    frame_scan_file = os.path.join(extracted_building_data_dir, 'per_frame', f'{frame_num:010d}.bin')
    with open(frame_scan_file, 'wb') as bin_file:
        np.array(new_pcd_3D.points).tofile(bin_file)

    ## Save masked_points_frame as frame_{frame_num}.bin    f'{frame_num:010d
    frame_build_scan_file = os.path.join(extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_build.bin')
    with open(frame_build_scan_file, 'wb') as bin_file:
        np.array(masked_points_frame).tofile(bin_file)

    ## Save building.points for each building hit by this scan as frame_{frame_num}_accumscan.bin
    frame_buildaccum_scan_file = os.path.join(extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_build_accum.bin')
    with open(frame_buildaccum_scan_file, 'wb') as bin_file:
        np.array(accum_points_frame).tofile(bin_file)

    ## Save difference as frame_{frame_num}_diffscan.bin
    if len(masked_points_frame)>0:
        frame_builddiff_scan_file = os.path.join(extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_build_diff.bin')
        with open(frame_builddiff_scan_file, 'wb') as bin_file:
            np.array(diff_points_frame).tofile(bin_file)

class extractBuildingData():
    # Constructor
    def __init__(self, seq=5, frame_inc=50):

        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..','data/KITTI-360')

        self.seq = seq
        sequence = '2013_05_28_drive_%04d_sync' % self.seq
        self.kitti360Path = kitti360Path
        
        train_test = 'train'
        if (self.seq==8 or self.seq==18): train_test = 'test'

        self.raw_pc_path  = os.path.join(kitti360Path, 'data_3d_raw', sequence, 'velodyne_points', 'data')
        self.label_path = os.path.join(kitti360Path, 'data_3d_semantics', train_test, sequence, 'labels')

        # Used to create accumulated semantic pc (step 2) and extracting building edge points (step 6)
        self.inc_frame = frame_inc

        self.init_frame, self.fin_frame = self.find_min_max_file_names()
        if self.init_frame is not None and self.fin_frame is not None:
            print(f"Minimum file name (as int): {self.init_frame}, Maximum file name (as int): {self.fin_frame}")
        else:
            print("No .bin files found in the specified directory.")

        # 1) Create velodyne poses in world frame
        print("\n\n1) Create velodyne poses in world frame\n    |")
        self.imu_poses_file = os.path.join(kitti360Path, 'data_poses', sequence, 'poses.txt')
        self.velodyne_poses_file = os.path.join(kitti360Path, 'data_poses', sequence, 'velodyne_poses.txt')
        self.velodyne_poses = get_trans_poses_from_imu_to_velodyne(self.imu_poses_file, self.velodyne_poses_file, save_to_file=True)
        self.velodyne_poses = read_vel_poses(self.velodyne_poses_file) # This is okay for now ...
        # TODO: Why is read_vel_poses different from read_poses? 
        # see get() in utils/get_transformed_point_cloud -> would like to use read_poses() instead of read_vel_poses()
        print("     --> Done.\n")

        # 2) Get imu in lat-long frame
        print("\n\n2) Get imu in lat-long frame\n    |")
        # TODO: clean up below
        [ts, poses] = loadPoses(self.imu_poses_file)
        poses = postprocessPoses(poses)
        oxts = convertPoseToOxts(poses) # convert to lat/lon coordinate
        oxts_pose_file_path = os.path.join(kitti360Path, 'data_poses', sequence, 'poses_latlong.txt')
        with open(oxts_pose_file_path, 'w') as f:
            for oxts_ in oxts:
                oxts_ = ' '.join(['%.6f'%x for x in oxts_])
                f.write('%s\n'%oxts_)
        print('         --> Output written to %s' % oxts_pose_file_path)
        xyz_point_clouds, xyz_positions = get_pointcloud_from_txt(oxts_pose_file_path) # Create point clouds from XYZ positions
        print("     --> Done.\n")

        # 3) Filter buildings to be within threshold_dist of path
        print("\n\n3) Filter buildings to be within threshold_dist of path\n    |")
        threshold_dist = 0.0008
        self.radius = threshold_dist*0.01
        osm_file = 'map_%04d.osm' % self.seq
        self.osm_file_path = os.path.join(kitti360Path, 'data_osm', osm_file) 
        self.building_list, building_line_set = get_buildings_near_poses(self.osm_file_path, xyz_positions, threshold_dist)
        print("     --> Done.\n")

        # 4) Remove edges on filtered buildings that are connected to other buildings. These are often just subsets of an entire building structure.

        # 5) Get accumulated points with labels "building" and "unlabeled" in lat-long frame for every 1000 frames
        discretize_all_building_edges(self.building_list, self.num_points_per_edge)
        self.labels_dict = {label.id: label.color for label in labels}         # Create a dictionary for label colors

        min_frame = self.init_frame
        max_frame = 1000
        while True:
            # Get 3D accumulated color pc with labels "building" and "unlabeled"
            self.accum_ply_path = os.path.join(kitti360Path, 'data_3d_semantics', train_test, sequence, 'accum_ply', f'output3D_incframe_{self.inc_frame}_minframe_{min_frame}_maxframe{max_frame}.ply')
            if os.path.exists(self.accum_ply_path):
                print(f"Ply file for sequence {self.seq} with inc {self.inc_frame} exists! Will be using.")
                self.accumulated_color_pc = o3d.io.read_point_cloud(self.accum_ply_path)
            else:
                print(f"Ply file for sequence {self.seq} with inc {self.inc_frame} does not exist. Will be generating it now.")
                self.accumulated_color_pc = get_accum_colored_pc(self.init_frame, self.fin_frame, self.inc_frame, self.raw_pc_path, self.label_path, self.velodyne_poses, self.labels_dict, self.accum_ply_path)

            # Get 2D representation of accumulated_color_pc
            points_2D = np.asarray(np.copy(self.accumulated_color_pc.points))
            points_2D[:, 2] = 0
            self.accumulated_pc_2D = o3d.geometry.PointCloud()
            self.accumulated_pc_2D.points = o3d.utility.Vector3dVector(points_2D)
            self.accumulated_pc_2D.colors = self.accumulated_color_pc.colors

            # TODO: Maybe here would be a good point to do some sort of scan-matching so that the buildings and OSM-polygons are better aligned
            calc_points_on_building_edges(self.building_list, self.accumulated_color_pc, self.accumulated_pc_2D, self.label_path, self.radius)
            # o3d.visualization.draw_geometries([self.accumulated_color_pc, building_line_set])
            # o3d.visualization.draw_geometries([self.accumulated_pc_2D, building_line_set])

            if last_batch:
                break

            min_frame = max_frame + 1
            max_frame += 1000
            if max_frame >= self.fin_frame:
                max_frame = self.fin_frame
                last_batch = True

        # 7) Extract and save points corresponding to OSM building edges
        print("\n\n7) Extract and save points corresponding to OSM building edges\n    |")
        self.extracted_building_data_dir = os.path.join(kitti360Path, 'data_3d_extracted', sequence, 'buildings')
        self.num_points_per_edge = 100
        self.extract_per_frame_building_edge_points()
        print("     --> Done.\n")

    def find_min_max_file_names(self):
        # Pattern to match all .bin files in the directory
        pattern = os.path.join(self.label_path, '*.bin')
        # List all .bin files
        files = glob.glob(pattern)
        # Extract the integer part of the file names
        file_numbers = [int(os.path.basename(file).split('.')[0]) for file in files]
        # Find and return min and max
        if file_numbers:  # Check if list is not empty
            min_file, max_file = min(file_numbers), max(file_numbers)
            return min_file, max_file
        else:
            return None, None
        
    def save_building_edges(self, hit_building_list):
        '''
        Save building edges as np .bin file for each building that is hit by points during seq.
        '''
        for iter, hit_building in enumerate(hit_building_list):
            iter += 1

            building_edges_file = os.path.join(self.extracted_building_data_dir, 'per_building', f'build_{iter}_edges.bin')
            with open(building_edges_file, 'wb') as bin_file:
                np.array(hit_building.edges).tofile(bin_file)

            building_accum_scan_file = os.path.join(self.extracted_building_data_dir, 'per_building', f'build_{iter}_accum.bin')
            with open(building_accum_scan_file, 'wb') as bin_file:
                np.array(hit_building.accum_points).tofile(bin_file)

    def extract_per_frame_building_edge_points(self):
        ''' 
            Do this very last. Sequence accum cloud not needed.
        '''
        min_edge_points = 100
        hit_building_list, hit_building_line_set = get_building_hit_list(self.building_list, min_edge_points)
        self.save_building_edges(hit_building_list)
        # o3d.visualization.draw_geometries([self.accumulated_pc_2D, hit_building_line_set])

        # Send some of the big ass variables to garbage collection
        del self.accumulated_pc_2D
        del self.accumulated_color_pc
        del self.building_list


        ''' 
            Do this very last. Sequence accum cloud not needed.
        '''
        frame_num = self.init_frame
        while True: 
            raw_pc_frame_path = os.path.join(self.raw_pc_path, f'{frame_num:010d}.bin')
            pc_frame_label_path = os.path.join(self.label_path, f'{frame_num:010d}.bin')
            new_pcd = load_and_visualize(raw_pc_frame_path, pc_frame_label_path, self.velodyne_poses, frame_num, self.labels_dict)
            
            if new_pcd is not None:
                extract_and_save_building_points(new_pcd, hit_building_list, self.radius, frame_num, self.extracted_building_data_dir)
                print(f"        --> Extracted points from frame: {frame_num} that hit OSM building edges.")
            frame_num += self.inc_frame

            # Exit the loop if you've processed all frames
            if frame_num > self.fin_frame:  # Define the maximum frame number you want to process
                break