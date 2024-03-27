'''
By: Doncey Albin

Extract building points for each frame in each sequence, as well as save them.

Save per-building /accumscan for full sequence and scan/diffscan for each frame to: 
KITTI-360/data_3d_extracted/2013_05_28_drive_{sequence}_sync/buildings/per_building/
    - build_{build_num}_scan_{scan_num}.bin
    - build_{build_num}_accumscan.bin
    - build_{build_num}_diffscan_{scan_num}.bin     (build_{build_num}_accumscan.bin - build_{build_num}_scan_{scan_num}.bin)

- Save per frame scan of building edges to KITTI-360/data_3d_extracted/2013_05_28_drive_{seq}_sync/buildings/per_frame/
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
from sklearn.neighbors import KDTree as sklearnKDTree
from scipy.spatial import KDTree as scipyKDTree
from scipy.spatial import cKDTree
from datetime import datetime
import math
from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor

# Internal
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *

def remove_overlapping_points(accum_points, frame_points):
    # Convert lists to numpy arrays for efficient computation
    accum_points_array = np.asarray(accum_points)
    frame_points_array = np.asarray(frame_points)
    filtered_frame_points = []
    frame_points_kdtree = scipyKDTree(frame_points_array) # Use KD-tree for faster searching
    
    # Query each accum_point in the KDTree of frame_points
    # to check if there's an exact match (distance = 0)
    for accum_point in accum_points_array:
        distance, _ = frame_points_kdtree.query(accum_point)
        if distance > 0:
            filtered_frame_points.append(accum_point)
    return np.array(filtered_frame_points)

def extract_and_save_building_points(new_pcd_3D, hit_building_list, radius, frame_num, extracted_building_data_dir):
    # o3d.visualization.draw_geometries([new_pcd_3D])     # Visualize current scan
    new_pcd_2D = np.copy(np.asarray(new_pcd_3D.points))
    new_pcd_2D[:, 2] = 0

    # Create a new 2D point cloud from the modified points
    point_cloud_2D = o3d.geometry.PointCloud()
    point_cloud_2D.points = o3d.utility.Vector3dVector(new_pcd_2D)

    point_cloud_2D_kdtree = sklearnKDTree(np.asarray(point_cloud_2D.points))
    unobserved_edges_frame = []
    observed_edges_frame = []
    observed_points_frame = []
    accum_points_frame = []
    unobserved_points_frame = []
    for iter, hit_building in enumerate(hit_building_list):
        iter += 1
        masked_points_building = []
        for edge in hit_building.edges:
            edge.times_hit = 0
            for expanded_vertex in edge.expanded_vertices:
                distances, indices = point_cloud_2D_kdtree.query([expanded_vertex])
                # Use a mask to filter 3D points that are within the XY radius from the edge point
                mask = abs(distances) <= radius
                masked_points = np.asarray(new_pcd_3D.points)[indices[mask]]

                if len(masked_points) > 0:
                    masked_points_building.extend(masked_points)
                    edge.times_hit += len(masked_points)
        
        # Only if current building is hit by current scan
        if len(masked_points_building) > 0:
            # Below is per-building extraction

            # Save per-build current scan points
            building_scan_file = os.path.join(extracted_building_data_dir, 'per_building', 'scan', f'build_{iter}_scan_{hit_building.scan_num}.bin')
            with open(building_scan_file, 'wb') as bin_file:
                np.array(masked_points_building).tofile(bin_file)

            # Save per-build diff points
            diff_points_build = remove_overlapping_points(hit_building.accum_points, masked_points_building)
            if len(diff_points_build) > 0:
                building_diff_scan_file = os.path.join(extracted_building_data_dir, 'per_building', 'diffscan', f'build_{iter}_diffscan_{hit_building.scan_num}.bin')
                with open(building_diff_scan_file, 'wb') as bin_file:
                    np.array(diff_points_build).tofile(bin_file)

            for edge in hit_building.edges:
                if (edge.times_hit > 0): 
                    observed_edges_frame.extend(edge.edge_vertices)
                else:
                    unobserved_edges_frame.extend(edge.edge_vertices)

            observed_points_frame.extend(masked_points_building)
            accum_points_frame.extend(hit_building.accum_points)
    
    if len(observed_points_frame)>0:
        unobserved_points_frame = remove_overlapping_points(accum_points_frame, observed_points_frame)

        ## Save accumulated points for each building hit by this scan as {frame_num}_accumulted_points.bin
        frame_buildaccum_scan_file = os.path.join(extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_accum_points.bin')
        with open(frame_buildaccum_scan_file, 'wb') as bin_file:
            np.array(accum_points_frame).tofile(bin_file)

        ## Save observed_edges as frame_{frame_num}.bin    f'{frame_num:010d
        frame_obs_edges_file = os.path.join(extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_obs_edges.bin')
        with open(frame_obs_edges_file, 'wb') as bin_file:
            np.array(observed_edges_frame).tofile(bin_file)

        ## Save observed_edges as frame_{frame_num}.bin    f'{frame_num:010d
        frame_unobs_edges_file = os.path.join(extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_unobs_edges.bin')
        with open(frame_unobs_edges_file, 'wb') as bin_file:
            np.array(unobserved_edges_frame).tofile(bin_file)

        ## Save observed_points_frame as frame_{frame_num}.bin    f'{frame_num:010d
        frame_obs_points_file = os.path.join(extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_obs_points.bin')
        with open(frame_obs_points_file, 'wb') as bin_file:
            np.array(observed_points_frame).tofile(bin_file)

        ## Save difference as frame_{frame_num}_diffscan.bin
        # TODO: check why using 'masked_points_frame' frame below results in err - hopefully using 'unobserved_points_frame' fixes things.
        if len(unobserved_points_frame)>0:
            frame_unobs_points_file = os.path.join(extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_unobs_points.bin')
            with open(frame_unobs_points_file, 'wb') as bin_file:
                np.array(unobserved_points_frame).tofile(bin_file)

def extract_and_save_wrapper(frame_num, new_pcd, hit_building_list, radius, extracted_building_data_dir):
    """
    Wrapper function for extract_and_save_building_points to be used with threading.
    """
    print(f"        --> Extracting points from frame {frame_num}.")
    extract_and_save_building_points(new_pcd, hit_building_list, radius, frame_num, extracted_building_data_dir)
    print(f"        --> Extracted points from frame: {frame_num} that hit OSM building edges.")

class extractBuildingData():
    def __init__(self, seq=5, frame_inc=1):

        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..','data/KITTI-360')

        self.seq = seq
        sequence_dir = '2013_05_28_drive_%04d_sync' % self.seq
        self.kitti360Path = kitti360Path

        # File Paths
        self.raw_pc_path  = os.path.join(kitti360Path, 'data_3d_raw', sequence_dir, 'velodyne_points', 'data')  #
        self.semantics_dir_path = os.path.join(kitti360Path, 'data_3d_semantics', sequence_dir)                 #
        self.label_path = os.path.join(self.semantics_dir_path, 'labels')                                       #
        self.imu_poses_file = os.path.join(kitti360Path, 'data_poses', sequence_dir, 'poses.txt')               # Step 1
        self.velodyne_poses_file = os.path.join(kitti360Path, 'data_poses', sequence_dir, 'velodyne_poses.txt') # Step 1
        oxts_pose_file_path = os.path.join(kitti360Path, 'data_poses', sequence_dir, 'poses_latlong.txt')       # Step 2
        self.extracted_building_data_dir = os.path.join(kitti360Path, 'data_3d_extracted', sequence_dir, 'buildings')   # Step 3
        building_edge_files = os.path.join(self.extracted_building_data_dir, 'per_building', 'edges_accum', f'build_1_edges.bin') # Step 3

        # Used in Step 2, Step 4, and the final step, Step 6
        self.inc_frame = frame_inc
        self.init_frame, self.fin_frame = self.find_min_max_file_names()
        print(f"self.init_frame, self.fin_frame: {self.init_frame}, {self.fin_frame}")
        
        # Create a dict to store all semantic labels
        self.labels_dict = {label.id: label.color for label in labels}         # Create a dictionary for label colors

        # 1) Create velodyne poses in world frame
        if not os.path.exists(self.velodyne_poses_file):
            self.velodyne_poses = get_trans_poses_from_imu_to_velodyne(self.imu_poses_file, self.velodyne_poses_file, save_to_file=True)
            print(f'   Step 1) Created velodyne poses in world frame.\n')
        else:
            print(f'   Step 1) File for Velodyne poses in world frame exists. Will be using.\n')
        self.velodyne_poses = read_vel_poses(self.velodyne_poses_file) # This is okay for now ...
        # TODO: Why is read_vel_poses different from read_poses? 
        # see get() in utils/get_transformed_point_cloud -> would like to use read_poses() instead of read_vel_poses()
        
        # 2) Get imu in lat-long frame
        # TODO: clean up below
        if not os.path.exists(oxts_pose_file_path):
            [ts, poses] = loadPoses(self.imu_poses_file)
            poses = postprocessPoses(poses)
            oxts = convertPoseToOxts(poses) # convert to lat/lon coordinate
            with open(oxts_pose_file_path, 'w') as f:
                for oxts_ in oxts:
                    oxts_ = ' '.join(['%.6f'%x for x in oxts_])
                    f.write('%s\n'%oxts_)
        xyz_point_clouds, xyz_positions = get_pointcloud_from_txt(oxts_pose_file_path) # Create point clouds from XYZ positions:
        print(f'   Step 2) Transformed imu to lat-long frame.\n')

        # 3) Get list of all buildings hit have been hit during the sequence to reduce overall search per scan
        # If hit build egdes & accum points files exist for this seq, then use those and skip steps 1-4.
        threshold_dist = 0.0008
        self.radius = threshold_dist*0.01
        if not os.path.exists(building_edge_files): # Only do extractions if below has not yet been done
            # 3) Filter buildings to be within threshold_dist of path
            print(f'   Step 3) Filtering buildings to be within threshold_dist of path and discretizing their edges.\n')
            osm_file = 'map_%04d.osm' % self.seq
            self.osm_file_path = os.path.join(kitti360Path, 'data_osm', osm_file) 
            self.building_list, building_line_set = get_buildings_near_poses(self.osm_file_path, xyz_positions, threshold_dist)
            print(f'       - Resulted in {len(self.building_list)} buildings to be within threshold_dist of path.\n')

            self.num_points_per_edge = 100
            discretize_all_building_edges(self.building_list, self.num_points_per_edge)   

            # 4) Getting hit building list and saving hit build egdes & accum points
            #   4.1) Filter buildings to be within threshold_dist of path
            print(f'       4.1) Getting accumulated points with labels "building" and "unlabeled" in lat-long frame for every 500 frames.\n')

            last_batch = False
            min_frame = self.init_frame
            batch_size = 500
            max_frame = math.ceil(min_frame / batch_size) * batch_size # Round min_frame up to the nearest multiple of batch size
            if max_frame >= self.fin_frame:
                max_frame = self.fin_frame
                last_batch = True

            while True:
                print(f'       - min_frame = {min_frame}, max_frame: {max_frame}')
                curr_time = datetime.now()
                curr_time_str = curr_time.strftime('%Y-%m-%d %H:%M:%S')
                print(f', begin: {curr_time_str}')

                # Get 3D accumulated color pc with labels "building" and "unlabeled"
                self.accum_ply_path = os.path.join(self.semantics_dir_path, 'accum_ply', f'output3D_minframe_{min_frame}_maxframe_{max_frame}_incframe_{self.inc_frame}.ply')
                if os.path.exists(self.accum_ply_path):
                    print(f"Ply file for sequence {self.seq} with minframe: {min_frame}, maxframe: {max_frame}, inc: {self.inc_frame} exists! Will be using.")
                    self.accumulated_color_pc = o3d.io.read_point_cloud(self.accum_ply_path)
                else:
                    print(f"Ply file for sequence {self.seq} with minframe: {min_frame}, maxframe: {max_frame}, inc: {self.inc_frame} does not exist. Will be generating it now.")
                    self.accumulated_color_pc = get_accum_colored_pc(min_frame, max_frame, self.inc_frame, self.raw_pc_path, self.label_path, self.velodyne_poses, self.labels_dict, self.accum_ply_path)
                
                if len(self.accumulated_color_pc.points)>0:
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

                    curr_time = datetime.now()
                    curr_time_str = curr_time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f', finish: {curr_time_str}\n')
                
                if last_batch:
                    break

                min_frame = max_frame + 1
                max_frame += batch_size
                if max_frame >= self.fin_frame:
                    max_frame = self.fin_frame
                    last_batch = True

            #   4.2) Getting hit building list and saving hit build egdes & accum points:
            print(f'       Step 4.2) Getting hit building list and saving hit build egdes & accum points.\n')
                
            min_edges_hit = 2 # TODO: Maybe this metric should be in the file name?
            self.hit_building_list, self.hit_building_line_set = get_building_hit_list(self.building_list, min_edges_hit)
            self.save_building_edges_and_accum()
            # o3d.visualization.draw_geometries([self.accumulated_pc_2D, hit_building_line_set])

            # # TODO: Send some of the big-ass variables to garbage collection (Necessary?)
            # del self.accumulated_pc_2D
            # del self.accumulated_color_pc
            # del self.building_list
        else:
            print("not os.path.exists(building_edge_files): false")
            building_edgeaccum_dir = os.path.join(self.extracted_building_data_dir, 'per_building', 'edges_accum')
            self.hit_building_list, self.hit_building_line_set = get_building_hit_list_from_files(building_edgeaccum_dir)
            self.num_points_per_edge = 100
            discretize_all_building_edges(self.hit_building_list, self.num_points_per_edge)

        # 5) Extract and save points corresponding to OSM building edges
        print(f'   Step 5) Extracting and saving per-scan points corresponding to OSM building edges.\n')
        self.extract_per_frame_building_edge_points()

        # 6) Extraction complete for sequence
        curr_time = datetime.now()
        curr_time_str = curr_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'Sequence {seq} completed. Timestamp: {curr_time_str}\n')

    def find_min_max_file_names(self):
        # Pattern to match all .bin files in the directory
        pattern = os.path.join(self.label_path, '*.bin')
        files = glob.glob(pattern)
        file_numbers = [int(os.path.basename(file).split('.')[0]) for file in files] # integer part of the file names
        # Find and return min and max
        if file_numbers:  # Check if list is not empty
            min_file, max_file = min(file_numbers), max(file_numbers)
            return min_file, max_file
        else:
            return None, None
        
    def save_building_edges_and_accum(self):
        '''
        Save building edges and accumulated scan as np .bin file for each building that is hit by points during seq.
        '''
        for iter, hit_building in enumerate(self.hit_building_list):
            iter += 1
            
            hit_building_edges = []
            for edge in hit_building.edges:
                hit_building_edges.append(edge.edge_vertices)
            hit_building_edges = np.array(hit_building_edges)

            building_edges_file = os.path.join(self.extracted_building_data_dir, 'per_building', 'edges_accum', f'build_{iter}_edges.bin')
            with open(building_edges_file, 'wb') as bin_file:
                np.array(hit_building_edges).tofile(bin_file)

            building_accum_scan_file = os.path.join(self.extracted_building_data_dir, 'per_building', 'edges_accum', f'build_{iter}_accum.bin')
            with open(building_accum_scan_file, 'wb') as bin_file:
                np.array(hit_building.accum_points).tofile(bin_file)

    def extract_per_frame_building_edge_points(self):
        frame_num = self.init_frame
        # with ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_executor: # Initialize the ThreadPoolExecutor with the desired number of workers
        while True:
            frame_build_scan_file = os.path.join(self.extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_obs_points.bin')
            if os.path.exists(frame_build_scan_file):
                print(f"        --> Found existing extracted points from frame: {frame_num} that hit OSM building edges.")
            else:
                raw_pc_frame_path = os.path.join(self.raw_pc_path, f'{frame_num:010d}.bin')
                pc_frame_label_path = os.path.join(self.label_path, f'{frame_num:010d}.bin')
                new_pcd = load_and_visualize(raw_pc_frame_path, pc_frame_label_path, self.velodyne_poses, frame_num, self.labels_dict)

                if new_pcd is not None:
                    extract_and_save_building_points(new_pcd, self.hit_building_list, self.radius, frame_num, self.extracted_building_data_dir)
                    # Use executor to submit jobs to be processed in parallel
                    # thread_executor.submit(extract_and_save_wrapper, frame_num, new_pcd, self.hit_building_list, self.radius, self.extracted_building_data_dir)
            frame_num += self.inc_frame
            if frame_num > self.fin_frame:  # Exit the loop all frames in sequence processed
                break