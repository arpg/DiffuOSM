'''
Brendan Crowe and Doncey Albin

1) Refactor and clean.
    - Remove any 1-indexing found
    - Use args for main()
    - Make it an option to save data (including edges) in xyz, not lat-long.
        - For xyz, only need to convert OSM data to XYZ
        - for lat-long, need to convert each scan to lat-long
        * No matter what, TF of scan needs to happen.

Allow batch processing.
'''

import os
import math
import re
import glob
import numpy as np
import open3d as o3d
from datetime import datetime
from collections import namedtuple

from concurrent.futures import ThreadPoolExecutor

# Internal imports
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *
from tools.point_processor import PointCloudProcessor

class ExtractBuildingData:
    def __init__(self, seq=5, frame_inc=1):
        self.seq = seq
        self.near_path_threshold_latlon = 0.001            # Distance to do initial filter of buildings near path in lat-long
        self.min_edges_hit = 2                      # Example criterion for each building
        self.min_num_points = 1                     # Example criterion for each building
        self.extract_total_accum_in_batches = False # Use batch processing for total_accum_points extraction or not
        self.batch_size = 500                       # If using batch processing for extracting total_accum_points
        self.visualize_total_accum_seq = False      # Visualize total_accum points for buildings over entire sequence after extracting
        self.extraction_method = 'per_frame'        # Data to be saved ('per_frame', 'per_build', or 'both)
        self.use_multithreaded_extraction = False   # Use multithreading for per_frame / per_building point extraction

        self.PCProc = PointCloudProcessor()         # Instantiate PointCloudProcessor

        self.setup_path_variables()
        self.initial_setup(frame_inc)               # Steps 1 & 2

        self.initiate_extraction()
        self.extract_hit_buildings()                # TODO: FINISH . (Step 1)
        self.perform_extraction()                   # TODO: FINISH . (Step 2)
        self.conclude_extraction()

    def initial_setup(self, frame_inc):
        self.inc_frame = frame_inc                                          #
        self.init_frame, self.fin_frame = self.find_min_max_file_names()    #
        self.labels_dict = {label.id: label.color for label in labels}      #
        self.get_velo_poses()                                               #
        self.get_imu_poses_lat_long()                                       # For initial filtering of building points along path

    # TODO: Rename below 'paths' to directories 'dir'
    def setup_path_variables(self):
        self.kitti360Path = os.environ.get('KITTI360_DATASET', os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/KITTI-360'))
        sequence_dir = f'2013_05_28_drive_{self.seq:04d}_sync'

        self.sequence_dir_path = os.path.join(self.kitti360Path, sequence_dir)
        self.raw_pc_path = os.path.join(self.kitti360Path, 'data_3d_raw', sequence_dir, 'velodyne_points', 'data')
        self.semantics_dir_path = os.path.join(self.kitti360Path, 'data_3d_semantics', sequence_dir)
        self.label_path = os.path.join(self.semantics_dir_path, 'labels')
        self.accum_ply_path = os.path.join(self.semantics_dir_path, 'accum_ply') # TODO: Will specify specific file later in loop
        self.imu_poses_file = os.path.join(self.kitti360Path, 'data_poses', sequence_dir, 'poses.txt')
        self.velodyne_poses_file = os.path.join(self.kitti360Path, 'data_poses', sequence_dir, 'velodyne_poses.txt')
        self.oxts_pose_file_path = os.path.join(self.kitti360Path, 'data_poses', sequence_dir, 'poses_latlong.txt')
        self.extracted_building_data_dir = os.path.join(self.kitti360Path, 'data_3d_extracted', sequence_dir, 'buildings')
        self.extracted_per_frame_dir = os.path.join(self.extracted_building_data_dir, 'per_frame')
        self.osm_file_path = os.path.join(self.kitti360Path, 'data_osm', f'map_{self.seq:04d}.osm')

    def initiate_extraction(self):
        self.seq_extraction_begin = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Sequence {self.seq} data extraction beginning. Timestamp: {self.seq_extraction_begin}\n')

    def conclude_extraction(self):
        self.seq_extraction_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Sequence {self.seq} completed. Timestamp: {self.seq_extraction_end}\n')

    def find_min_max_file_names(self):
        """
        """
        pattern = os.path.join(self.label_path, '*.bin')
        files = glob.glob(pattern)
        file_numbers = [int(os.path.basename(file).split('.')[0]) for file in files]
        return min(file_numbers), max(file_numbers) if file_numbers else (None, None)
    
    def get_velo_poses(self):
        """
        """
        if not os.path.exists(self.velodyne_poses_file):
            self.velodyne_poses = get_trans_poses_from_imu_to_velodyne(self.imu_poses_file, self.velodyne_poses_file, save_to_file=True)
        self.velodyne_poses = read_vel_poses(self.velodyne_poses_file) # TODO: Why is read_vel_poses different from read_poses? (see get() in utils/get_transformed_point_cloud -> would like to use read_poses() instead of read_vel_poses())

    def get_imu_poses_lat_long(self):
        '''
        Below is used to get xyz_positions to do initial filter of buildings to be near traveled path.
        self.xyz_positions is used later in filter_and_discretize_building_edges().
        '''
        if not os.path.exists(self.oxts_pose_file_path):
            convert_and_save_oxts_poses(self.imu_poses_file, self.oxts_pose_file_path)
        xyz_point_clouds, self.xyz_positions = get_pointcloud_from_txt(self.oxts_pose_file_path)


    '''
    THIS IS STEP 1

    Needs to find what the total number of accumulated points will be per building for all scans in sequence.

    We could find per-scan points here and per-scan-curr-accum here too, but I imagine we would want to instantiate a per_frame class,
    but then the total overhead of the per_frame_list and hit_building_list could be a lot.

    '''

    def extract_hit_buildings(self):
        """
        """
        hit_buildings_file_path = os.path.join(self.extracted_building_data_dir, 'hit_building_list.pkl') # TODO: Need to create this file

        if not os.path.exists(hit_buildings_file_path):
            self.building_list = get_buildings_near_poses(self.osm_file_path, self.xyz_positions, self.near_path_threshold_latlon)
            self.extract_accumulated_points()                                   # TODO: Complete this method ( line 223 on extract_building_data.py )
            self.filter_and_save_hit_building_list(hit_buildings_file_path)     # DONE
        else:
            self.hit_building_list = load_pkl_data(hit_buildings_file_path)     # DONE

        # self.save_per_build_edges_and_total_accum_points()                             # DONE
        # vis_total_accum_points(self.hit_building_list)                          # DONE
    
    def extract_accumulated_points(self):
        """
        """
        print(f"\nStep 1) Begining extraction of total accum points.\n")
        begin_time = datetime.now()

        if self.extract_total_accum_in_batches:
            print("     - Extracting total accum points per building using batch processing.")
            self.extract_total_accum_points_in_batches()
        else:
            print("     - Extracting total accum points per building using per scan.")
            self.extract_total_accum_points_per_scan()

        elapsed_time = datetime.now() - begin_time
        print(f"    - Finished extraction of total accum points. Elapsed time: {elapsed_time}")
        
    def extract_total_accum_points_per_scan(self):
        """
        This method extracts all of the points that hit buildings over the full sequence. It is done per scan.
        """

        num_frames = len(range(self.init_frame, self.fin_frame + 1, self.inc_frame))
        progress_bar = tqdm(total=num_frames, desc="            ")

        for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame):
            # print(f"    - Frame: {frame_num} / {self.fin_frame}")
            new_pcd = load_and_visualize(self.raw_pc_path, self.label_path, self.velodyne_poses, frame_num, self.labels_dict)

            if new_pcd is not None:
                transformation_matrix = self.velodyne_poses.get(frame_num)
                trans_matrix_oxts = np.asarray(convertPoseToOxts(transformation_matrix))
                pos_latlong = trans_matrix_oxts[:3]
                calc_points_within_build_poly(frame_num, self.building_list, new_pcd, [pos_latlong], self.near_path_threshold_latlon, self.extracted_per_frame_dir)

            progress_bar.update(1)

    # # TODO: Change to for loop using in range(init, batch, final)
    # def extract_total_accum_points_in_batches(self):
    #     '''
    #     This method extracts all of the points that hit buildings over the full sequence. It is done in batches 
    #     of scans, since it may be more efficient to utilize a KDTree with many points than do it per scan.
        
    #     '''
    #     last_batch = False
    #     min_frame = self.init_frame
    #     max_frame = math.ceil(min_frame / self.batch_size) * self.batch_size # Round min_frame up to the nearest multiple of batch size
    #     if max_frame >= self.fin_frame:
    #         max_frame = self.fin_frame
    #         last_batch = True

    #     while True:
    #         self.accum_ply_path = os.path.join(self.semantics_dir_path, 'accum_ply', f'output3D_minframe_{min_frame}_maxframe_{max_frame}_incframe_{self.inc_frame}.ply')

    #         if os.path.exists(self.accum_ply_path):
    #             print(f"    - Ply file for sequence {self.seq} with minframe: {min_frame}, maxframe: {max_frame}, inc: {self.inc_frame} exists! Will be using.")
    #             self.accumulated_color_pc = o3d.io.read_point_cloud(self.accum_ply_path)
    #         else:
    #             print(f"    - Ply file for sequence {self.seq} with minframe: {min_frame}, maxframe: {max_frame}, inc: {self.inc_frame} does not exist. Will be generating it now.")
    #             self.accumulated_color_pc = get_accum_pc(min_frame, max_frame, self.inc_frame, self.raw_pc_path, self.label_path, self.velodyne_poses, self.labels_dict, self.accum_ply_path)
            
    #         pos_latlong_list = get_velo_poses_list(min_frame, max_frame, self.inc_frame, self.velodyne_poses, self.label_path)

    #         if len(self.accumulated_color_pc.points) > 0:
    #             print(f"\n        - Calculating accum points within each building poly.\n")
    #             calc_points_within_build_poly(self.building_list, self.accumulated_color_pc, pos_latlong_list, self.near_path_threshold_latlon, show_prog_bar=True)

    #         if last_batch:
    #             break

    #         min_frame = max_frame + 1
    #         max_frame += self.batch_size
    #         if max_frame >= self.fin_frame:
    #             max_frame = self.fin_frame
    #             last_batch = True

    def extract_total_accum_points_in_batches(self):
        '''
        This method extracts all of the points that hit buildings over the full sequence. It is done in batches 
        of scans, since it may be more efficient to utilize a KDTree with many points than do it per scan.
        
        '''
        for min_frame in range(self.init_frame, self.fin_frame + 1, self.batch_size):
            max_frame = min(min_frame + self.batch_size - 1, self.fin_frame)
            last_batch = max_frame == self.fin_frame

            self.accum_ply_path = os.path.join(self.semantics_dir_path, 'accum_ply', f'output3D_minframe_{min_frame}_maxframe_{max_frame}_incframe_{self.inc_frame}.ply')

            if os.path.exists(self.accum_ply_path):
                print(f"    - Ply file for sequence {self.seq} with minframe: {min_frame}, maxframe: {max_frame}, inc: {self.inc_frame} exists! Will be using.")
                self.accumulated_color_pc = o3d.io.read_point_cloud(self.accum_ply_path)
            else:
                print(f"    - Ply file for sequence {self.seq} with minframe: {min_frame}, maxframe: {max_frame}, inc: {self.inc_frame} does not exist. Will be generating it now.")
                self.accumulated_color_pc = get_accum_pc(min_frame, max_frame, self.inc_frame, self.raw_pc_path, self.label_path, self.velodyne_poses, self.labels_dict, self.accum_ply_path)

            pos_latlong_list = get_velo_poses_list(min_frame, max_frame, self.inc_frame, self.velodyne_poses, self.label_path)

            if len(self.accumulated_color_pc.points) > 0:
                print(f"\n        - Calculating accum points within each building poly.\n")
                calc_points_within_build_poly(None, self.building_list, self.accumulated_color_pc, pos_latlong_list, self.near_path_threshold_latlon, self.extracted_per_frame_dir, show_prog_bar=True)

            if last_batch:
                break

    def save_per_build_edges_and_total_accum_points(self):
        '''
        Save building edges and accumulated scan as np .bin file for each building that is hit by points during seq.
        Use later to pick up off where we find unobs points for each scan.
        '''
        for iter, hit_building in enumerate(self.hit_building_list):
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

    def filter_and_save_hit_building_list(self, hit_buildings_file_path):
        # Filter hit build list and save
        self.hit_building_list = get_building_hit_list(self.building_list, self.min_num_points)

        # Save hit building list
        # save_pkl_data(hit_buildings_file_path, self.hit_building_list)
        
        # Garbage collect
        del self.building_list


    '''
    THIS IS ALL WHERE STEP 2 HAPPENS

    - Per-frame extractions

    '''

    def perform_extraction(self):
        print("\nStep 2) Begining extraction from each frame.")
        for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame):
            self.process_scan(frame_num)

    def process_scan(self, frame_num):
        """
        """
        
        # perscan_obs_points_file = os.path.join(self.extracted_building_data_dir, 'per_frame', f'{frame_num:010d}_obs_points.bin')
        # if os.path.exists(perscan_obs_points_file):
        #     print(f"        --> Found existing extracted points from frame {frame_num} that hit OSM building edges. Will be skipping.")
        # else:
        new_pcd = load_and_visualize(self.raw_pc_path, self.label_path, self.velodyne_poses, frame_num, self.labels_dict)

        if new_pcd is not None:
            if self.use_multithreaded_extraction: # Use executor to submit jobs to be processed in parallel
                with ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_executor: # Initialize the ThreadPoolExecutor with the desired number of workers
                    thread_executor.submit(self.extract_and_save_per_scan_points, new_pcd, frame_num)
            else:
                    self.extract_and_save_per_scan_points(new_pcd, frame_num)

    def extract_and_save_per_scan_points(self, new_pcd_3D, frame_num):
        """
        """

        total_accum_points_frame = []

        frame_obs_points_file = os.path.join(self.extracted_per_frame_dir, f'{frame_num:010d}_obs_points.bin')
        frame_obs_curr_accum_points_file = os.path.join(self.extracted_per_frame_dir, f'{frame_num:010d}_curr_accum_points.bin')
        observed_points_frame = read_building_pc_file(frame_obs_points_file)
        curr_accum_points_frame = read_building_pc_file(frame_obs_curr_accum_points_file)
        
        # Cycle through each building that is in the filtered 'hit' list.
        for hit_building in self.hit_building_list:
            if frame_num in hit_building.scan_list:
                hit_building.total_accum_obs_points = hit_building.curr_accum_obs_points   
                total_accum_points_frame.extend(hit_building.total_accum_obs_points)

        if len(observed_points_frame) > 0:
            unobserved_points_frame = self.PCProc.remove_overlapping_points(total_accum_points_frame, observed_points_frame)
            unobserved_curr_accum_points_frame = self.PCProc.remove_overlapping_points(total_accum_points_frame, curr_accum_points_frame)

            save_per_scan_unobs_data(self.extracted_per_frame_dir, frame_num, total_accum_points_frame, unobserved_points_frame, unobserved_curr_accum_points_frame)
                                           

'''
This is in case we would like to do per frame extraction after getting hit building list with 
more constraints than just having a single point from a scan.
'''
#     def extract_and_save_per_scan_points(self, new_pcd_3D, frame_num):
#         """
#         """

#         points_2D = np.copy(np.asarray(new_pcd_3D.points))
#         points_2D[:, 2] = 0

#         # Preprocess building offset vertices to Shapely Polygon objects
#         hit_building_polygons = [Polygon(hit_building.offset_vertices) for hit_building in self.hit_building_list]
#         point_cloud_2D_kdtree = scipyKDTree(points_2D) # cKDTree
#         points_3d = np.asarray(new_pcd_3D.points)

#         building_edges_frame = []
#         observed_points_frame = []
#         total_accum_points_frame = []
#         curr_accum_points_frame = []

#         transformation_matrix = self.velodyne_poses.get(frame_num)
#         trans_matrix_oxts = np.asarray(convertPoseToOxts(transformation_matrix))
#         pos_latlong = trans_matrix_oxts[:3]
        
#         # Cycle through each building that is in the filtered 'hit' list.
#         for hit_building, hit_building_polygon in zip(self.hit_building_list, hit_building_polygons):
#             distance = np.linalg.norm(pos_latlong[:2] - hit_building.center[:2])
#             if distance <= self.near_path_threshold_latlon:
#                 # Filter points within a threshold distance of the building center using KDTree
#                 indices = point_cloud_2D_kdtree.query_ball_point(hit_building.center, hit_building.max_dist_vertex_from_center)
                
#                 # Convert indices to numpy array
#                 indices = np.array(indices)

#                 # Filter points within the polygon
#                 points_within_polygon = [
#                     points_3d[idx]
#                     for idx in indices
#                     if hit_building_polygon.contains(Point(points_3d[idx, :2]))
#                 ]

#                 hit_building.total_accum_obs_points = hit_building.curr_accum_obs_points   

#                 total_accum_points_frame.extend(hit_building.total_accum_obs_points)
                
#                 update_per_frame_data(hit_building, building_edges_frame, observed_points_frame, curr_accum_points_frame, total_accum_points_frame)

#         if len(observed_points_frame) > 0:
#             unobserved_points_frame = self.PCProc.remove_overlapping_points(total_accum_points_frame, observed_points_frame)
#             unobserved_curr_accum_points_frame = self.PCProc.remove_overlapping_points(total_accum_points_frame, curr_accum_points_frame)

#             save_per_scan_extracted_data(self.extracted_per_frame_dir, frame_num, building_edges_frame, observed_points_frame, curr_accum_points_frame, total_accum_points_frame,
#                                               unobserved_points_frame, unobserved_curr_accum_points_frame)

# def save_per_scan_extracted_data(extracted_per_frame_dir, frame_num, building_edges_frame, observed_points_frame, curr_accum_points_frame, total_accum_points_frame,
#                                 unobserved_points_frame, unobserved_curr_accum_points_frame):
#     """
#     """
    
#     # Save all edges from buildings that were observed in current scan
#     frame_build_edges_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_build_edges.bin')
#     with open(frame_build_edges_file, 'wb') as bin_file:
#         np.array(building_edges_frame).tofile(bin_file)

#     # Save total accumulated points for all buildings that have been observed by current scan
#     frame_totalbuildaccum_scan_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_total_accum_points.bin')
#     with open(frame_totalbuildaccum_scan_file, 'wb') as bin_file:
#         np.array(total_accum_points_frame).tofile(bin_file)
        
#     # Save observed_points_frame
#     frame_obs_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_obs_points.bin')
#     with open(frame_obs_points_file, 'wb') as bin_file:
#         np.array(observed_points_frame).tofile(bin_file)

#     # Save current scan difference from total
#     if len(unobserved_points_frame)>0:
#         frame_unobs_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_unobs_points.bin')
#         with open(frame_unobs_points_file, 'wb') as bin_file:
#             np.array(unobserved_points_frame).tofile(bin_file)

#     # Save the current accumulation of points of buildings that were observed in this scan
#     frame_obs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_curr_accum_points.bin')
#     with open(frame_obs_curr_accum_points_file, 'wb') as bin_file:
#         np.array(curr_accum_points_frame).tofile(bin_file)

#     # Save current accumulated difference from total
#     if len(unobserved_curr_accum_points_frame)>0:
#         frame_unobs_curr_accum_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_unobs_curr_accum_points.bin')
#         with open(frame_unobs_curr_accum_points_file, 'wb') as bin_file:
#             np.array(unobserved_curr_accum_points_frame).tofile(bin_file)