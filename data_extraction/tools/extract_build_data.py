'''
Brendan Crowe and Doncey Albin

Refactor and clean.
    - Make it an option to save data (including edges) in xyz, not lat-long.
        - For xyz, only need to convert OSM data to XYZ
        - for lat-long, need to convert each scan to lat-long
        * No matter what, TF of scan needs to happen.

    - Flip the points and OSM data - they are currently upside down
'''

import os
import glob
import numpy as np
from datetime import datetime, timedelta

from multiprocessing import Manager, Pool
from copy import copy, deepcopy

# Internal imports
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *
from tools.point_processor import PointCloudProcessor

class ExtractBuildingData:
    def __init__(self, seq=5, frame_inc=1):
        self.seq = seq
        self.near_path_threshold_latlon = 0.001     # Distance to do initial filter of buildings near path in lat-long
        self.min_num_points = 1000                  # Example criterion for each building

        self.PCProc = PointCloudProcessor()

        self.setup_path_variables()
        self.initial_setup(frame_inc)

        self.initiate_extraction()
        self.extract_obs_and_accum_obs_points()     # Step 1
        self.save_all_obs_points()                  # Step 2
        # self.extract_and_save_unobs_points()      # Step 3
        self.conclude_extraction()

    def initial_setup(self, frame_inc):
        self.inc_frame = frame_inc                                          #
        self.init_frame, self.fin_frame = self.find_min_max_file_names(self.label_path)    #
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
        print(f'\n\nSequence {self.seq} data extraction beginning. Timestamp: {self.seq_extraction_begin}\n')

    def conclude_extraction(self):
        self.seq_extraction_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'\nSequence {self.seq} completed. Timestamp: {self.seq_extraction_end}\n')

    # TODO: Move to file utils
    def find_min_max_file_names(self, file_dir_path):
        """
        """
        pattern = os.path.join(file_dir_path, '*.bin')
        files = glob.glob(pattern)
        file_numbers = [int(os.path.basename(file).split('.')[0]) for file in files]
        return min(file_numbers), max(file_numbers) if file_numbers else (None, None)
    
    def get_velo_poses(self):
        """
        """
        if not os.path.exists(self.velodyne_poses_file):
            self.velodyne_poses = get_trans_poses_from_imu_to_velodyne(self.imu_poses_file, self.velodyne_poses_file, save_to_file=False)
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
    STEP 1: Extract observed building points from each frame and filter the buildings.
    '''
    def extract_obs_and_accum_obs_points(self):
        print("\n     - Step 1) Extracting observed points from each frame.")

        # Initial filter of OSM buildings via boundary around IMU path in lat-long
        self.building_list = get_buildings_near_poses(self.osm_file_path, self.xyz_positions, self.near_path_threshold_latlon)
        
        # Main per-frame extraction
        self.extract_accumulated_points()

        # Filter hit buildings such that every building has at least one point accumulated
        self.filter_hit_building_list()

        # View, if desired (Not reccomended for inc_frame of 1 on an entire sequence)
        # vis_total_accum_points(self.hit_building_list)

    def extract_accumulated_points(self):
        """
        This method extracts all of the points that hit buildings over the full sequence. It is done per scan.
        """

        # Create batches of frame numbers
        frame_nums = range(self.init_frame, self.fin_frame + 1, self.inc_frame)
        batch_size = 10
        frame_batches = [frame_nums[i:i + batch_size] for i in range(0, len(frame_nums), batch_size)]
        
        # Create a batch list containing frame numbers and a copy of building_list for each batch
        batches = [(frame_batch, copy(self.building_list)) for frame_batch in frame_batches]

        with Pool() as pool:
            # Process each batch in parallel, with tqdm for progress tracking
            with tqdm(total=len(batches), desc="            Processing batches") as pbar:
                # Using `imap_unordered` for asynchronous iteration and progress updates
                results = []
                for result in pool.imap_unordered(self.process_batch, batches):
                    results.append(result)
                    pbar.update(1)  # Update progress bar for each batch processed
        
        # Merge or recombine results from each batch
        with time_block("           - merge_building_lists()"):
            self.building_list = self.merge_building_lists(results)

        # with Manager() as manager:
        #     self.shared_building_list = manager.list(self.building_list)  # Create a managed list
        #     # frame_nums = range(self.init_frame, self.fin_frame + 1, self.inc_frame)
            
        #     # tasks = [(frame_num, shared_building_list) for frame_num in frame_nums]

        #     # with Pool() as pool, tqdm(total=len(frame_nums), desc="Processing frames") as progress_bar:
        #     #     for _ in pool.imap_unordered(self.extract_per_scan_total_accum_obs_points_wrapper, tasks):
        #     #         progress_bar.update(1)
        #     # Creating chunks of frames

        #     frame_nums = range(self.init_frame, self.fin_frame + 1, self.inc_frame)
        #     chunk_size = 1
        #     chunks = [frame_nums[i:i + chunk_size] for i in range(0, len(frame_nums), chunk_size)]

        #     with Pool() as pool, tqdm(total=len(frame_nums), desc="Processing frames") as progress_bar:
        #         for _ in pool.imap_unordered(self.process_chunk, chunks):
        #             # Ensure progress is updated correctly, accounting for potentially smaller last chunk
        #             progress_count = min(chunk_size, len(frame_nums) - progress_bar.n)
        #             progress_bar.update(progress_count)

                    # ********************************
        # # Assuming self.init_frame, self.fin_frame, and self.inc_frame are defined
        # tasks = [(frame_num, self.building_list.copy()) for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame) for _ in range(os.cpu_count())]
        # with Pool() as pool:
        #     with tqdm(total=len(tasks), desc="Processing frames") as progress_bar:
        #         for _ in pool.imap_unordered(self.extract_per_scan_total_accum_obs_points, tasks):
        #             progress_bar.update(1)

                #************
        # # Create a copy of the building_list for each worker process
        # building_lists = self.create_building_list_copies()
        # # Main per-frame extraction using multiprocessing
        # num_frames = len(range(self.init_frame, self.fin_frame + 1, self.inc_frame))
        # with Pool() as pool, tqdm(total=num_frames, desc="            ") as progress_bar:
        #     pool.starmap(self.extract_per_scan_total_accum_obs_points, [(frame_num, building_list) for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame) for building_list in building_lists])
        #     progress_bar.update(num_frames)

        # Merge the hit_building_list from all worker processes
        # self.hit_building_list = get_building_hit_list(sum([bl for bl in building_lists], []), self.min_num_points)

                        # ************************
        # num_frames = len(range(self.init_frame, self.fin_frame + 1, self.inc_frame))
        # progress_bar = tqdm(total=num_frames, desc="            ")
        # for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame):
        #     #total_accum_points_file = os.path.join(self.extracted_per_frame_dir, f'{frame_num:010d}_total_accum_points.bin', )
        #     # Check if the file does not exist
        #     #if not os.path.exists(total_accum_points_file):
        #     self.extract_per_scan_total_accum_obs_points(frame_num)        
        #     progress_bar.update(1)

    def merge_building_lists(self, building_lists):
        merged_buildings = {}

        for sublist in building_lists:
            for building in sublist:
                building_id = building.get_building_id()
                
                if building_id not in merged_buildings:
                    merged_buildings[building_id] = building
                else:
                    self.merge_buildings(merged_buildings[building_id], building)

        return list(merged_buildings.values())

    def merge_buildings(self, building1, building2):
        for frame_num, points in building2.per_scan_points_dict.items():
            if frame_num in building1.per_scan_points_dict:
                combined_points = np.vstack({tuple(row) for row in np.vstack([building1.per_scan_points_dict[frame_num], points])})
                building1.per_scan_points_dict[frame_num] = combined_points
            else:
                building1.per_scan_points_dict[frame_num] = points

    def process_batch(self, batch):
        batch_of_scans, building_list = batch
        # Directly use self.shared_building_list here
        for scan_num in batch_of_scans:
            self.extract_per_scan_total_accum_obs_points(scan_num, building_list)
        return building_list

    def extract_per_scan_total_accum_obs_points(self, frame_num, building_list):
        # The total_accum file for this frame does not exist, extraction will continue
        new_pcd = load_and_visualize(self.raw_pc_path, self.label_path, self.velodyne_poses, frame_num, self.labels_dict)
        if new_pcd is not None:
            transformation_matrix = self.velodyne_poses.get(frame_num)
            trans_matrix_oxts = np.asarray(convertPoseToOxts(transformation_matrix))
            pos_latlong = trans_matrix_oxts[:3]
            calc_points_within_build_poly(frame_num, building_list, new_pcd, pos_latlong, self.near_path_threshold_latlon)

    def filter_hit_building_list(self):
        # Filter building list so only buildings hit are considered
        self.building_list = get_building_hit_list(self.building_list, self.min_num_points)

    '''
    STEP 2: So we dont need to repeat step 1.
    - This allows for us to optimize for faster method for unobserved extraction w/o needing to keep repeating step 1.
    '''
    def save_all_obs_points(self):
        print("\n     - Step 2) Saving observed points from each frame.")
        # Create batches of frame numbers
        frame_nums = range(self.init_frame, self.fin_frame + 1, self.inc_frame)
        batch_size = 2
        frame_batches = [frame_nums[i:i + batch_size] for i in range(0, len(frame_nums), batch_size)]

        with Pool(processes=2) as pool:
            # Process each batch in parallel, with tqdm for progress tracking
            with tqdm(total=len(frame_batches), desc="            Processing batches") as pbar:
                for _ in pool.imap_unordered(self.save_per_scan_obs_points_wrapper, frame_batches):
                    pbar.update(1)  # Update progress bar for each batch processed

        # num_frames = len(range(self.init_frame, self.fin_frame + 1, self.inc_frame))
        # progress_bar = tqdm(total=num_frames, desc="            ")
        # for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame):
        #     pc_frame_label_path = os.path.join(self.label_path, f'{frame_num:010d}.bin')
        #     # Check if the labes file for this scan exist
        #     if os.path.exists(pc_frame_label_path):
        #         self.save_per_scan_obs_points(frame_num)
        #         progress_bar.update(1)

    def save_per_scan_obs_points_wrapper(self, batch_of_scans):
        for scan_num in batch_of_scans:
            self.save_per_scan_obs_points(scan_num)
    
    def save_per_scan_obs_points(self, frame_num):
        """
        """
        # total_accum_points_frame = []
        # observed_points_frame = []
        building_edges_frame = []
        curr_accum_points_frame = []
        unobserved_curr_accum_points_frame = []

        pc_frame_label_path = os.path.join(self.label_path, f'{frame_num:010d}.bin')
        if os.path.exists(pc_frame_label_path):
            for hit_building in self.building_list:
                if frame_num in hit_building.per_scan_points_dict:
                    # Update the building edges for the frame using the building edges
                    building_edges_frame.extend(edge.edge_vertices for edge in hit_building.edges)

                    # Update current frame's points
                    # hit_building_curr_obs_points = hit_building.get_curr_obs_points(frame_num)
                    # observed_points_frame.extend(hit_building_curr_obs_points)

                    hit_building_total_accum_obs_points = hit_building.get_total_accum_obs_points()
                    # total_accum_points_frame.extend(hit_building_total_accum_obs_points)

                    hit_building_curr_accum_obs_points = hit_building.get_curr_accum_obs_points(frame_num)
                    curr_accum_points_frame.extend(hit_building_curr_accum_obs_points)

                    # Only extract unobserved points if there are more total accumulated points than current accumulated points
                    if len(hit_building_total_accum_obs_points) > len(hit_building_curr_accum_obs_points):
                        hit_building_curr_unobs_accum_points = self.PCProc.remove_overlapping_points(hit_building.total_accum_obs_points, hit_building_curr_accum_obs_points)
                        unobserved_curr_accum_points_frame.extend(hit_building_curr_unobs_accum_points)

            if len(unobserved_curr_accum_points_frame) > 0:
                save_per_scan_data(self.extracted_per_frame_dir, frame_num, building_edges_frame, curr_accum_points_frame, unobserved_curr_accum_points_frame)