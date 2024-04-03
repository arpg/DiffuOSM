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
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor

# Internal imports
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *
from tools.point_processor import PointCloudProcessor

class ExtractBuildingData:
    def __init__(self, seq=5, frame_inc=1):
        self.seq = seq
        self.near_path_threshold_latlon = 0.001     # Distance to do initial filter of buildings near path in lat-long
        self.min_num_points = 1                     # Example criterion for each building
        self.use_multithreaded_extraction = False   # Use multithreading for per_frame / per_building point extraction

        self.PCProc = PointCloudProcessor()

        self.setup_path_variables()
        self.initial_setup(frame_inc)

        self.initiate_extraction()
        self.extract_obs_and_accum_obs_points()     # Step 1
        self.extract_total_and_unobs_points()       # Step 2
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
        print(f'\n\nSequence {self.seq} data extraction beginning. Timestamp: {self.seq_extraction_begin}\n')

    def conclude_extraction(self):
        self.seq_extraction_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'\nSequence {self.seq} completed. Timestamp: {self.seq_extraction_end}\n')

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

        num_frames = len(range(self.init_frame, self.fin_frame + 1, self.inc_frame))
        progress_bar = tqdm(total=num_frames, desc="            ")

        for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame):
            new_pcd = load_and_visualize(self.raw_pc_path, self.label_path, self.velodyne_poses, frame_num, self.labels_dict)
            if new_pcd is not None:
                transformation_matrix = self.velodyne_poses.get(frame_num)
                trans_matrix_oxts = np.asarray(convertPoseToOxts(transformation_matrix))
                pos_latlong = trans_matrix_oxts[:3]
                calc_points_within_build_poly(frame_num, self.building_list, new_pcd, [pos_latlong], self.near_path_threshold_latlon, self.extracted_per_frame_dir)
            progress_bar.update(1)

    def filter_hit_building_list(self):
        # Filter hit build list
        self.hit_building_list = get_building_hit_list(self.building_list, self.min_num_points)
        
        # Garbage collect
        del self.building_list

    def extract_total_and_unobs_points(self):
        print("\n     - Step 1) Extracting unobserved points from each frame.")

        num_frames = len(range(self.init_frame, self.fin_frame + 1, self.inc_frame))
        progress_bar = tqdm(total=num_frames, desc="            ")
        for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame):
            self.process_scan(frame_num)
            progress_bar.update(1)

    def process_scan(self, frame_num):
        """
        """

        if self.use_multithreaded_extraction: # Use executor to submit jobs to be processed in parallel
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_executor: # Initialize the ThreadPoolExecutor with the desired number of workers
                thread_executor.submit(self.extract_and_save_per_scan_points, new_pcd, frame_num)
        else:
                self.extract_and_save_per_scan_points(frame_num)

    def extract_and_save_per_scan_points(self, frame_num):
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