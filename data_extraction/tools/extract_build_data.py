'''
Brendan Crowe and Doncey Albin



'''
import os
import glob
import numpy as np
import open3d as o3d
from datetime import datetime
from collections import namedtuple
from scipy.spatial import cKDTree, KDTree as scipyKDTree
from sklearn.neighbors import KDTree as sklearnKDTree

# Internal imports
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *

class ExtractBuildingData:
    def __init__(self, seq=5, frame_inc=1):
        self.setup_path_variables(seq)
        self.initial_setup(seq, frame_inc)      # Steps 1 & 2
        self.extract_hit_buildings(self, seq)   # Step 3 & 4 (Uses batch-processing)
        self.perform_extraction(seq)            # Step 5
        self.conclude_extraction(seq)           # Step 6

    def setup_path_variables(self, seq):
        self.kitti360Path = os.environ.get('KITTI360_DATASET', os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/KITTI-360'))
        self.seq = seq
        sequence_dir = f'2013_05_28_drive_{seq:04d}_sync'
        self.sequence_dir_path = os.path.join(self.kitti360Path, sequence_dir)
        self.raw_pc_path = os.path.join(self.sequence_dir_path, 'velodyne_points', 'data')
        self.semantics_dir_path = os.path.join(self.kitti360Path, 'data_3d_semantics', sequence_dir)
        self.label_path = os.path.join(self.semantics_dir_path, 'labels')
        self.imu_poses_file = os.path.join(self.kitti360Path, 'data_poses', sequence_dir, 'poses.txt')
        self.velodyne_poses_file = os.path.join(self.kitti360Path, 'data_poses', sequence_dir, 'velodyne_poses.txt')
        self.oxts_pose_file_path = os.path.join(self.kitti360Path, 'data_poses', sequence_dir, 'poses_latlong.txt')
        self.extracted_building_data_dir = os.path.join(self.kitti360Path, 'data_3d_extracted', sequence_dir, 'buildings')

    def initial_setup(self, seq, frame_inc):
        self.inc_frame = frame_inc
        self.init_frame, self.fin_frame = self.find_min_max_file_names()
        self.labels_dict = {label.id: label.color for label in labels}
        if not os.path.exists(self.velodyne_poses_file):
            self.velodyne_poses = get_trans_poses_from_imu_to_velodyne(self.imu_poses_file, self.velodyne_poses_file, save_to_file=True)
        self.velodyne_poses = read_vel_poses(self.velodyne_poses_file) # TODO: Why is read_vel_poses different from read_poses? (see get() in utils/get_transformed_point_cloud -> would like to use read_poses() instead of read_vel_poses())
        if not os.path.exists(self.oxts_pose_file_path):
            self.convert_and_save_oxts_poses(self.imu_poses_file, self.oxts_pose_file_path)

    def convert_and_save_oxts_poses(self, imu_poses_file, output_path):
        [timestamps, poses] = loadPoses(imu_poses_file)
        poses = postprocessPoses(poses)
        oxts = convertPoseToOxts(poses)  # Convert to lat/lon coordinate
        with open(output_path, 'w') as f:
            for oxts_ in oxts:
                oxts_line = ' '.join(['%.6f' % x for x in oxts_])
                f.write(f'{oxts_line}\n')

    def extract_hit_buildings(self, seq):
        threshold_dist = 0.0008             # Distance to do initial filter of buildings near path
        self.radius = threshold_dist * 0.01 # Distance of points to an edge to be considered a 'hit'
        building_edge_files_path = os.path.join(self.extracted_building_data_dir, 'per_building', 'edges_accum', 'build_1_edges.bin')
        if not os.path.exists(building_edge_files_path):
            self.filter_and_discretize_building_edges(seq)
            self.extract_accumulated_points(seq)
            self.get_and_save_hit_building_edges_and_accum_points(seq)
        else:
            self.load_hit_building_list_from_files()
        self.extract_per_frame_building_edge_points()

    def conclude_extraction(self, seq):
        curr_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Sequence {seq} completed. Timestamp: {curr_time_str}\n')

    def find_min_max_file_names(self):
        pattern = os.path.join(self.label_path, '*.bin')
        files = glob.glob(pattern)
        file_numbers = [int(os.path.basename(file).split('.')[0]) for file in files]
        return min(file_numbers), max(file_numbers) if file_numbers else (None, None)

    def filter_and_discretize_building_edges(self, seq):
        threshold_dist = 0.0008  # TODO: Good threshold found via testing, but should be relative to scale
        self.osm_file_path = os.path.join(self.kitti360Path, 'data_osm', f'map_{seq:04d}.osm')
        xyz_positions = self.load_xyz_positions()  # Placeholder for actual loading of XYZ positions
        self.building_list, building_line_set = get_buildings_near_poses(self.osm_file_path, xyz_positions, threshold_dist)
        discretize_all_building_edges(self.building_list, num_points_per_edge=100)

    def extract_accumulated_points(self, seq):
        # Placeholder for accumulating points with specific labels (Uses the batch-processing)
        self.accumulated_color_pc = self.accumulate_points_with_labels(['building', 'unlabeled'])  # Example functionality

    def get_and_save_hit_building_edges_and_accum_points(self, seq):
        # `save_building_edges_and_accum` saves the processed edges and accumulated points.
        min_edges_hit = 2  # Example criterion
        self.hit_building_list, self.hit_building_line_set = get_building_hit_list(self.building_list, min_edges_hit)
        self.save_building_edges_and_accum()

    def load_hit_building_list_from_files(self):
        # Assuming a function `get_building_hit_list_from_files` that loads a pre-processed list of buildings.
        building_edgeaccum_dir = os.path.join(self.extracted_building_data_dir, 'per_building', 'edges_accum')
        self.hit_building_list, self.hit_building_line_set = get_building_hit_list_from_files(building_edgeaccum_dir)

    def extract_per_frame_building_edge_points(self):
        # Iterate through frames, loading and processing each to extract building edge points.
        for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame):
            self.process_frame_for_building_edges(frame_num)
