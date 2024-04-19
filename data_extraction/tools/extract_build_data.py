'''
Brendan Crowe and Doncey Albin

Refactor and clean.
    - Make it an option to save data (including edges) in xyz, not lat-long.
        - For xyz, only need to convert OSM data to XYZ
        - for lat-long, need to convert each scan to lat-long
        * No matter what, TF of scan needs to happen.
    - Flip the points and OSM data - they are currently upside down
'''

# External
import os
import glob
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from copy import copy

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
        self.ds_voxel_leaf_size = 0.000001          # Voxel size for downsampling per frame

        self.PCProc = PointCloudProcessor()

        self.setup_path_variables()
        self.initial_setup(frame_inc)

    def initial_setup(self, frame_inc):
        self.inc_frame = frame_inc                                          #
        self.init_frame, self.fin_frame = self.find_min_max_file_names(self.label_path)    #
        self.labels_dict = {label.id: label.color for label in labels}      #
        self.get_velo_poses()                                               #
        # self.get_imu_poses_lat_long()                                       # For initial filtering of building points along path

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
        self.velodyne_poses, self.velodyne_poses_latlon = read_vel_poses(self.velodyne_poses_file) # TODO: Why is read_vel_poses different from read_poses? (see get() in utils/get_transformed_point_cloud -> would like to use read_poses() instead of read_vel_poses())
    
    # def get_imu_poses_lat_long(self):
    #     '''
    #     Below is used to get xyz_positions to do initial filter of buildings to be near traveled path.
    #     self.xyz_positions is used later in filter_and_discretize_building_edges().
    #     '''
    #     if not os.path.exists(self.oxts_pose_file_path):
    #         convert_and_save_oxts_poses(self.imu_poses_file, self.oxts_pose_file_path)
    #     xyz_point_clouds, self.xyz_positions = get_pointcloud_from_txt(self.oxts_pose_file_path)

    '''
    STEP 1: Extract observed building points from each frame and filter the buildings.
    '''
    def extract_obs_and_accum_obs_points(self):
        print("\n     - Step 1) Extracting observed points from each frame.")

        # Initial filter of OSM buildings via boundary around IMU path in lat-long
        self.building_list = get_buildings_near_poses(self.osm_file_path, self.velodyne_poses_latlon, self.near_path_threshold_latlon)
        
        # Building point extraction
        path_pattern = os.path.join(self.extracted_per_frame_dir, '*_build_point_dict.npy')
        matching_files = glob.glob(path_pattern)
        if not matching_files:
            # Main per-frame extraction
            self.extract_accumulated_points()

            # Filter hit buildings such that every building has at least one point accumulated
            self.filter_hit_building_list()
        
            # View, if desired (Not reccomended for inc_frame of 1 on an entire sequence)
            # vis_total_accum_points(self.building_list)
            
            # Save all building scan dicts
            print("         - Saving all building scan dicts.")
            self.save_all_building_scan_dicts()
            print("         - done.")
        else:
            print("         - Saved building point dicts found. Filtering building list now.")
            self.building_list = self.filter_hit_building_list_from_saved_dicts()
            print("         - Done.")

    def extract_accumulated_points(self):
        """
        This method extracts all of the points that hit buildings over the full sequence. It is done per scan.
        """

        # Create batches of frame numbers
        frame_nums = range(self.init_frame, self.fin_frame + 1, self.inc_frame)
        batch_size = 100
        frame_batches = [frame_nums[i:i + batch_size] for i in range(0, len(frame_nums), batch_size)]
        
        # TODO: Maybe pass in offset polys as a seperate list!
        # self.building_polygons = [Polygon(building.offset_vertices) for building in building_list]
        # building_list = copy(self.building_list)

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
        print("         - Merging lists now:")
        with time_block("           - merge_building_lists()"):
            self.building_list = self.merge_building_lists(results)

        # del frame_batches
        # del results
        # del batches

        # ************************ No multi-processing *********************************
        # num_frames = len(range(self.init_frame, self.fin_frame + 1, self.inc_frame))
        # progress_bar = tqdm(total=num_frames, desc="            ")
        # for frame_num in range(self.init_frame, self.fin_frame + 1, self.inc_frame):
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
            pos_latlong = self.velodyne_poses_latlon.get(frame_num)[:3]
            calc_points_within_build_poly(frame_num, building_list, new_pcd, pos_latlong, self.near_path_threshold_latlon)

    def filter_hit_building_list(self):
        # Filter building list so only buildings hit are considered
        self.building_list = get_building_hit_list(self.building_list, self.min_num_points)

    def filter_hit_building_list_from_saved_dicts(self):
        # Filter building list so only buildings hit are considered
        filtered_building_list = []
        for build in self.building_list:
            build_points_dict = os.path.join(self.extracted_per_frame_dir, f'{build.id}_build_point_dict.npy')
            if os.path.exists(build_points_dict):
                build.per_scan_points_dict_keys = list(self.get_building_scan_dict(build.id).keys())
                filtered_building_list.append(build)
        return filtered_building_list

    def save_all_building_scan_dicts(self):
        for build in self.building_list:
            build_point_dict = pickle.dumps(build.per_scan_points_dict)
            path = os.path.join(self.extracted_per_frame_dir, f'{build.id}_build_point_dict.npy')
            np.save(path, build_point_dict)
            build.per_scan_points_dict_keys = list(build.per_scan_points_dict.keys())
            build.per_scan_points_dict = None
        
    def get_building_scan_dict(self, build_id):
        # Load the serialized dictionary using numpy.load
        path = os.path.join(self.extracted_per_frame_dir, f'{build_id}_build_point_dict.npy')
        loaded_serialized_dict = np.load(path, allow_pickle=False)

        # Deserialize the dictionary
        per_scan_points_dict = pickle.loads(loaded_serialized_dict)

        return per_scan_points_dict

    def remove_saved_build_dicts(self):
        for build in self.building_list:
            build_points_dict = os.path.join(self.extracted_per_frame_dir, f'{build.id}_build_point_dict.npy')
            if os.path.exists(build_points_dict):
                os.remove(build_points_dict)
    
    '''
    STEP 2: So we dont need to repeat step 1.
    - This allows for us to optimize for faster method for unobserved extraction w/o needing to keep repeating step 1.
    '''
    def save_all_obs_points(self):
        print("\n     - Step 2) Saving observed points from each frame.")
        # Create batches of frame numbers
        frame_nums = range(self.init_frame, self.fin_frame + 1, self.inc_frame)
        batch_size = 50
        frame_batches = [frame_nums[i:i + batch_size] for i in range(0, len(frame_nums), batch_size)]

        with Pool(processes=5) as pool:
            # Process each batch in parallel, with tqdm for progress tracking
            with tqdm(total=len(frame_batches), desc="            Processing batches") as pbar:
                for _ in pool.imap_unordered(self.save_per_scan_obs_points_wrapper, frame_batches):
                    pbar.update(1)  # Update progress bar for each batch processed

        # ************************ No multi-processing *********************************
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
                if frame_num in hit_building.per_scan_points_dict_keys:
                    # Update the building edges for the frame using the building edges
                    building_edges_frame.extend(edge.edge_vertices for edge in hit_building.edges)

                    # Update current frame's points
                    per_scan_points_dict = self.get_building_scan_dict(hit_building.id)
                    # hit_building_curr_obs_points = hit_building.get_curr_obs_points(frame_num)
                    # observed_points_frame.extend(hit_building_curr_obs_points)

                    # hit_building_total_accum_obs_points = hit_building.get_total_accum_obs_points(per_scan_points_dict)
                    # total_accum_points_frame.extend(hit_building_total_accum_obs_points)

                    curr_accum_points_frame.extend(hit_building.get_curr_accum_obs_points(frame_num, per_scan_points_dict))
                    unobserved_curr_accum_points_frame.extend(hit_building.get_curr_accum_unobs_points(frame_num, per_scan_points_dict))

                    # # Only extract unobserved points if there are more total accumulated points than current accumulated points
                    # if len(hit_building_total_accum_obs_points) > len(hit_building_curr_accum_obs_points):
                    #     hit_building_curr_unobs_accum_points = self.PCProc.remove_overlapping_points(hit_building_total_accum_obs_points, hit_building_curr_accum_obs_points)
                    #     unobserved_curr_accum_points_frame.extend(hit_building_curr_unobs_accum_points)
            
            # Downsample frame's points (TODO: This would be better if we ds an accumulation of points and not per-frame)
            curr_accum_points_frame_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curr_accum_points_frame))
            unobs_curr_accum_points_frame_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unobserved_curr_accum_points_frame))

            curr_accum_points_frame = curr_accum_points_frame_pcd.voxel_down_sample(voxel_size=self.ds_voxel_leaf_size).points
            unobserved_curr_accum_points_frame = unobs_curr_accum_points_frame_pcd.voxel_down_sample(voxel_size=self.ds_voxel_leaf_size).points

            # Center DS frame about robot lidar
            pos_latlong = self.velodyne_poses_latlon.get(frame_num)[:3]
            print(f"\nbuild_edges array[0]: {np.asarray(building_edges_frame)[0]}")
            building_edges_frame = np.asarray(building_edges_frame) - pos_latlong
            print("unobs")
            unobserved_curr_accum_points_frame = np.asarray(unobserved_curr_accum_points_frame) - pos_latlong
            print("curr")
            print(f"len_curr: {len(np.asarray(curr_accum_points_frame))}")
            curr_accum_points_frame = np.asarray(curr_accum_points_frame) - pos_latlong
            # Test the mean of the points in this frame
            # print(f"Mean lat: {np.mean(curr_accum_points_frame[:,0])}, Mean lon: {np.mean(curr_accum_points_frame[:,1])}")

            if len(unobserved_curr_accum_points_frame) > 0:
                save_per_scan_data(self.extracted_per_frame_dir, frame_num, building_edges_frame, curr_accum_points_frame, unobserved_curr_accum_points_frame)
