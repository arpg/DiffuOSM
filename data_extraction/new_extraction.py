import argparse
import os
import glob
import open3d as o3d
from open3d.visualization import gui
import numpy as np
from collections import namedtuple
# import osmnx as ox
# from sklearn.neighbors import KDTree as sklearnKDTree
# from scipy.spatial import KDTree as scipyKDTree
# from scipy.spatial import cKDTree
# from datetime import datetime
import math
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor
import pyclipper # For OSM off-setting
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Internal
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *

class extractBuildingData():
    # Constructor
    def __init__(self, seq=0, frame_inc=1):
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

        # Create a dict to store all semantic labels
        self.labels_dict = {label.id: label.color for label in labels}

        # # 1) Create velodyne poses in world frame
        self.imu_poses_file = os.path.join(kitti360Path, 'data_poses', sequence, 'poses.txt')

        # self.velodyne_poses_file = os.path.join(kitti360Path, 'data_poses', sequence, 'velodyne_poses.txt')
        # if not os.path.exists(self.velodyne_poses_file):
        #     # print("\n\n1) Create velodyne poses in world frame\n    |")
        #     self.velodyne_poses = get_trans_poses_from_imu_to_velodyne(self.imu_poses_file, self.velodyne_poses_file, save_to_file=True)
        # self.velodyne_poses = read_vel_poses(self.velodyne_poses_file) # This is okay for now ...
        
        # 2) Get imu in lat-long frame
        oxts_pose_file_path = os.path.join(kitti360Path, 'data_poses', sequence, 'poses_latlong.txt')
        if not os.path.exists(oxts_pose_file_path):
            print("Converting vehicle poses to lat-long.")
            [ts, poses] = loadPoses(self.imu_poses_file)
            poses = postprocessPoses(poses)
            oxts = convertPoseToOxts(poses) # convert to lat/lon coordinate
            # oxts_pose_file_path = os.path.join(kitti360Path, 'data_poses', sequence, 'poses_latlong.txt')
            with open(oxts_pose_file_path, 'w') as f:
                for oxts_ in oxts:
                    oxts_ = ' '.join(['%.6f'%x for x in oxts_])
                    f.write('%s\n'%oxts_)
        xyz_point_clouds, xyz_positions = get_pointcloud_from_txt(oxts_pose_file_path) # Create point clouds from XYZ positions

        # Get correct OSM data (near poses)
        threshold_dist = 0.0008
        self.radius = threshold_dist*0.01
        self.num_points_per_edge = 100

        osm_file = 'map_%04d.osm' % self.seq
        self.osm_file_path = os.path.join(kitti360Path, 'data_osm', osm_file) 
        self.building_list, building_line_set = get_buildings_near_poses(self.osm_file_path, xyz_positions, threshold_dist)

        # vertices = [point for edge in self.building_list[0].edges for point in edge.edge_vertices]
        # print(f"vertices = {vertices}")
        # polygon = Polygon(vertices) # Create a Shapely Polygon object
        # polygon = polygon.buffer(0)

        # offset_polygon = polygon.buffer(distance=self.radius*2, resolution=100) 
        # # offset_polygon = polygon.buffer(distance=self.radius*2) # Perform offset operation
        # offset_exterior_coords = np.array(offset_polygon.exterior.coords) # Extract the exterior coordinates of the offset polygon
        # zeros_column = np.zeros((offset_exterior_coords.shape[0], 1))
        # offset_exterior_coords_with_z = np.hstack((offset_exterior_coords, zeros_column))
        # print(f"Solution: {offset_exterior_coords_with_z}")

        # Extract vertices from edges and scale them
        scale_factor = 9999999999 #6378137  # Choose a suitable scale factor
        scaled_vertices = [[int(point[0] * scale_factor), int(point[1] * scale_factor)] 
                        for edge in self.building_list[0].edges 
                        for point in edge.edge_vertices]
        print(f"Scaled vertices = {scaled_vertices}")

        # Create a PyClipper Clipper object
        # pc = pyclipper.Pyclipper()
        pc = pyclipper.PyclipperOffset()
        pc.AddPath(scaled_vertices, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # pc.AddPath(scaled_vertices, pyclipper.PT_SUBJECT, True)
        offset_distance =  99999999 #self.radius * 10  # Adjust as needed
        solution = pc.Execute(offset_distance)

        # Convert the solution back to original scale
        # offset_exterior_coords = [[x / scale_factor, y / scale_factor] for x, y in solution[0]]
        offset_exterior_coords = [[x, y, 0] for x, y in solution[0]]
        # Print or use the offset exterior coordinates
        print(f"Solution: {offset_exterior_coords}")

        building_offset_line_set = o3d.geometry.LineSet()
        building_offset_lines_idx = [[i, i + 1] for i in range(0, len(offset_exterior_coords) - 1, 2)]
        building_offset_line_set.points = o3d.utility.Vector3dVector(offset_exterior_coords)
        building_offset_line_set.lines = o3d.utility.Vector2iVector(building_offset_lines_idx)
        building_offset_line_set.paint_uniform_color([1, 0, 0])  # Red color for building offset

        # subj = ((180, 200), (260, 200), (260, 150), (180, 150))
        # print(f"subj = {subj}")
        # pco = pyclipper.PyclipperOffset()
        # pco.AddPath(subj, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # solution = pco.Execute(-self.radius)
        # print(f"Solution: {solution}")

        # building_line_set = building_list_to_o3d_lineset([self.building_list])
        o3d.visualization.draw_geometries([building_line_set])

        # discretize_all_building_edges(self.building_list, self.num_points_per_edge)

        # accum_points = []
        # for building in self.hit_building_list:
        #     build_accum_leveled = building.accum_points
        #     build_accum_leveled[:, 2] -= np.min(build_accum_leveled[:, 2])
        #     accum_points.extend(build_accum_leveled)

        # build_points_accum = np.array(accum_points).reshape(-1, 3)
        # accum_frame_pcd = o3d.geometry.PointCloud()
        # accum_frame_pcd.points = o3d.utility.Vector3dVector(build_points_accum)
        # accum_frame_pcd.paint_uniform_color([0, 1, 0.5])
        # o3d.visualization.draw_geometries([self.hit_building_line_set, accum_frame_pcd])

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
        
def main():
    seq_list = [0]
    frame_inc = 1
    for seq in seq_list:
        extractBuildingData(seq, frame_inc)

if __name__=="__main__": 
    main() 