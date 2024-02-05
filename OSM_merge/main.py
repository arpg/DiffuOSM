"""
By: Doncey Albin


Refactoring of kitti360scripts and recoverKitti repositories was made in order to create this pipeline.
I couldn't have done it without them.
    - kitti360scripts:
    - recoverKitti:


Preprosses data (A) and extract training data for diffusion-based inpainting (B).

A) Preprosses data and transform it from world -> lat-long, as well as save it.
    A.1) This includes lidar frames for each seq.

B) Extract building points for each frame in each sequence, as well as save them.
    B.1) Do the same for road points later.

"""

import argparse
import os
from extract_building_data import extractBuildingData

# 1) Create vel poses txt for all sequences

# 2) Create all data_oxsx

# 3) Get labels for each frame in each seq

# 4) Get osm for each seq

# 5) Get extracted building points for each building seq
#   5.1) Get extracted building points for each frame
#   5.2) Get accum building points for each frame
#   5.2) Get accum building points for each frame

def get_kitti_data_path():
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    return kitti360Path
            
# def preprocess_and_save_data(seq = '0005'):
#     if seq is None:
#         print("No sequence selected for preprocessing. Please add sequence number in main module.")

# def extract_and_save_building_points(seq = '0005'):
#     if seq is None:
#         print("No sequence selected for extracting training data. Please add sequence number in main module.")

def main(): 
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--mode', choices=['rgb', 'semantic', 'instance', 'confidence', 'bbox'], default='semantic',
    #                             help='The modality to visualize')
    parser.add_argument('--sequence', type=int, default=5, help='The sequence to visualize')
    # parser.add_argument('--max_bbox', type=int, default=100,
    #                             help='The maximum number of bounding boxes to visualize')

    args = parser.parse_args()

    # for i in range(10):
    # extractBuildingData(i)
    extractBuildingData(args.sequence)
  
if __name__=="__main__": 
    main() 