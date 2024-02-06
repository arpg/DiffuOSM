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
#   5.3) Get accum-scan building points for each frame

def main(): 
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # # parser.add_argument('--mode', choices=['rgb', 'semantic', 'instance', 'confidence', 'bbox'], default='semantic',
    # #                             help='The modality to visualize')
    # parser.add_argument('--sequence', type=int, default=5, help='The sequence to visualize')
    # # parser.add_argument('--max_bbox', type=int, default=100,
    # #                             help='The maximum number of bounding boxes to visualize')

    # args = parser.parse_args()
    # extractBuildingData(args.sequence)

    # 0, 2-7, 9, 10
    # for i in range(10):
    #     print(i)
    extractBuildingData(8)
  
if __name__=="__main__": 
    main() 