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

def main(): 
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # # parser.add_argument('--mode', choices=['rgb', 'semantic', 'instance', 'confidence', 'bbox'], default='semantic',
    # #                             help='The modality to visualize')
    # parser.add_argument('--sequence', type=int, default=5, help='The sequence to visualize')
    # # parser.add_argument('--max_bbox', type=int, default=100,
    # #                             help='The maximum number of bounding boxes to visualize')

    # args = parser.parse_args()
    # extractBuildingData(args.sequence)

    # seq_list = [0, 2, 3, 4, 5, 6, 7, 9, 10]
    # for seq in seq_list:
    #     extractBuildingData(8)
    extractBuildingData(5)

if __name__=="__main__": 
    main() 