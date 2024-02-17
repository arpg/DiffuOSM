"""
By: Doncey Albin

- Use building points to get hit_building_list & Only use buildings that have been hit on more than 3 times
- Use only edges that accum points lie on for each building
- Remove interior edges from buildings

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
    extractBuildingData(0)

if __name__=="__main__": 
    main() 