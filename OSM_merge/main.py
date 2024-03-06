"""
By: Doncey Albin

- Save building edges as a pickle file (or bin) and do checking if it exist.
- Use building points to get hit_building_list & Only use buildings that have been hit on more than 2 edges
- Do batches of point accumulations


- Remove interior edges from buildings

"""

import argparse
import os
from extract_building_data import extractBuildingData

def main():
    monitor_file = './extracted_0_2.txt'
    seq_list = [0, 2] #[0, 2, 3, 4, 5, 7, 9, 10]
    frame_inc = 1
    for seq in seq_list:
        extractBuildingData(seq, frame_inc, monitor_file)

    # # Just for testing
    # extractBuildingData(2, frame_inc)

if __name__=="__main__": 
    main() 
