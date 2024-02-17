"""
By: Doncey Albin

- Use building points to get hit_building_list & Only use buildings that have been hit on more than 3 times
- Use only edges that accum points lie on for each building
- Remove interior edges from buildings

"""

import argparse
import os
from extract_building_data import extractBuildingData
from datetime import datetime

def main():
    seq_list = [0, 2, 3, 4, 5, 6, 7, 9, 10]
    frame_inc = 10
    for seq in seq_list:
        extractBuildingData(seq, frame_inc)
        curr_time = datetime.now()
        curr_time_str = curr_time.strftime('%Y-%m-%d %H:%M:%S')
        with open('/home/donceykong/kitti_360/kitti360Scripts/completed.txt', 'a') as file:
            file.write(f'\nSequence {seq} completed. Timestamp: {curr_time_str}')

    # # Just for testing
    # extractBuildingData(2, frame_inc)

if __name__=="__main__": 
    main() 