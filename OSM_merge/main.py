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
from datetime import datetime

def main():
    monitor_file = './completed.txt'
    seq_list = [0, 2, 3, 4, 5, 6, 7, 9, 10]
    frame_inc = 1
    for seq in seq_list:
        curr_time = datetime.now()
        curr_time_str = curr_time.strftime('%Y-%m-%d %H:%M:%S')
        with open(monitor_file, 'a') as file:
            file.write(f'\nSequence {seq} started. Timestamp: {curr_time_str}')

        # extractBuildingData(seq, frame_inc)

        curr_time = datetime.now()
        curr_time_str = curr_time.strftime('%Y-%m-%d %H:%M:%S')
        with open(monitor_file, 'a') as file:
            file.write(f'\nSequence {seq} completed. Timestamp: {curr_time_str}\n')
            file.write(' ')

    # # Just for testing
    # extractBuildingData(2, frame_inc)

if __name__=="__main__": 
    main() 