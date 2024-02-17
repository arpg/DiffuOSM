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
    seq_list = [0, 2, 3, 4, 5, 6, 7, 9, 10]
    frame_inc = 100
    # for seq in seq_list:
    #     extractBuildingData(seq, frame_inc)

    # Just for testing
    extractBuildingData(2, frame_inc)

if __name__=="__main__": 
    main() 