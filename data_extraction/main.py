"""
By: Doncey Albin

"""

import argparse
import os
from tools.extract_build_data import ExtractBuildingData

def main():
    seq_list = [0] #, 2, 3, 4, 5, 6, 7, 9, 10]
    frame_inc = 1

    for seq in seq_list:
        seq_extract = ExtractBuildingData(seq, frame_inc)
        seq_extract.initiate_extraction()
        seq_extract.extract_obs_and_accum_obs_points()
        seq_extract.save_all_obs_points()
        seq_extract.remove_saved_build_dicts()
        seq_extract.conclude_extraction()

if __name__=="__main__": 
    main() 
