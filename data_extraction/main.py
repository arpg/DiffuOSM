"""
By: Doncey Albin

"""

import argparse
import os
from tools.extract_build_data import ExtractBuildingData

def test_pass_by_ref(this_list):
    new_list = (1, 2)
    this_list.extend(new_list)

def main():
    # seq_list = [0, 2, 3, 4, 5, 6, 7, 9, 10]
    seq_list = [0]
    frame_inc = 1
    for seq in seq_list:
        ExtractBuildingData(seq, frame_inc)

if __name__=="__main__": 
    main() 
