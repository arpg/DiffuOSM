"""
By: Doncey Albin

"""

import argparse
import os
from tools.extract_build_data import ExtractBuildingData

def main():
    seq_list = [0, 2, 3, 4, 5, 6, 7, 9, 10]
    frame_inc = 1
    for seq in seq_list:
        ExtractBuildingData(seq, frame_inc)

if __name__=="__main__": 
    main() 
