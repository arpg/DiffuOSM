'''
By: Doncey Albin

Main driver for data extraction code

'''

import argparse
import os

import numpy as np 


from tools.extract_build_data import ExtractBuildingData


if __name__=="__main__":
    parser = argparse.ArgumentParser('Argument parser for main thread of data extrqaaction code')
    parser.add_argument(
        '--seq_min',
        nargs=1,
        default=0,
        required=False,
        help='The minimum index of sequences to parse.',
        type=int,
        dest='seq_min'
    )
    parser.add_argument(
        '--seq_max',
        nargs=1,
        required=True,
        help='Index maximum index of sequences to parse.',
        type=int,
        dest='seq_max'
    )
    parser.add_argument(
        '--frame_inc',
        nargs=1.
        required=True,
        help='placeholder --- I don\'t know what this is yet'
        type=int,
        dest=frame_inc
    )
    parser.parse_args()
    seq_list = np.arange(parser.seq_min, parser.seq_max+1)
    frame_inc = 1
    for seq in seq_list:
        ExtractBuildingData(seq, parser.frame_inc)
