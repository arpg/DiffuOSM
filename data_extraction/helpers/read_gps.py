'''
Utitlity for reading GPS data
provides two functions

Example of usuage provided in main thread
'''

import os
from typing import List, Tuple

import osmnx as ox
import matplotlib.pyplot as plt
import glob

# ox.config(log_console=True)

# Function to parse OXTS file
def parse_oxts_file(file_path: str) -> List[Tuple[float, float]]:
    '''
    Parses given OXTS file

    params:
        file_path (str): Path to OXTS file

    returns:
        OXTS_data (List[Tuple[float, float]]): the OXTS data parsed in (lat, lon)
    '''
    gps_coords = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            lat, lon = float(values[0]), float(values[1])
            gps_coords.append((lat, lon))
    return gps_coords

# Function to load all GPS data
def load_gps_data(directory: str) -> List[Tuple[float, float]]:
    '''
    Load all gps files form a given directory.
    Data is in OXTS format

    params:
        directory (str): directory to load GPS data from

    returns:
        all_gps_coords (List[Tuple[float, float]]): all gps data in (lat, lon)
    '''
    all_gps_coords = []
    # for i in range(10)
    for sub_dir in glob.glob(os.path.join(directory, '2013_05_28_drive_0003_sync/oxts/data/')):
        for file_path in glob.glob(os.path.join(sub_dir, '*.txt')):
            gps_coords = parse_oxts_file(file_path)
            all_gps_coords.extend(gps_coords)
    return all_gps_coords


if __name__ == '__main__':
    # Directory containing your .txt files
    directory = '/home/donceykong/kitti_360/kitti360Scripts/data/KITTI-360/data_poses'
    gps_data = load_gps_data(directory)
    print("GPS data read!")

    # Load your local OSM file
    # osm_file = 'karlsruhe-regbez-latest.osm'  # Replace with your OSM file path
    # graph = ox.graph_from_xml(osm_file)
    graph = ox.graph_from_place("Karlsruhe, Germany", network_type="drive")
    print("OSM graph made!")

    # Plot the graph
    fig, ax = ox.plot_graph(graph, show=False, close=False)

    # # Plot the GPS data
    for lat, lon in gps_data:
        ax.scatter(lon, lat, c='red', s=1)  # longitude, latitude

    # Save the figure as a PNG file
    # output_filename = '0002_gps_data.png'
    # plt.savefig(output_filename, bbox_inches='tight')

    # Display the plot
    plt.show()

