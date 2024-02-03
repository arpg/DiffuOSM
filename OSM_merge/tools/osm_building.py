import osmnx as ox
import numpy as np

class OSMBuilding:
    def __init__(self, building_lines):
        self.edges = np.array(building_lines)
        self.expanded_edges = []
        self.scan_num = 0
        self.times_hit = 0
        self.points = []

