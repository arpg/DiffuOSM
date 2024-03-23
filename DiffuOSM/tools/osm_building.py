import osmnx as ox
import numpy as np

class OSMBuildingEdge:
    def __init__(self, building_line):
        self.edge_vertices = np.array(building_line)
        self.expanded_vertices = []
        self.times_hit = 0

class OSMBuilding:
    def __init__(self, building_lines):
        self.edges = [OSMBuildingEdge(line) for line in building_lines]
        self.scan_num = 0
        self.edges_hit = 0
        self.times_hit = 0
        self.points = []
        self.accum_points = []

