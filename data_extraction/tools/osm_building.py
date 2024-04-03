import osmnx as ox
import numpy as np
from shapely.geometry import Polygon
from itertools import chain

class OSMBuildingEdge:
    def __init__(self, building_line):
        self.edge_vertices = np.array(building_line)
        self.expanded_vertices = []
        self.times_hit = 0

# TODO: Represent OSM buildings in lat-long and XYZ
class OSMBuilding:
    def __init__(self, building_lines, offset_distance = 0.000008):
        self.edges = [OSMBuildingEdge(line) for line in building_lines]
        self.center = np.mean(np.array(building_lines), axis=(0, 1))
        self.scan_num = 0

        self.per_scan_points_dict = dict()

        # Create offset verticies for the building
        self.offset_vertices = self.get_offset_vertices(building_lines, offset_distance)

        # Find furthest offset vertex from building center
        self.max_dist_vertex_from_center = max(np.linalg.norm(np.array(vertex) - self.center) for vertex in self.offset_vertices)

        # Create offset edges for the building
        self.offset_edges = self.get_offset_edges()

    def get_total_accum_obs_points(self):
        """
        returns accumulated observed points of building up to and including the last scan which observed it.
        """
        total_accum_flat = list(chain.from_iterable(self.per_scan_points_dict.values()))
        return np.asarray(total_accum_flat).reshape(-1, 3)
    
    def get_curr_accum_obs_points(self, frame_num):
        """
        Returns accumulated observed points of building up to and including the {frame_num} scan.
        """
        sub_dict = {current_frame: points for current_frame, points in self.per_scan_points_dict.items() if current_frame <= frame_num}
        curr_accum_flat = list(chain.from_iterable(sub_dict.values()))
        return np.asarray(curr_accum_flat).reshape(-1, 3)
    
    def get_curr_obs_points(self, frame_num):
        """
        returns current observed points of building at the {frame_num} scan.
        """
        curr_scan = self.per_scan_points_dict[frame_num]
        return curr_scan

    def set_curr_obs_points(self, frame_num, points):
        self.per_scan_points_dict[frame_num] = np.asarray(points)

    def get_offset_vertices(self, building_lines, offset_distance):
        """
        Create offset verticies for the building
        """
        vertices = [point for edge in building_lines for point in edge]
        polygon = Polygon(vertices)
        offset_polygon = polygon.buffer(offset_distance)
        offset_vertices = list(offset_polygon.exterior.coords)
        return [(float(coord[0]), float(coord[1]), float(0.0)) for coord in offset_vertices]

    def get_offset_edges(self):
        """
        Create offset edges for the building
        """
        offset_edges = []
        for i in range(len(self.offset_vertices) - 1):
            offset_edges.append([self.offset_vertices[i], self.offset_vertices[i + 1]])
        offset_edges.append([self.offset_vertices[-1], self.offset_vertices[0]])  # Close the loop
        return [OSMBuildingEdge(line) for line in offset_edges]