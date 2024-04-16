'''
Utility for read Open Street Map (OSM) data and extracting the required geometries
'''

import open3d as o3d
import osmnx as ox

if __name__ == '__main__':
    osm_file_path = '/home/donceykong/kitti_360/kitti360Scripts/data/KITTI-360/data_osm/map_0009.osm'

    # Filter features for buildings and sidewalks
    buildings = ox.geometries_from_xml(osm_file_path, tags={'building': True})
    sidewalks = ox.geometries_from_xml(osm_file_path, tags={'highway': 'footway', 'footway': 'sidewalk'})

    # Process Sidewalks
    sidewalk_lines = []
    for _, sidewalk in sidewalks.iterrows():
        if sidewalk.geometry.type == 'LineString':
            x, y = sidewalk.geometry.xy
            for i in range(len(x) - 1):
                sidewalk_lines.append([[x[i], y[i], 0], [x[i + 1], y[i + 1], 0]])

    sidewalk_line_set = o3d.geometry.LineSet()
    sidewalk_points = [point for line in sidewalk_lines for point in line]
    sidewalk_lines_idx = [[i, i + 1] for i in range(0, len(sidewalk_points), 2)]
    sidewalk_line_set.points = o3d.utility.Vector3dVector(sidewalk_points)
    sidewalk_line_set.lines = o3d.utility.Vector2iVector(sidewalk_lines_idx)
    sidewalk_line_set.paint_uniform_color([0, 1, 0])  # Green color for sidewalks

    # Process Buildings as LineSets
    building_lines = []
    for _, building in buildings.iterrows():
        if building.geometry.type == 'Polygon':
            exterior_coords = building.geometry.exterior.coords
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                end_point = [exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0]
                building_lines.append([start_point, end_point])

    building_line_set = o3d.geometry.LineSet()
    building_points = [point for line in building_lines for point in line]
    building_lines_idx = [[i, i + 1] for i in range(0, len(building_points), 2)]
    building_line_set.points = o3d.utility.Vector3dVector(building_points)
    building_line_set.lines = o3d.utility.Vector2iVector(building_lines_idx)
    building_line_set.paint_uniform_color([0, 0, 1])  # Blue color for buildings

    # Visualize
    o3d.visualization.draw_geometries([sidewalk_line_set, building_line_set])

