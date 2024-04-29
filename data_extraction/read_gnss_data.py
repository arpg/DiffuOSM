import open3d as o3d
import osmnx as ox
import numpy as np
import re

def get_gnss_data_pcd(file_path, color_array):
    gnss_data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split the line by comma and extract latitude, longitude, and altitude
            parts = line.strip().split(',')
            latitude = float(parts[0].split(':')[1].strip())
            longitude = float(parts[1].split(':')[1].strip())
            altitude = float(parts[2].split(':')[1].strip())

            gnss_data.append([longitude, latitude, 0])

    gnss_data_points_pcd = o3d.geometry.PointCloud()
    gnss_data_points_pcd.points = o3d.utility.Vector3dVector(gnss_data)
    gnss_data_points_pcd.paint_uniform_color(color_array)    # Black color for gnss frame points

    return gnss_data_points_pcd

def get_osm_building_data_pcd(osm_file_path):
    # Filter features for buildings and sidewalks
    buildings = ox.features_from_xml(osm_file_path, tags={'building': True})

    # Process Buildings as LineSets
    building_lines = []
    for _, building in buildings.iterrows():
        if building.geometry.geom_type == 'Polygon':
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
    
    return building_line_set


def convert_OSM_list_to_o3d_rect(osm_list, rgb_color):
    # Initialize the LineSet object
    osm_line_set = o3d.geometry.LineSet()

    # Initialize an empty list to store points and lines
    points = []
    lines = []
    point_index = 0  # This will keep track of the index for points to form lines

    # Function to compute perpendicular vector to a line segment
    def perpendicular_vector(v):
        perp = np.array([-v[1], v[0], 0])
        norm = np.linalg.norm(perp)
        if norm == 0:
            return perp  # To avoid division by zero
        return perp / norm

    # Process each road line
    for road_line in osm_list:
        start_point = np.array(road_line['start_point'])
        end_point = np.array(road_line['end_point'])
        # print(f"road_line['width'] : {road_line['width']}")
        width = np.float64(road_line['width'])/(63781.37/2.0)   # Need to make sure width is not a string and Need to convert the points to latlon

        # Compute the unit vector perpendicular to the line segment
        line_vec = end_point - start_point
        perp_vec = perpendicular_vector(line_vec) * (width / 2)

        # Calculate vertices for the polygon (rectangle)
        v1 = start_point - perp_vec
        v2 = start_point + perp_vec
        v3 = end_point + perp_vec
        v4 = end_point - perp_vec

        # Add points to the list
        points.extend([v1, v2, v3, v4])

        # Add lines to connect these points into a rectangle
        # Connect v1-v2, v2-v3, v3-v4, and v4-v1 to close the rectangle
        lines.extend([
            [point_index, point_index + 1],
            [point_index + 1, point_index + 2],
            [point_index + 2, point_index + 3],
            [point_index + 3, point_index]
        ])
        point_index += 4  # Move to the next set of vertices

    # Convert list of points and lines into Open3D Vector format
    osm_line_set.points = o3d.utility.Vector3dVector(np.array(points))
    osm_line_set.lines = o3d.utility.Vector2iVector(np.array(lines))

    # Paint all lines with the specified color
    osm_line_set.paint_uniform_color(rgb_color)

    return osm_line_set

def convert_OSM_list_to_o3d(osm_list, rgb_color):
    # Initialize the LineSet object
    osm_line_set = o3d.geometry.LineSet()

    # Initialize an empty list to store points
    points = []
    lines = []
    line_idx = 0  # This will keep track of the index for line connections

    # Iterate over each road line in the osm_list
    for road_line in osm_list:
        # Get start and end points from the road line entry
        start_point = road_line['start_point']
        end_point = road_line['end_point']
        # Add the start and end points to the points list
        points.append(start_point)
        points.append(end_point)
        # Append a line connection from start to end point
        lines.append([line_idx, line_idx + 1])
        line_idx += 2  # Increment by 2 because each line uses two new points

    # Convert list of points and lines into Open3D Vector format
    osm_line_set.points = o3d.utility.Vector3dVector(np.array(points))
    osm_line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    # Paint all lines with the specified color
    osm_line_set.paint_uniform_color(rgb_color)
    return osm_line_set

def road_near_pose(road_vertex, pos, threshold):
    road_vertex = np.array(road_vertex)
    vert_dist = np.sqrt((pos[0] - road_vertex[0])*(pos[0] - road_vertex[0])+(pos[1] - road_vertex[1])*(pos[1] - road_vertex[1]))
    return vert_dist <= threshold

default_widths = {
    'path': 3.0,
    'steps': 3.0,
    'cycleway': 3.0,
    'footway': 3.0
}
road_width_lists = {
    'path': [],
    'steps': [],
    'cycleway': [],
    'footway': []
}
def parse_width_to_meters(width_str):
    # Extract feet and inch from the string like "10'0\""
    parts = width_str.split("'")
    feet = int(parts[0])  # Extract feet part
    inches = int(parts[1].replace('"', ''))  # Extract inches part after removing the inch symbol

    # Convert to meters
    meters = feet * 0.3048 + inches * 0.0254
    return meters

import re
def parse_width_to_meters(width_str):
    # Check if the width string matches the expected format (e.g., "10'0\"")
    if re.match(r"^\d+'\d+\"$", width_str):
        # Extract feet and inches from the string like "10'0\""
        parts = width_str.split("'")
        feet = int(parts[0])  # Extract feet part
        inches = int(parts[1].replace('"', ''))  # Extract inches part after removing the inch symbol

        # Convert to meters
        meters = feet * 0.3048 + inches * 0.0254
        return meters
    elif re.match(r"^\d+'\d$", width_str):
        # Extract feet and inches from the string like "10'0\""
        parts = width_str.split("'")
        feet = int(parts[0])  # Extract feet part
        inches = int(parts[1])  # Extract inches part after removing the inch symbol

        # Convert to meters
        meters = feet * 0.3048 + inches * 0.0254
        return meters
    else:
        # If the format is not as expected, return None or raise an error
        return width_str
    
# def update_default_widths(road_type, road_width):
#     print(f"road_width: {road_width}")
#     road_width_lists[f'{road_type}'].append(np.float64(road_width))
#     default_widths[f'{road_type}'] = np.mean(road_width_lists[f'{road_type}'])

def update_default_widths_US(road_type, road_width_str):
    road_width = parse_width_to_meters(road_width_str)
    road_width_lists[f'{road_type}'].append(np.float64(road_width))
    default_widths[f'{road_type}'] = np.mean(road_width_lists[f'{road_type}'])

def get_osm_roads_list_new(osm_file_path):
    # Define tags for querying roads
    tags = {'highway': ['footway', 'path', 'cycleway']}  # This will fetch tertiary and residential roads
    
    # Fetch roads using defined tags
    roads = ox.features_from_xml(osm_file_path, tags=tags)
    
    # Process Roads as LineSets with width
    road_lines = []
    for _, road in roads.iterrows():
        if road.geometry.geom_type == 'LineString':
            road_type = road['highway']

            if 'width' in road:
                if str(road['width']) != 'nan':
                    update_default_widths_US(road_type, road['width'])
            road_width = default_widths.get(road_type, 2.0)
            # print(f"road_type: {road_type}, road_width: {road_width}")

            coords = np.array(road.geometry.xy).T
            for i in range(len(coords) - 1):
                start_point = [coords[i][0], coords[i][1], 0]  # Assuming roads are at ground level (z=0)
                end_point = [coords[i + 1][0], coords[i + 1][1], 0]
                # road_lines.append([start_point, end_point])
                road_lines.append({
                'start_point': start_point,
                'end_point': end_point,
                'width': road_width
                })
    return road_lines

def get_osm_grass_data_pcd(osm_file_path):
    # Filter features for grassland areas
    grasslands = ox.features_from_xml(osm_file_path, tags={'landuse': 'grass'})

    # Process Grassland as LineSets
    grassland_lines = []
    for _, grassland in grasslands.iterrows():
        if grassland.geometry.geom_type == 'Polygon':
            exterior_coords = grassland.geometry.exterior.coords[:-1]  # Skip the last point because it's a repeat of the first
            for i in range(len(exterior_coords)):
                start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                end_point = [exterior_coords[(i + 1) % len(exterior_coords)][0], exterior_coords[(i + 1) % len(exterior_coords)][1], 0]
                grassland_lines.append([start_point, end_point])

    # Create an Open3D line set from the grassland lines
    grassland_line_set = o3d.geometry.LineSet()
    grassland_points = [point for line in grassland_lines for point in line]  # Flatten list of points
    grassland_lines_idx = [[i, i + 1] for i in range(0, len(grassland_points), 2)]
    grassland_line_set.points = o3d.utility.Vector3dVector(grassland_points)
    grassland_line_set.lines = o3d.utility.Vector2iVector(grassland_lines_idx)
    grassland_line_set.paint_uniform_color([0.5, 1, 0.5])  # Light green color for grassland
    return grassland_line_set

if __name__ == "__main__":
    gnss_1_data_file_path = '/Users/donceykong/Desktop/ARPG/projects/spring_2024/DiffuOSM/DiffuOSM/data_extraction/gnss_1_data.txt'
    gnss_2_data_file_path = '/Users/donceykong/Desktop/ARPG/projects/spring_2024/DiffuOSM/DiffuOSM/data_extraction/gnss_2_data.txt'
    osm_file_path = '/Users/donceykong/Desktop/ARPG/projects/spring_2024/DiffuOSM/DiffuOSM/data_extraction/cu_west_campus.osm'
    
    gnss_1_data = get_gnss_data_pcd(gnss_1_data_file_path, color_array= [0, 0, 0])
    gnss_2_data = get_gnss_data_pcd(gnss_2_data_file_path, color_array= [0, 1, 0])
    osm_buildings = get_osm_building_data_pcd(osm_file_path)
    osm_roads_list = get_osm_roads_list_new(osm_file_path)
    osm_roads_o3d = convert_OSM_list_to_o3d(osm_roads_list, [1, 0, 0])
    osm_roads_o3d_rect = convert_OSM_list_to_o3d_rect(osm_roads_list, [1, 0.5, 0])
    grass_land_o3d = get_osm_grass_data_pcd(osm_file_path)

    # Visualize
    o3d.visualization.draw_geometries([gnss_1_data, gnss_2_data, osm_buildings, osm_roads_o3d, osm_roads_o3d_rect, grass_land_o3d])

