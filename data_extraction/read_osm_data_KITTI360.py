import open3d as o3d
import osmnx as ox
import numpy as np

import os
import glob

# Internal imports
from tools.labels import labels
from tools.utils import *
from tools.convert_oxts_pose import *
from tools.point_processor import PointCloudProcessor

class ExtractBuildingData:
    def __init__(self, seq=0, frame_inc=1):
        self.seq = seq
        self.near_path_threshold_latlon = 0.001     # Distance to do initial filter of buildings near path in lat-long
        self.min_num_points = 1000                  # Example criterion for each building
        self.ds_voxel_leaf_size = 0.00001           # Voxel size for downsampling per frame

        self.PCProc = PointCloudProcessor()

        self.setup_path_variables()
        self.initial_setup(frame_inc)

    def initial_setup(self, frame_inc):
        self.inc_frame = frame_inc                                          #
        self.init_frame, self.fin_frame = self.find_min_max_file_names(self.label_path)    #
        self.labels_dict = {label.id: label.color for label in labels}      #
        self.get_velo_poses()                                               #

    # TODO: Rename below 'paths' to directories 'dir'
    def setup_path_variables(self):
        self.kitti360Path = os.environ.get('KITTI360_DATASET', os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/KITTI-360'))
        sequence_dir = f'2013_05_28_drive_{self.seq:04d}_sync'

        self.sequence_dir_path = os.path.join(self.kitti360Path, sequence_dir)
        self.raw_pc_path = os.path.join(self.kitti360Path, 'data_3d_raw', sequence_dir, 'velodyne_points', 'data')
        self.semantics_dir_path = os.path.join(self.kitti360Path, 'data_3d_semantics', sequence_dir)
        self.label_path = os.path.join(self.semantics_dir_path, 'labels')
        self.accum_ply_path = os.path.join(self.semantics_dir_path, 'accum_ply') # TODO: Will specify specific file later in loop
        self.imu_poses_file = os.path.join(self.kitti360Path, 'data_poses', sequence_dir, 'poses.txt')
        self.velodyne_poses_file = os.path.join(self.kitti360Path, 'data_poses', sequence_dir, 'velodyne_poses.txt')
        self.oxts_pose_file_path = os.path.join(self.kitti360Path, 'data_poses', sequence_dir, 'poses_latlong.txt')
        self.extracted_building_data_dir = os.path.join(self.kitti360Path, 'data_3d_extracted', sequence_dir, 'buildings')
        self.extracted_per_frame_dir = os.path.join(self.extracted_building_data_dir, 'per_frame')
        self.osm_file_path = os.path.join(self.kitti360Path, 'data_osm', f'map_{self.seq:04d}.osm')

    def find_min_max_file_names(self, file_dir_path):
        """
        """
        pattern = os.path.join(file_dir_path, '*.bin')
        files = glob.glob(pattern)
        file_numbers = [int(os.path.basename(file).split('.')[0]) for file in files]
        return min(file_numbers), max(file_numbers) if file_numbers else (None, None)
    
    def get_velo_poses(self):
        """
        """
        if not os.path.exists(self.velodyne_poses_file):
            self.velodyne_poses = get_trans_poses_from_imu_to_velodyne(self.imu_poses_file, self.velodyne_poses_file, save_to_file=False)
        self.velodyne_poses, self.velodyne_poses_latlon = read_vel_poses(self.velodyne_poses_file) # TODO: Why is read_vel_poses different from read_poses? (see get() in utils/get_transformed_point_cloud -> would like to use read_poses() instead of read_vel_poses())
    
    def get_pointcloud_pcd(self, raw_pc_frame_path, pc_frame_label_path, frame_num):
        pc = read_bin_file(raw_pc_frame_path)
        pc = get_transformed_point_cloud(pc, self.velodyne_poses, frame_num)
        labels_np = read_label_bin_file(pc_frame_label_path)

        # Filter out any points with a z-position below min_building_z_point
        building_label_mask = (labels_np == 7)
        pc_buildings = pc[building_label_mask]
        if (len(pc_buildings) > 0):
            min_building_z_point = np.min(pc_buildings[:, 2])
        else: 
            return None
        z_position_mask = pc[:, 2] >= min_building_z_point
        pc = pc[z_position_mask]
        labels_np = labels_np[z_position_mask]

        # Get RGB for each point
        rgb_np = color_point_cloud(pc, labels_np, self.labels_dict)

        # Reshape pointcloud to fit in convertPointsToOxts function
        pc_reshaped = np.array([np.eye(4) for _ in range(pc.shape[0])])
        pc_reshaped[:, 0:3, 3] = pc[:, :3]

        # Convert to lat-lon-alt
        pc_lla = np.asarray(postprocessPoses(pc_reshaped))
        pc_lla = np.asarray(convertPointsToOxts(pc_lla))

        # Post-process
        # pc_lla[:, 2] = 0
        pc_lla[:, 2] -= np.min(pc_lla[:, 2])
        pc_lla[:, 2] *= 0.00002 # TODO: Remove this and only use for visualization
        
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(pc_lla[:, :3])  # Only use lat, lon, alt for geometry
        colored_pcd.colors = o3d.utility.Vector3dVector(rgb_np) # Set colors

        colored_pcd = colored_pcd.voxel_down_sample(voxel_size=self.ds_voxel_leaf_size)
        return colored_pcd
    
    def get_osm_data_for_pointcloud(self, frame_num):
        """
        """
        building_edges_frame = []
        building_roads_frame = []

        pc_frame_label_path = os.path.join(self.label_path, f'{frame_num:010d}.bin')
        raw_pc_frame_path = os.path.join(self.raw_pc_path, f'{frame_num:010d}.bin')
        if os.path.exists(pc_frame_label_path):
            colored_pcd = self.get_pointcloud_pcd(raw_pc_frame_path, pc_frame_label_path, frame_num)
            pos_latlong = self.velodyne_poses_latlon.get(frame_num)[:3]

            # # Filter buildings and Roads
            # filtered_building_list = get_roads_near_current_pose()
            # filtered_road_list = get_buildings_near_current_pose()

            # for building in filtered_building_list:
            #     for disc_edge in building.disc_edges:
            #         for vert in disc_edge:
            #             if vert near any(pcd.points):
            #                 building_edges_frame.append(disc_edge)
                        
            # for road in filtered_road_list:

            osm_buildings_list = get_osm_buildings_list(self.osm_file_path, pos_latlong, self.near_path_threshold_latlon)
            osm_roads_list = get_osm_roads_list_new(self.osm_file_path, pos_latlong, self.near_path_threshold_latlon)

            # Now only include road and building segements that fall very near labeled road/building points when they are flattened
            # Implement here
            # Use class OSMRoad/OSMBuilding

            osm_buildings_o3d = building_list_to_o3d_lineset(osm_buildings_list)
            osm_roads_o3d = convert_OSM_list_to_o3d(osm_roads_list, [1, 0, 0])
            osm_roads_o3d_rect = convert_OSM_list_to_o3d_rect(osm_roads_list, [1, 0.5, 0])
            o3d.visualization.draw_geometries([colored_pcd, osm_buildings_o3d, osm_roads_o3d, osm_roads_o3d_rect])


def calc_points_within_build_poly(frame_num, building_list, point_cloud_3D, pos_latlong, near_path_threshold):
    # Get 2D representation of accumulated_color_pc
    points_2D = np.asarray(np.copy(point_cloud_3D.points))
    points_2D[:, 2] = 0

    # Preprocess building offset vertices to Shapely Polygon objects
    building_polygons = [Polygon(building.offset_vertices) for building in building_list]
    point_cloud_2D_kdtree = cKDTree(points_2D) # cKDTree
    points_3d = np.asarray(point_cloud_3D.points)
    
    for building, building_polygon in zip(building_list, building_polygons):
        distance = np.linalg.norm(pos_latlong[:2] - building.center[:2])
        if distance <= near_path_threshold:
            # Filter points within a threshold distance of the building center using KDTree
            indices = point_cloud_2D_kdtree.query_ball_point(building.center, building.max_dist_vertex_from_center)
            
            # Convert indices to numpy array
            indices = np.array(indices)

            # Filter points within the polygon
            points_within_polygon = [
                points_3d[idx]
                for idx in indices
                if building_polygon.contains(Point(points_3d[idx, :2]))
            ]
            
            if len(points_within_polygon) > 0:
                building.set_curr_obs_points(frame_num, points_within_polygon)


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

    # Print the data for verification
    for data in gnss_data:
        print(data)

    gnss_data_points_pcd = o3d.geometry.PointCloud()
    gnss_data_points_pcd.points = o3d.utility.Vector3dVector(gnss_data)
    gnss_data_points_pcd.paint_uniform_color(color_array)    # Black color for gnss frame points

    return gnss_data_points_pcd


def building_near_pose(building_vertex, pos, threshold):
    building_vertex = np.array(building_vertex)
    vert_dist = np.sqrt((pos[0] - building_vertex[0])*(pos[0] - building_vertex[0])+(pos[1] - building_vertex[1])*(pos[1] - building_vertex[1]))
    return vert_dist <= threshold

def get_osm_buildings_list(osm_file_path, pos_lat_lon, threshold_dist):
    # Filter features for buildings and sidewalks
    buildings = ox.geometries_from_xml(osm_file_path, tags={'building': True})

    # Process Buildings as LineSets
    building_lines = []
    building_list = []
    building_id = 0
    for _, building in buildings.iterrows():
        if building.geometry.type == 'Polygon':
            exterior_coords = building.geometry.exterior.coords
            build_center = [np.mean(np.array(exterior_coords)[:, 0]), np.mean(np.array(exterior_coords)[:, 1])]
            if building_near_pose(build_center, np.asarray(pos_lat_lon), threshold_dist):
                per_building_lines = []
                for i in range(len(exterior_coords) - 1):
                    start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                    end_point = [exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0]
                    building_lines.append([start_point, end_point])
                    per_building_lines.append([start_point, end_point])
                new_building = osm_building.OSMBuilding(per_building_lines, building_id)
                building_list.append(new_building)
                building_id += 1
    return building_list


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
        width = np.float64(road_line['width'])/(63781.37/2.0)   # TODO: Need to make sure width is not a string and Need to convert the points to latlon

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

# def convert_OSM_list_to_o3d(osm_list, rgb_color):
#     osm_lines = osm_list
#     osm_line_set = o3d.geometry.LineSet()
#     osm_points = [point for line in osm_lines for point in line]
#     osm_lines_idx = [[i, i + 1] for i in range(0, len(osm_points), 2)]
#     osm_line_set.points = o3d.utility.Vector3dVector(osm_points)
#     osm_line_set.lines = o3d.utility.Vector2iVector(osm_lines_idx)
#     osm_line_set.paint_uniform_color(rgb_color)
#     return osm_line_set

def road_near_pose(road_vertex, pos, threshold):
    road_vertex = np.array(road_vertex)
    vert_dist = np.sqrt((pos[0] - road_vertex[0])*(pos[0] - road_vertex[0])+(pos[1] - road_vertex[1])*(pos[1] - road_vertex[1]))
    return vert_dist <= threshold

default_widths = {
    'motorway': 10.0,
    'primary': 7.0,
    'secondary': 5.0,
    'tertiary': 4.0,
    'unclassified': 2.0,
    'residential': 2.0
}
road_width_lists = {
    'motorway': [],
    'primary': [],
    'secondary': [],
    'tertiary': [],
    'unclassified': [],
    'residential': []
}
def update_default_widths(road_type, road_width):
    road_width_lists[f'{road_type}'].append(np.float64(road_width))
    default_widths[f'{road_type}'] = np.mean(road_width_lists[f'{road_type}'])

def get_osm_roads_list_new(osm_file_path, pos_lat_lon, threshold_dist):
    # Define tags for querying roads
    tags = {'highway': ['residential', 'tertiary']}  # This will fetch tertiary and residential roads
    
    # Fetch roads using defined tags
    roads = ox.geometries_from_xml(osm_file_path, tags=tags)
    
    # Process Roads as LineSets with width
    road_lines = []
    for _, road in roads.iterrows():
        if road.geometry.type == 'LineString':
            road_type = road['highway'] if 'highway' in road else 'unclassified'

            road_width = None   # Initialize road_width
            if str(road['width']) != 'nan':
                road_width = road['width']
                update_default_widths(road_type, road_width)
            else:
                road_width = default_widths.get(road_type, 2.0)

            # print(f"road_type: {road_type}, road_width: {road_width}")

            coords = np.array(road.geometry.xy).T
            road_center = [np.mean(np.array(coords)[:, 0]), np.mean(np.array(coords)[:, 1])]
            if road_near_pose(road_center, np.asarray(pos_lat_lon), threshold_dist):
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

# def get_osm_building_entrance_data_pcd(osm_file_path):
#     # Filter features for building entrances tagged as service
#     entrances = ox.geometries_from_xml(osm_file_path, tags={'entrance': 'service'})

#     # Process Building Entrances
#     entrance_points = []
#     for _, entrance in entrances.iterrows():
#         if entrance.geometry.type == 'Point':
#             x, y = entrance.geometry.xy
#             entrance_points.append([x[0], y[0], 0])  # Assuming entrances are at ground level (z=0)

#     # Create Open3D point cloud for building entrances
#     entrance_point_cloud = o3d.geometry.PointCloud()
#     entrance_point_cloud.points = o3d.utility.Vector3dVector(np.array(entrance_points))
#     entrance_point_cloud.paint_uniform_color([1, 0, 0])  # Red color for building entrances
#     return entrance_point_cloud


# def get_osm_tree_data_pcd(osm_file_path):
#     # Define tags for querying trees
#     tags = {'natural': 'tree'}

#     # Fetch tree points using defined tags
#     trees = ox.geometries_from_xml(osm_file_path, tags=tags)

#     # Process Trees
#     tree_points = []
#     for _, tree in trees.iterrows():
#         if tree.geometry.type == 'Point':
#             x, y = tree.geometry.xy
#             tree_points.append([x[0], y[0], 0])  # Assuming trees are at ground level (z=0)

#     # Create Open3D point cloud for trees
#     tree_point_cloud = o3d.geometry.PointCloud()
#     tree_point_cloud.points = o3d.utility.Vector3dVector(np.array(tree_points))
#     tree_point_cloud.paint_uniform_color([0, 1, 0])  # Green color for trees
#     return tree_point_cloud


# def get_osm_grass_data_pcd(osm_file_path):
#     # Filter features for grassland areas
#     grasslands = ox.geometries_from_xml(osm_file_path, tags={'landuse': 'grass'})

#     # Process Grassland as LineSets
#     grassland_lines = []
#     for _, grassland in grasslands.iterrows():
#         if grassland.geometry.type == 'Polygon':
#             exterior_coords = grassland.geometry.exterior.coords[:-1]  # Skip the last point because it's a repeat of the first
#             for i in range(len(exterior_coords)):
#                 start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
#                 end_point = [exterior_coords[(i + 1) % len(exterior_coords)][0], exterior_coords[(i + 1) % len(exterior_coords)][1], 0]
#                 grassland_lines.append([start_point, end_point])

#     # Create an Open3D line set from the grassland lines
#     grassland_line_set = o3d.geometry.LineSet()
#     grassland_points = [point for line in grassland_lines for point in line]  # Flatten list of points
#     grassland_lines_idx = [[i, i + 1] for i in range(0, len(grassland_points), 2)]
#     grassland_line_set.points = o3d.utility.Vector3dVector(grassland_points)
#     grassland_line_set.lines = o3d.utility.Vector2iVector(grassland_lines_idx)
#     grassland_line_set.paint_uniform_color([0.5, 1, 0.5])  # Light green color for grassland
#     return grassland_line_set


# def get_osm_road_data_pcd(osm_file_path):
#     # Define tags for querying sidewalks. Combine tags using OR logic within tuples for the same key.
#     tags = {
#         'footway': 'sidewalk',
#         'highway': 'footway',
#         'sidewalk': ('both', 'left', 'right', 'separate'),
#         'surface': 'concrete'  # Specifically retrieving asphalt surfaces
#     }

#     # Fetch sidewalks using defined tags
#     sidewalks = ox.geometries_from_xml(osm_file_path, tags=tags)

#     # Process Sidewalks
#     sidewalk_lines = []
#     for _, sidewalk in sidewalks.iterrows():
#         if sidewalk.geometry.type == 'LineString':
#             x, y = sidewalk.geometry.xy
#             for i in range(len(x) - 1):
#                 sidewalk_lines.append([[x[i], y[i], 0], [x[i + 1], y[i + 1], 0]])

#     # Create Open3D line set for sidewalks
#     sidewalk_line_set = o3d.geometry.LineSet()
#     sidewalk_points = [point for line in sidewalk_lines for point in line]
#     sidewalk_lines_idx = [[i, i + 1] for i in range(0, len(sidewalk_points), 2)]
#     sidewalk_line_set.points = o3d.utility.Vector3dVector(sidewalk_points)
#     sidewalk_line_set.lines = o3d.utility.Vector2iVector(sidewalk_lines_idx)
#     sidewalk_line_set.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for sidewalks
#     return sidewalk_line_set


if __name__ == "__main__":
    seq_extract = ExtractBuildingData(seq=0, frame_inc=100)
    seq_extract.get_osm_data_for_pointcloud(9)

    # OSM data
    # osm_buildings = get_osm_building_data_pcd(osm_file_path)
    # osm_sidewalks = get_osm_road_data_pcd(osm_file_path)
    # osm_roads = get_osm_road_data_lineset(osm_file_path)
    # osm_trees = get_osm_tree_data_pcd(osm_file_path)
    # osm_grassland = get_osm_grass_data_pcd(osm_file_path)
    # osm_building_enterances = get_osm_building_entrance_data_pcd(osm_file_path)

