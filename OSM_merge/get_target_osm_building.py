import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

# Path to your OSM file
osm_file_path = '/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/map_0005.osm'

# Load the .osm file
G = ox.graph_from_xml(osm_file_path)

crs_utm = 'EPSG:32632'

# Filter features for buildings and sidewalks
buildings = ox.features_from_xml(osm_file_path, tags={'building': True})
sidewalks = ox.features_from_xml(osm_file_path, tags={'highway': 'footway', 'footway': 'sidewalk'})

# Filter target building
target_building_names = ["Evangelischer Kindergarten"]
target_building_name = target_building_names[0]
target_building = ox.features_from_xml(osm_file_path, tags={'name': f'{target_building_name}'})

# Reproject the data
buildings = buildings.to_crs(crs_utm)
sidewalks = sidewalks.to_crs(crs_utm)
target_building = target_building.to_crs(crs_utm)

# Plot the streets
fig, ax = ox.plot_graph(G, show=False, close=False)

# Plot the buildings and sidewalks
buildings.plot(ax=ax, facecolor='blue', alpha=0.7)  # Blue colored buildings
sidewalks.plot(ax=ax, edgecolor='red', alpha=1.0)   # Red colored sidewalks

# Assuming 'target_building' is not empty and contains a single building
if not target_building.empty:
    # Get the geometry of the target building
    target_geom = target_building.geometry.iloc[0]

    # In case the target is a point (like a node), convert it to a shapely Point
    if isinstance(target_geom, Point):
        target_point = target_geom
    else:
        # If it's a polygon, use its centroid
        target_point = target_geom.centroid

    # Find the nearest building in the 'buildings' GeoDataFrame
    # This creates a temporary column 'distance' with the distance to the target
    buildings['distance'] = buildings.distance(target_point)

    # Find the index of the building with the smallest distance
    nearest_building_idx = buildings['distance'].idxmin()

    # Extract the nearest building
    nearest_building = buildings.loc[[nearest_building_idx]]
    
    # Initialize a set with the nearest building index
    processed_buildings_idx = {nearest_building_idx}
    
    # Initialize a set for buildings to be processed
    buildings_to_process = {nearest_building_idx}

    max_iterations = 100  # Set a reasonable limit to prevent infinite loops
    current_iteration = 0

    while buildings_to_process and current_iteration < max_iterations:
        current_iteration += 1
        buildings_to_process_list = list(buildings_to_process)

        # Find buildings that touch any building in buildings_to_process
        new_touching_buildings = buildings[
            buildings.geometry.touches(buildings.loc[buildings_to_process_list].unary_union)
        ]

        # Exclude already processed buildings
        new_touching_buildings = new_touching_buildings[~new_touching_buildings.index.isin(processed_buildings_idx)]

        # Update buildings_to_process for the next iteration
        new_indices = set(new_touching_buildings.index) - processed_buildings_idx
        if not new_indices:  # Break the loop if no new buildings are found
            break
        buildings_to_process = new_indices

        # Add new touching buildings to processed_buildings_idx
        processed_buildings_idx.update(new_indices)

    # Convert set to list for DataFrame indexing after all processing is done
    processed_buildings_idx_list = list(processed_buildings_idx)

    # Extract all processed buildings
    touching_buildings = buildings.loc[processed_buildings_idx_list]

    # Plot the touching buildings in yellow
    touching_buildings.plot(ax=ax, facecolor='yellow', alpha=0.7)

    # Plot the nearest building in a different color, to differentiate
    nearest_building.plot(ax=ax, facecolor='orange', alpha=0.7)

    # Plot the target building (or building node) in green
    target_building.plot(ax=ax, facecolor='green', alpha=1.0)
else:
    print(f"Building named {target_building_name} not found in the specified area.")