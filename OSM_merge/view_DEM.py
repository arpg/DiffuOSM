import rasterio
import numpy as np
import open3d as o3d

# Read DEM data from a TIFF file
with rasterio.open('/home/donceykong/Desktop/OSM_KITTI360/kitti360Scripts/data/DEM_30m_0005.tif') as dem:
    elevation_data = dem.read(1)  # read the first band

# Convert elevation data to a point cloud
points = []
for i in range(elevation_data.shape[0]):
    for j in range(elevation_data.shape[1]):
        # Assuming i, j are the coordinates, and elevation_data[i, j] is the elevation
        points.append([i, j, elevation_data[i, j]])

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.array(points))

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
