'''
Doncey Albin

Example script to read extracted building data from OSM-KITTI360 mapping.

The only buildings here are building #95 and building #105 from sequence 5.

- To go to building 105, press 'o'. To go back to building 95, press 'b'.

- To cycle up through scans, press 'n,' and to go back to previous scans, press 'p.'
'''

import os
import numpy as np
import open3d as o3d

try:
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        raise KeyError("KITTI360_DATASET environment variable not found.")
except KeyError as e:
    print("Error:", e)
    print("Set absolute path to where the directory for KITTI-360 dataset is. \nSet using \"export KITTI360_DATASET = <path/to/dataset>\"")


def read_building_pc_file(file_path):
    point_cloud = np.fromfile(file_path)
    return point_cloud.reshape(-1, 3)

def read_building_edges_file(building_edges_file):
    with open(building_edges_file, 'rb') as bin_file:
        edges_array = np.fromfile(bin_file, dtype=float).reshape(-1, 2, 3)  # Reshape to 3D array
    return edges_array

def get_building_scan_pcd(seq, building_index):
    per_building_dir = os.path.join(kitti360Path, 'data_3d_extracted', '2013_05_28_drive_%04d_sync' % seq, 'buildings/per_building')
    per_build_edges_file = os.path.join(per_building_dir, f'build_{building_index}_edges.bin', )
    per_build_file = os.path.join(per_building_dir, f'build_{building_index}_scan_{scan_num}.bin', )
    per_build_accum_file = os.path.join(per_building_dir, f'build_{building_index}_accum.bin' )
    per_build_diff_file = os.path.join(per_building_dir, f'build_{building_index}_diffscan_{scan_num}.bin', )

    build_edges = read_building_edges_file(per_build_edges_file).reshape(-1, 3)
    build_edges_span = np.linalg.norm(np.ptp(build_edges, axis=0))
    build_edges_lines_idx = [[i, i + 1] for i in range(0, len(build_edges) - 1, 2)]
    build_points = read_building_pc_file(per_build_file)
    build_points_accum = read_building_pc_file(per_build_accum_file)
    build_points_diff = read_building_pc_file(per_build_diff_file)

    build_points = np.array(build_points)
    build_points[:, 2] = build_points[:, 2] - np.min(build_points[:, 2])

    build_points_diff = np.array(build_points_diff)
    build_points_diff[:, 2] = build_points_diff[:, 2] - np.min(build_points_diff[:, 2])
    
    build_points_accum = np.array(build_points_accum)
    # build_points_accum[:, 2] = 0

    print(build_points_diff)

    build_points_pcd = o3d.geometry.PointCloud()
    build_edges_pcd = o3d.geometry.LineSet()
    build_points_accum_pcd = o3d.geometry.PointCloud()
    build_points_diff_pcd = o3d.geometry.PointCloud()

    build_points_pcd.points = o3d.utility.Vector3dVector(build_points)
    build_edges_pcd.points = o3d.utility.Vector3dVector(build_edges)
    build_edges_pcd.lines = o3d.utility.Vector2iVector(build_edges_lines_idx)
    build_points_accum_pcd.points = o3d.utility.Vector3dVector(build_points_accum)
    build_points_diff_pcd.points = o3d.utility.Vector3dVector(build_points_diff)

    build_points_pcd.paint_uniform_color([1, 0, 0])         # Red color for per frame, per building points
    build_edges_pcd.paint_uniform_color([0, 1, 0])          # Green color for per building OSM polygon edges
    build_points_accum_pcd.paint_uniform_color([0, 0, 0])   # Black color for accum per building points
    build_points_diff_pcd.paint_uniform_color([0, 0, 1])    # Blue color for diff per frame, per building points

    return build_points_pcd, build_edges_pcd, build_points_accum_pcd, build_points_diff_pcd, build_edges_span

def get_max_scan_for_build(seq, building_index):
    per_building_dir = os.path.join(kitti360Path, 'data_3d_extracted', '2013_05_28_drive_%04d_sync' % seq, 'buildings/per_building')
    max_scan = 1
    while True:
        build_pc_file = os.path.join(per_building_dir, f'build_{building_index}_scan_{max_scan}.bin')
        if os.path.exists(build_pc_file):
            max_scan += 1
        else:
            break

    return max_scan

def change_frame(vis, key_code):
    global scan_num
    global building_index
    global seq
    global initial_span

    if key_code == ord('B')  and building_index < 105:
        building_index += 1
        scan_num = 1
    elif key_code == ord('O') and building_index > 95:
        building_index -= 1
        scan_num = 1

    max_scan_num = get_max_scan_for_build(seq, building_index)

    if key_code == ord('N') and scan_num < (max_scan_num - 1):
        scan_num += 1
    elif key_code == ord('P') and scan_num > 1:
        scan_num -= 1
    
    build_points_pcd, build_edges_pcd, build_points_accum_pcd, build_points_diff_pcd, build_edges_span = get_building_scan_pcd(seq, building_index)
    
    # Get the center of the bounding box of the build_edges_pcd
    center = build_edges_pcd.get_center()
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.eye(3)
    extrinsic[:3, 3] = center

    vis.clear_geometries()
    vis.add_geometry(build_points_pcd)
    vis.add_geometry(build_edges_pcd)
    # vis.add_geometry(build_points_accum_pcd)
    vis.add_geometry(build_points_diff_pcd)
    vis.add_geometry(build_points_accum_pcd)

    # Control where the visualizer looks at
    vis.get_view_control().set_lookat(center)
    vis.get_view_control().set_front([-0.5, -0.3, 1])
    zoom = 0.5 * build_edges_span/initial_span
    print(f"new_zoom: {zoom}")
    vis.get_view_control().set_zoom(zoom)  

    return True

initial_span = None
scan_num = 1            # starting scan number
building_index = 6     # Could cycle through and collect all edges, if desired. Buidling indeces begin at 1.
seq = 0                 # starting sequence number
def main(): 
    global initial_span
    build_points_pcd, build_edges_pcd, build_points_accum_pcd, build_points_diff_pcd, build_edges_span = get_building_scan_pcd(seq, building_index)
    initial_span = build_edges_span

    key_to_callback = {
        ord('N'): lambda vis: change_frame(vis, ord('N')),
        ord('P'): lambda vis: change_frame(vis, ord('P')),
        ord('B'): lambda vis: change_frame(vis, ord('B')),
        ord('O'): lambda vis: change_frame(vis, ord('O'))
    }
    o3d.visualization.draw_geometries_with_key_callbacks([build_points_pcd, build_edges_pcd, build_points_accum_pcd, build_points_diff_pcd], key_to_callback)

if __name__=="__main__": 
    main() 