'''
By: Doncey Albin


Refactoring of kitti360scripts and recoverKitti repositories was made in order to create this pipeline.
I couldn't have done it without them.

    - kitti360scripts:
    - recoverKitti:


Preprocess KITTI360 data via:

    1) Extract semantic labels for each frame in each sequence.
        -->

    2) Transorm imu world poses to be in frame of velodyne.
        --> saved in KITTI360/data_poses/2013_05_28_drive_{sequence}_sync/velodyne_poses_world.txt

    3) Transform each pointcloud frame to be at correct location from the velodyne_poses_world.txt
        --> saved in KITTI360/data_3d_transformed/2013_05_28_drive_{sequence}_sync/velodyne_points/data_world/

    4) Transform each transformed point cloud to be in lat-long.
        --> saved in KITTI360/data_3d_transformed/2013_05_28_drive_{sequence}_sync/velodyne_points/data_latlong/

    5) Transform world (improved imu) poses to lat-long.
        --> saved in KITTI360/data_poses/2013_05_28_drive_{sequence}_sync/poses_world.txt

'''

