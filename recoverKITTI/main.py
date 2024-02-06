from accumulation import *
import argparse
import os



seq = 5
if 'KITTI360_DATASET' in os.environ:
    kitti360Path = os.environ['KITTI360_DATASET']
else:
    kitti360Path = os.path.join(os.path.dirname(
                        os.path.realpath(__file__)), '..', '..')
train_test = 'train'
if (seq==8 or seq==18): train_test = 'test'
sequence = '2013_05_28_drive_%04d_sync' % seq
label_path = os.path.join(kitti360Path, 'data_3d_semantics', train_test, sequence, 'labels')

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kitti_dir", default='/home/donceykong/kitti_360/kitti360Scripts/data/KITTI-360', help="path to kitti360 dataset")
parser.add_argument("-o", "--output_dir", default='/home/donceykong/kitti_360/kitti360Scripts/data/KITTI-360/data_3d_semantics/train/2013_05_28_drive_0000_sync/labels', help="path to output_dir")
parser.add_argument("-s", "--sequence", default='2013_05_28_drive_0000_sync', help="sequence name")
parser.add_argument("-f", "--first_frame", type=int, default=0)
parser.add_argument("-l", "--last_frame", type=int, default=11517)
parser.add_argument("-d", "--data_source", help="1:velodyne scans", default=1, type=int)
args = parser.parse_args()

root_dir = args.kitti_dir
sequence = args.sequence
output_dir = args.output_dir
first_frame = args.first_frame
last_frame = args.last_frame
source = args.data_source
travel_padding = 20
min_dist_dense = 0.02

PA = PointAccumulation(root_dir, output_dir, sequence, first_frame, last_frame, travel_padding, source, min_dist_dense, True, False)

print('Initialization Done!')

if not PA.createOutputDir():
    print('Error: Unable to create the output directory!')

if not PA.loadTransformation():
    print('Error: Unable to load the calibrations!')

if not PA.getInterestedWindow():
    print('Error: Invalid window of interested!')

print("Loaded " + str(len(PA.Tr_pose_world)) + " poses")

PA.loadTimestamps()

print("Loaded " + str(len(PA.veloTimestamps)) + " velo timestamps")

PA.addVelodynePoints()
PA.getPointsInRange()
PA.writeToFiles()