import os
import numpy as np

er = 6378137. # average earth radius at the equator

'''
utils.py
'''
def latlonToMercator(lat,lon,scale):
  ''' converts lat/lon coordinates to mercator coordinates using mercator scale '''
  mx = scale * lon * np.pi * er / 180
  my = scale * er * np.log( np.tan((90+lat) * np.pi / 360) )
  return mx,my

def mercatorToLatlon(mx,my,scale):
  ''' converts mercator coordinates using mercator scale to lat/lon coordinates '''
  lon = mx * 180. / (scale * np.pi * er) 
  lat = 360. / np.pi * np.arctan(np.exp(my / (scale * er))) - 90.
  return lat, lon

def latToScale(lat):
  ''' compute mercator scale from latitude '''
  scale = np.cos(lat * np.pi / 180.0)
  return scale

def postprocessPoses (poses_in):
  
  R = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
  
  poses  = []
  for i in range(len(poses_in)):
    # if there is no data => no pose
    if not len(poses_in[i]):
      poses.append([])
      continue
    P = poses_in[i]
    poses.append( np.matmul(R, P.T).T )
    
  return poses

'''
data.py
'''
def loadOxtsData(oxts_dir, frames=None):
    ''' reads GPS/IMU data from files to memory. requires base directory
    (=sequence directory as parameter). if frames is not specified, loads all frames. '''

    ts = []
    
    if frames==None:
        ts = loadTimestamps(oxts_dir)
        oxts  = []
        for i in range(len(ts)):
            if len(ts[i]):
                try:
                    oxts.append(np.loadtxt(os.path.join(oxts_dir, 'data', '%010d.txt'%i)))
                except:
                    oxts.append([])
            else:
                oxts.append([])
    else:
        if len(frames)>1:
            k = 1
            oxts = []
            for i in range(len(frames)):
                try:
                    oxts.append(np.loadtxt(os.path.join(oxts_dir, 'data', '%010d.txt'%k)))
                except:
                    oxts.append([])
                    k=k+1
        # no list for single value
        else:
            file_name = os.path.join(oxts_dir, 'data', '%010d.txt'%k)
            try:
                oxts = np.loadtxt(file_name)
            except:
                oxts = []

    return oxts,ts

def loadTimestamps(ts_dir):
    ''' load timestamps '''
    with open(os.path.join(ts_dir, 'timestamps.txt')) as f:
        data=f.read().splitlines()
    ts = [l.split(' ')[0] for l in data] 
    return ts

def loadPoses (pos_file):
    ''' load system poses '''
    data = np.loadtxt(pos_file)
    ts = data[:, 0].astype(np.int)
    poses = np.reshape(data[:, 1:], (-1, 3, 4))
    poses = np.concatenate((poses, np.tile(np.array([0, 0, 0, 1]).reshape(1,1,4),(poses.shape[0],1,1))), 1)
    return ts, poses

'''
convertPoseToOxts.py
'''

def convertPoseToOxts(pose):
  '''converts a list of metric poses into oxts measurements,
  starting at (0,0,0) meters, OXTS coordinates are defined as
  x = forward, y = right, z = down (see OXTS RT3000 user manual)
  afterwards, pose{i} contains the transformation which takes a
  3D point in the i'th frame and projects it into the oxts
  coordinates with the origin at a lake in Karlsruhe. '''

  single_value = not isinstance(pose, list)
  if single_value:
    pose = [pose]
  
  # origin in OXTS coordinate
  origin_oxts = [48.9843445, 8.4295857] # lake in Karlsruhe
  
  # compute scale from lat value of the origin
  scale = latToScale(origin_oxts[0])
  
  # origin in Mercator coordinate
  ox, oy = latlonToMercator(origin_oxts[0],origin_oxts[1],scale)
  origin = np.array([ox,oy,0])
  
  oxts = []
  
  # for all oxts packets do
  for i in range(len(pose)):
    
    # if there is no data => no pose
    if not len(pose[i]):
      oxts.append([])
      continue
  
    # rotation and translation
    R = pose[i][0:3, 0:3]
    t = pose[i][0:3, 3]
  
    # unnormalize translation
    t = t+origin
  
    # translation vector
    lat, lon = mercatorToLatlon(t[0], t[1], scale)
    alt = t[2]
  
    # rotation matrix (OXTS RT3000 user manual, page 71/92)
    yaw = np.arctan2(R[1,0] , R[0,0])
    pitch = np.arctan2( - R[2,0] , np.sqrt(R[2,1]**2 + R[2,2]**2))
    roll = np.arctan2(R[2,1] , R[2,2])
  
    # rx = oxts{i}(4) # roll
    # ry = oxts{i}(5) # pitch
    # rz = oxts{i}(6) # heading 
    # Rx = [1 0 0 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)]; # base => nav  (level oxts => rotated oxts)
    # Ry = [cos(ry) 0 sin(ry) 0 1 0; -sin(ry) 0 cos(ry)]; # base => nav  (level oxts => rotated oxts)
    # Rz = [cos(rz) -sin(rz) 0 sin(rz) cos(rz) 0; 0 0 1]; # base => nav  (level oxts => rotated oxts)
    # R  = Rz*Ry*Rx
        
    # add oxts 
    oxts.append([lat, lon, alt, roll, pitch, yaw])
  
  if single_value:
    oxts = oxts[0]
  
  return oxts


'''
convertOxtsToPose.py
'''

def convertOxtsToPose(oxts):
  ''' converts a list of oxts measurements into metric poses,
  starting at (0,0,0) meters, OXTS coordinates are defined as
  x = forward, y = right, z = down (see OXTS RT3000 user manual)
  afterwards, pose{i} contains the transformation which takes a
  3D point in the i'th frame and projects it into the oxts
  coordinates with the origin at a lake in Karlsruhe. '''

  single_value = not isinstance(oxts, list)
  if single_value:
    oxts = [oxts]
  
  # origin in OXTS coordinate
  origin_oxts = [48.9843445, 8.4295857] # lake in Karlsruhe                                             <------------------------------------ Mercator Origin
  
  # compute scale from lat value of the origin
  scale = latToScale(origin_oxts[0])
  
  # origin in Mercator coordinate
  ox,oy = latlonToMercator(origin_oxts[0],origin_oxts[1],scale)
  origin = np.array([ox, oy, 0])
  
  pose     = []
  
  # for all oxts packets do
  for i in range(len(oxts)):
    
    # if there is no data => no pose
    if not len(oxts[i]):
      pose.append([])
      continue
  
    # translation vector
    tx, ty = latlonToMercator(oxts[i][0],oxts[i][1],scale)
    t = np.array([tx, ty, oxts[i][2]])
  
    # rotation matrix (OXTS RT3000 user manual, page 71/92)
    rx = oxts[i][3] # roll
    ry = oxts[i][4] # pitch
    rz = oxts[i][5] # heading 
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]]) # base => nav  (level oxts => rotated oxts)
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]]) # base => nav  (level oxts => rotated oxts)
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]]) # base => nav  (level oxts => rotated oxts)
    R  = np.matmul(np.matmul(Rz, Ry), Rx)

    # normalize translation
    t = t-origin
        
    # add pose
    pose.append(np.vstack((np.hstack((R,t.reshape(3,1))),np.array([0,0,0,1]))))
  
  if single_value:
    pose = pose[0]
  
  return pose



'''
testing
'''

if __name__=="__main__":
    kitti360_dir = "/Users/donceykong/Desktop/kitti360Scripts/data/KITTI360"

    # load Oxts data
    seq_id = 5
    oxts_dir = os.path.join(kitti360_dir, 'data_poses', '2013_05_28_drive_%04d_sync'%seq_id, 'oxts')
    if not os.path.isdir(oxts_dir):
        raise ValueError('%s does not exist! \nPlease specify KITTI360_DATASET in your system path.\nPlease check if you have downloaded OXTS poses (data_poses_oxts.zip) and unzipped them under KITTI360_DATASET' % oxts_dir)
    oxts,ts = loadOxtsData(oxts_dir)
    print('Loaded oxts data from %s' % oxts_dir)
    
    # convert to Mercator coordinate
    poses = convertOxtsToPose(oxts)
    
    # convert coordinate system from
    #   x=forward, y=right, z=down 
    # to
    #   x=forward, y=left, z=up
    poses = postprocessPoses(poses)

    # write to file
    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = '/Users/donceykong/Desktop/kitti360Scripts/2013_05_28_drive_%04d_sync_oxts2pose.txt'% (seq_id)
    
    with open(output_file, 'w') as f:
        for pose_ in poses:
            pose_ = ' '.join(['%.6f'%x for x in pose_.flatten()])
            f.write('%s\n'%pose_)
    print('Output written to %s' % output_file)