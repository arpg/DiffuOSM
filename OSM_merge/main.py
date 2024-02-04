"""
By: Doncey Albin


Refactoring of kitti360scripts and recoverKitti repositories was made in order to create this pipeline.
I couldn't have done it without them.
    - kitti360scripts:
    - recoverKitti:


Preprosses data (A) and extract training data for diffusion-based inpainting (B).

A) Preprosses data and transform it from world -> lat-long, as well as save it.
    A.1) This includes lidar frames for each seq.

B) Extract building points for each frame in each sequence, as well as save them.
    B.1) Do the same for road points later.

"""


# 1) Create vel poses txt for all sequences

# 2) Create all data_oxsx

# 3) Get labels for each frame in each seq

# 4) Get osm for each seq

# 5) Get extracted building points for each building seq

def preprocess_and_save_data(seq = '0005'):
    if seq is None:
        print("No sequence selected for preprocessing. Please add sequence number in main module.")

def extract_and_save_building_points(seq = '0005'):
    if seq is None:
        print("No sequence selected for extracting training data. Please add sequence number in main module.")

def main(): 
    extract_and_save_building_points()
  
if __name__=="__main__": 
    main() 