
import torch as nn
import numpy as np
import smplx
import pickle
from utils import quat, bvh
import os

SMPLH_JOINT_NAMES = [
    'pelvis', #0
    'left_hip', #1
    'right_hip', #5
    'spine1', #9
    'left_knee', #2
    'right_knee', #6
    'spine2', #10
    'left_ankle', #3
    'right_ankle', #7
    'spine3', #11
    'left_foot', #4
    'right_foot', #8
    'neck', #12
    'left_collar', #14
    'right_collar', #33
    'head', #13
    'left_shoulder', #15
    'right_shoulder', #34
    'left_elbow', #16
    'right_elbow', #35
    'left_wrist', #17
    'right_wrist', #36
    'left_index1', #18
    'left_index2', #19
    'left_index3', #20
    'left_middle1', #21
    'left_middle2', #22
    'left_middle3', #23
    'left_pinky1', #24
    'left_pinky2', #25
    'left_pinky3', #26
    'left_ring1', #27
    'left_ring2', #28
    'left_ring3', #29
    'left_thumb1', #30
    'left_thumb2', #31
    'left_thumb3', #32
    'right_index1', #37
    'right_index2', #38
    'right_index3', #39
    'right_middle1', #40
    'right_middle2', #41
    'right_middle3', #42
    'right_pinky1', #43
    'right_pinky2', #44
    'right_pinky3', #45
    'right_ring1', #46
    'right_ring2', #47
    'right_ring3', #48
    'right_thumb1', #49
    'right_thumb2', #50
    'right_thumb3' #51
]

def get_offset():
    model = smplx.create(model_path='./datasets/smpl', 
                        model_type="smplh",
                        gender='male', 
                        batch_size=1)

    rest = model(
        # betas = nn.from_numpy(poses['betas']) 
    )

    parents = model.parents.detach().cpu().numpy()
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:52,:]

    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100
    return offsets

def get_parrents():
    model = smplx.create(model_path='./datasets/smpl', 
                        model_type="smplh",
                        gender='male', 
                        batch_size=1)

    rest = model(
        # betas = nn.from_numpy(poses['betas']) 
    )

    return model.parents.detach().cpu().numpy()

def convert_to_bvh(model_name, offsets, parents):
    
    poses = np.load(model_name)
    rots = poses["poses"].reshape((-1, 52, 3))
    trans = poses["trans"]
    order = 'zyx'

    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:,0] += trans * 100

    rotations = np.degrees(quat.to_euler(quat.from_axis_angle(rots), order))

    return {
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": SMPLH_JOINT_NAMES,
        "order": order,
        "frametime": 1 / 120,
    }

def load_data(main_dir_path = "./datasets/amass"):
    offsets = get_offset()
    parrents = get_parrents()


    for dataset_directory in os.listdir(main_dir_path):
        if dataset_directory in ():
            print(f"Skip file {dataset_directory}")
            continue
        dataset_path = os.path.join(main_dir_path, dataset_directory)
        if not os.path.isdir(dataset_path):
            continue
        for subdataset_directory in os.listdir(dataset_path):
            subdataset_path = os.path.join(dataset_path, subdataset_directory)
            if not os.path.isdir(subdataset_path):
                continue
            bvh_file_directory = os.path.join(main_dir_path + '_bvh', dataset_directory, subdataset_directory)
            os.makedirs(bvh_file_directory, exist_ok=True)
            for file in os.listdir(subdataset_path):
                if file == "shape.npz":
                    print(f"Skip file {file}")
                    continue
                print(f"Conversion of {file}")
                bvh_data = convert_to_bvh(os.path.join(subdataset_path, file), offsets, parrents)
                
                print(bvh_file_directory)
                
                bvh.save(os.path.join(bvh_file_directory, file[:-3] + "bvh"), bvh_data)

if __name__ == '__main__':
    load_data()