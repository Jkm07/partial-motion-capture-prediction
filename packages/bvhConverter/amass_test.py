import numpy as np
from bvh_converter import get_hierarchy
from motion_array import MotionArray
from bvh import Bvh
from bvh_converter import save_bvh_to_file

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

def append_node(node, name: str):
    node[1].append((name, []))

smplh_hierarchy = ('pelvis', [])

append_node(smplh_hierarchy, 'left_hip')
append_node(smplh_hierarchy, 'right_hip')
append_node(smplh_hierarchy, 'spine1')



print(len(SMPLH_JOINT_NAMES))

x = np.load("filename.npz")

print(x.files)

print(x['poses'].shape)
print(x['trans'].shape)
print(x['mocap_framerate'])
print(x['betas'])

perm = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 33, 13, 15, 34, 16, 35, 17, 36] + list(range(18, 33)) + list(range(37, 52))

perm_ext = []

for p in perm:
    idx = p * 3
    perm_ext += [idx, idx + 1, idx + 2]

poses = x['poses'][:, perm_ext]

poses = np.rad2deg(poses)

motion_array= np.concatenate((x['trans'], poses), axis=1)



print(motion_array.shape)

with open('amass_hierarchy') as f:
    joints = get_hierarchy(f)
    motion_array =  MotionArray(x['poses'].shape[0], x['mocap_framerate'], motion_array)
    bvh = Bvh(joints, motion_array)
    save_bvh_to_file(bvh, "amass_zapis.bvh")