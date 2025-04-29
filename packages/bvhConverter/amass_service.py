import numpy as np
from .motion_array import MotionArray
from .bvh import Bvh
from .bvh_converter import save_bvh_to_file


def save_output(positions, rotations, hierarchy, file_name):
    rotations_perm = rotations.reshape((rotations.shape[0], -1))
    rotations_perm = np.rad2deg(rotations_perm)

    motion_array =  MotionArray(rotations_perm.shape[0], 0.033333, np.concatenate((positions * 100, rotations_perm), axis=1))

    bvh = Bvh(hierarchy, motion_array)
    save_bvh_to_file(bvh, file_name)