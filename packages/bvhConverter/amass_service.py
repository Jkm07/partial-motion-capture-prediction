import numpy as np
from .motion_array import MotionArray
from .bvh import Bvh
from .bvh_converter import save_bvh_to_file


def save_output(positions, rotations, hierarchy, file_name):
    perm = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 33, 13, 15, 34, 16, 35, 17, 36] + list(range(18, 33)) + list(range(37, 52))
    rotations_perm = rotations[:, perm].reshape((rotations.shape[0], -1))
    rotations_perm = np.rad2deg(rotations_perm)

    motion_array =  MotionArray(rotations_perm.shape[0], 0.033333, np.concatenate((positions, rotations_perm), axis=1))

    bvh = Bvh(hierarchy, motion_array)
    save_bvh_to_file(bvh, file_name)