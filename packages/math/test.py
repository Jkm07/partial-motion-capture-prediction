import numpy as np
import math_utils

ss = np.load("datasets/amass/ACCAD/Female1General_c3d/A1 - Stand_poses.npz")['poses'].reshape((-1, 52, 3))

kk = np.load("datasets/amass/ACCAD/Female1General_c3d/A1 - Stand_poses.npz")['trans']

print(kk[0])

print(kk[1])

print(np.concatenate([kk[0][None], np.diff(kk, axis=0)])[0])

print(kk[0])

# pp = math_utils.to_rotation_matrix(ss)

# aa = math_utils.differ_rotation_matrix_series(pp)

# prev = np.tile(np.eye(3), (52, 1, 1))
# for org, diff in zip(pp, aa):
#     nw_value = prev @ diff 
#     prev = nw_value
#     print(np.max(np.abs(org - nw_value)))