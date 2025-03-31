import numpy as np

def to_angle(rotation_matrix):
    x = rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2]
    y = rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0]
    z = rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]

    angle = np.arccos((np.trace(rotation_matrix, axis1=-1, axis2=-2) - 1) / 2)[...,None]

    return np.concatenate([
        x[...,None], 
        y[...,None], 
        z[...,None]], axis=-1) * angle / (2 * np.sin(angle))


def to_rotation_matrix(axis_angle):
    angle = np.linalg.norm(axis_angle, axis=-1)
    divide_norm = angle.copy()
    divide_norm[divide_norm == 0] = 1
    axis = axis_angle / divide_norm[...,None]
    sin = np.sin(angle)
    cos = np.cos(angle)

    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    top_left = cos + x * x * (1 - cos)
    top_mid = x * y * (1 - cos) - z * sin
    top_right = x * z * (1- cos) + y * sin

    center_left = y * x * (1 - cos) + z * sin
    center_mid = cos + y * y * (1 - cos)
    center_right = y * z * (1 - cos) - x * sin

    bot_left = z * x * (1 - cos) - y * sin
    bot_mid = z * y * (1 - cos) + x * sin
    bot_right = cos + z * z * (1 - cos)

    return np.concatenate([
        top_left[..., None], 
        top_mid[..., None], 
        top_right[..., None], 
        center_left[..., None],
        center_mid[..., None],
        center_right[..., None],
        bot_left[..., None],
        bot_mid[..., None],
        bot_right[..., None]], axis=-1).reshape((-1, axis_angle.shape[1], 3, 3))

def differ_rotation_matrix_series(rotation_matrix):
    transpose = rotation_matrix.swapaxes(-1, -2)
    concat_shape = np.array(transpose.shape)
    concat_shape[[-4, -2, -1]] = 1
    concat = np.concatenate([np.tile(np.eye(3), concat_shape), transpose], axis=0)[..., :-1, :, :, :]
    return concat @ rotation_matrix

def matrix9D_to_6D(mat):
    return mat[..., :-1].reshape(*mat.shape[:-2], -1)