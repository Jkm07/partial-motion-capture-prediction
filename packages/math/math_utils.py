import numpy as np
import torch

def to_angle(rotation_matrix):
    x = rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2]
    y = rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0]
    z = rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]

    angle = np.arccos((np.trace(rotation_matrix, axis1=-1, axis2=-2) - 1) / 2)[...,None]

    return np.concatenate([
        x[...,None], 
        y[...,None], 
        z[...,None]], axis=-1) * angle / (2 * np.sin(angle))

def normalize_axis(axis_angle):
    norm = np.linalg.norm(axis_angle, axis=-1)
    divide_norm = norm.copy()
    divide_norm[divide_norm == 0] = 1
    axis = axis_angle / divide_norm[...,None]
    return norm, axis

def to_quternions(axis_angle):
    angle, axis = normalize_axis(axis_angle)
    sin = np.sin(angle / 2)[...,None]
    cos = np.cos(angle / 2)[...,None]
    return np.concatenate((cos, axis * sin), axis=-1)

def quaternion_inverse(quternion):
    quternion = quternion.copy()
    quternion[..., -3:] *= -1
    return quternion

def quaternion_multiply(q1, q2):
    """Multiply two quaternions (Hamilton product)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3], 
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3], 
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.concatenate([w[..., None], x[..., None], y[..., None], z[..., None]], axis=-1)

def quaternion_multiply_torch(q1, q2):
    # Split components
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    
    # Compute product components
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    # Stack components back together
    return torch.stack((w, x, y, z), dim=-1)

def to_decompose_quternion(quternion):
    sin, axis = normalize_axis(quternion[..., -3:])
    return np.concatenate([quternion[..., :1], sin[..., None], axis], axis=-1)

def from_decompose_quternion(dec_quternion):
    return torch.concatenate([dec_quternion[..., 0:1], dec_quternion[..., -3:] * dec_quternion[..., 1:2]], dim=-1)


def to_rotation_matrix(axis_angle):
    angle, axis = normalize_axis(axis_angle)
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

def batch_vector_dot_torch(vec1, vec2):
    dot = torch.matmul(vec1[..., None, :], vec2[..., None])
    return dot.squeeze(-1)

def normalize_torch(tensor, dim=-1, eps=1e-5):
    return torch.nn.functional.normalize(tensor, p=2, dim=dim, eps=eps)

def matrix6D_to_9D_torch(mat: torch.Tensor) -> torch.Tensor:
    if mat.shape[-1] != 6:
        raise ValueError(
            "Last two dimension should be 6, got {0}.".format(mat.shape[-1]))

    mat = mat.reshape(*mat.shape[:-1], 3, 2)

    col0 = normalize_torch(mat[..., 0], dim=-1)

    dot_prod = batch_vector_dot_torch(col0, mat[..., 1])
    col1 = normalize_torch(mat[..., 1] - dot_prod * col0, dim=-1)

    col0 = col0.unsqueeze(-1)
    col1 = col1.unsqueeze(-1)

    col2 = torch.cross(col0, col1, dim=-2)
    return torch.cat([col0, col1, col2], dim=-1)


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def matrix9D_to_quat_torch(mat: torch.Tensor) -> torch.Tensor:
    if mat.size(-1) != 3 or mat.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{mat.shape}.")
    m00 = mat[..., 0, 0]
    m11 = mat[..., 1, 1]
    m22 = mat[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, mat[..., 2, 1] - mat[..., 1, 2])
    o2 = _copysign(y, mat[..., 0, 2] - mat[..., 2, 0])
    o3 = _copysign(z, mat[..., 1, 0] - mat[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def undo_differ_rotation_matrix_torch(mat: torch.Tensor) -> torch.Tensor:
    mat = torch.clone(mat)
    for i in range(1, mat.shape[-4]):
        mat[..., i, :, :, :] = mat[..., i - 1, :, :, :] @ mat[..., i, :, :, :]

    return mat

def get_quat_from_matrix(mat: torch.Tensor) -> torch.Tensor:
    mat = matrix6D_to_9D_torch(mat)
    return matrix9D_to_quat_torch(mat)

def rotation_matrix_to_euler_torch(rot_mats: torch.Tensor) -> torch.Tensor:
    assert rot_mats.shape[-2:] == (3, 3), f"Input must be of shape (..., 3, 3), got {rot_mats.shape}"
    
    eulers = torch.zeros(rot_mats.shape[:-2] + (3,), device=rot_mats.device, dtype=rot_mats.dtype)
    
    pitch = torch.asin(-rot_mats[..., 2, 0])
    
    normal_case = torch.abs(rot_mats[..., 2, 0]) < 0.9999
    gimbal_lock_case = ~normal_case
    
    if torch.any(normal_case):
        eulers[..., 0][normal_case] = torch.atan2(
            rot_mats[..., 1, 0][normal_case], 
            rot_mats[..., 0, 0][normal_case]
        )
        
        eulers[..., 2][normal_case] = torch.atan2(
            rot_mats[..., 2, 1][normal_case], 
            rot_mats[..., 2, 2][normal_case]
        )
    
    if torch.any(gimbal_lock_case):
        eulers[..., 0][gimbal_lock_case] = torch.atan2(
            -rot_mats[..., 1, 2][gimbal_lock_case],
            rot_mats[..., 1, 1][gimbal_lock_case]
        )
        
        eulers[..., 2][gimbal_lock_case] = 0.0
    
    eulers[..., 1] = pitch
    
    return eulers

def get_euler_from_matrix(mat: torch.Tensor) -> torch.Tensor:
    mat = matrix6D_to_9D_torch(mat)
    return rotation_matrix_to_euler_torch(mat)
