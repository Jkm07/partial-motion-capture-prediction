import os
from torch.utils.data import Dataset
import numpy as np
from packages.math import math_utils
from packages.model.ModelConfig import QueternionConfig, RotationMatrixConfig

HIPS_SLICE = (..., 0, slice(None), slice(None))
FIRST_HIPS_ROTATION = (..., 0, 0, slice(None), slice(None))

HIPS_SLICE_QUATERNION = (..., 0, slice(None))
FIRST_HIPS_QUETERNION = (..., 0, 0, slice(None))

SMPLH_PERMUTATION_TO_BVH = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 14, 17, 19, 21] + list(range(37, 52))

class AmassDataloader(Dataset):
    def __init__(self, dataset_directory, config: QueternionConfig | RotationMatrixConfig, window_length = 64, offset = 16, skip_frame_ratio = 4):
        super(AmassDataloader, self).__init__()
        self.dataset_root_directory = dataset_directory
        self.window_length = window_length
        self.offset = offset
        self.skip_frame_ratio = skip_frame_ratio
        self.dataset_directories = os.listdir(self.dataset_root_directory)
        self.dataset_subdirectories = self.get_subdirectories()
        self.filies_paths = self.get_filies_paths()
        self.sample_idx = self.get_sample_indicies()
        self.config = config

    def get_sample_indicies(self):
        print("Indexing filies")
        sample_indicies = []

        for dataset_name in self.dataset_directories:
            print(f'Load data from {dataset_name}')
            for subdataset_name in self.dataset_subdirectories[dataset_name]:
                dataset_key = (dataset_name, subdataset_name)
                for file_idx, file_path in enumerate(self.filies_paths[dataset_key]):
                    with np.load(file_path) as file:
                        count_frames = file['trans'].shape[0] // self.skip_frame_ratio
                        usable_frames = count_frames - self.window_length
                        usable_frames -= usable_frames % self.offset
                        sample_indicies += [(dataset_key, file_idx, start_frame) for start_frame in range(0, usable_frames + 1, self.offset)]
        return sample_indicies

    def get_filies_paths(self):
        result = {}
        for dataset_name in self.dataset_directories:
            for subdataset_name in self.dataset_subdirectories[dataset_name]:
                dataset_path = os.path.join(self.dataset_root_directory, dataset_name, subdataset_name)
                filies_paths_subdirectory = []
                for file_name in os.listdir(dataset_path):
                    if file_name == "shape.npz" or not file_name.endswith(".npz"):
                        print(f"Skip file {file_name}")
                        continue
                    filies_paths_subdirectory.append(os.path.join(dataset_path, file_name))
                result[(dataset_name, subdataset_name)] = filies_paths_subdirectory
        return result

    def get_subdirectories(self):
        out = {}
        for dataset_name in self.dataset_directories:
            subdirectory_path = os.path.join(self.dataset_root_directory, dataset_name)
            subdirectories = os.listdir(subdirectory_path)
            out[dataset_name] = [sub for sub in subdirectories if os.path.isdir(os.path.join(subdirectory_path, sub))]
        return out

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, idx):
        dataset_key, file_idx, start_frame = self.sample_idx[idx]
        windows_length = self.window_length * self.skip_frame_ratio
        start_frame *= self.skip_frame_ratio
        slice_idx = slice(start_frame, start_frame + windows_length, self.skip_frame_ratio)
        with np.load(self.filies_paths[dataset_key][file_idx]) as file:
            rotations = file['poses'][slice_idx, :]
            positions = file['trans'][slice_idx, :]
            return np.concatenate((
                self.get_prepared_angle(rotations), 
                self.get_prepared_position(positions)), axis=-2)
        
    def get_prepared_angle(self, rotations):
        if isinstance(self.config, RotationMatrixConfig):
            return self.get_prepared_rotation_matrix(rotations)
        elif isinstance(self.config, QueternionConfig):
            return self.get_prepared_quternion_matrix(rotations)
        else:
            raise ValueError(f"Unknown config type: {type(self.config)}")
        
    def get_prepared_position(self, positions):
        if isinstance(self.config, RotationMatrixConfig):
            return self.get_prepared_position_matrix_for_rotation_matrix(positions)
        elif isinstance(self.config, QueternionConfig):
            return self.get_prepared_position_matrix_for_quaternion(positions)
        else:
            raise ValueError(f"Unknown config type: {type(self.config)}")
        
    def get_prepared_rotation_matrix(self, rotation):
        rotation = rotation.reshape((-1, 52, 3))
        rotation = self.permute_to_bvh_format(rotation)
        rotation_matrix = math_utils.to_rotation_matrix(rotation)
        rotation_matrix = self.normalize_hips_rotation(rotation_matrix) 
        rotation_matrix = math_utils.matrix9D_to_6D(rotation_matrix)
        return rotation_matrix
    
    def get_prepared_quternion_matrix(self, rotation):
        rotation = rotation.reshape((-1, 52, 3))
        rotation = self.permute_to_bvh_format(rotation)
        quternion_matrix = math_utils.to_quternions(rotation)
        quternion_matrix = self.normalize_hips_quternion(quternion_matrix)
        return math_utils.to_decompose_quternion(quternion_matrix)
    
    def get_prepared_position_matrix_for_rotation_matrix(self, position):
        out = self.normalize_position(position)
        out = np.tile(position, 2)
        out = np.expand_dims(out, axis=-2)
        return out
    
    def get_prepared_position_matrix_for_quaternion(self, position):
        out = self.normalize_position(position)
        zero_mock = np.zeros_like(out[..., :-1])
        out = np.concatenate([position, zero_mock], axis=-1)
        out = np.expand_dims(out, axis=-2)
        return out
    
    def permute_to_bvh_format(self, rotation):
        return rotation[..., SMPLH_PERMUTATION_TO_BVH, :]
    
    def normalize_hips_rotation(self, rotation_matrix: np.array):
        rotation_matrix[HIPS_SLICE] = rotation_matrix[HIPS_SLICE] @ rotation_matrix[FIRST_HIPS_ROTATION].T
        return rotation_matrix
    
    def normalize_hips_quternion(self, rotation_matrix: np.array):
        inverse_start_hips = math_utils.quaternion_inverse(rotation_matrix[FIRST_HIPS_QUETERNION])
        rotation_matrix[HIPS_SLICE_QUATERNION] = math_utils.quaternion_multiply(rotation_matrix[HIPS_SLICE_QUATERNION], inverse_start_hips)
        return rotation_matrix
    
    def normalize_position(self, position):
        return position - position[0]
    