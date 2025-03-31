import os
import torch
from torch.utils.data import Dataset
import numpy as np
from packages.math import math_utils

class AmassDataloader(Dataset):
    def __init__(self, dataset_directory, window_length = 60, offset = 10, skip_frame_ratio = 4):
        super(AmassDataloader, self).__init__()
        self.dataset_root_directory = dataset_directory
        self.window_length = window_length
        self.offset = offset
        self.skip_frame_ratio = skip_frame_ratio
        self.dataset_directories = os.listdir(self.dataset_root_directory)
        self.dataset_subdirectories = self.get_subdirectories()
        self.filies_paths = self.get_filies_paths()
        self.sample_idx = self.get_sample_indicies()

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
                self.get_prepared_rotation_matrix(rotations), 
                self.get_prepared_position_matrix(positions)), axis=-2)
        
    def get_prepared_rotation_matrix(self, rotation):
        rotation = rotation.reshape((-1, 52, 3))
        rotation_matrix = math_utils.to_rotation_matrix(rotation)
        rotation_matrix =  math_utils.differ_rotation_matrix_series(rotation_matrix)
        rotation_matrix = math_utils.matrix9D_to_6D(rotation_matrix)
        return rotation_matrix
    
    def get_prepared_position_matrix(self, position):
        out = np.concatenate([position[0][None], np.diff(position, axis=0)])
        out = np.tile(out, 2)
        out = np.expand_dims(out, axis=-2)
        return out
    