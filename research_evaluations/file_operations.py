import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DUMMY_ZERO


def execute_if_save_enabled(func: callable):
    def wrapper(*args, **kwargs):
        if args[DUMMY_ZERO].enable_save:
            return func(*args, **kwargs)

    return wrapper


class AnalyseResultsSaver:
    def __init__(self, trajectory_name, filename='', folder_suffix='', enable_save=True, use_time_stamp=True):
        if not use_time_stamp and folder_suffix != '':
            self.current_result_path: Path = Path('analyse_results') / trajectory_name / folder_suffix
        else:
            self.current_result_path: Path = Path('analyse_results') / trajectory_name / (datetime.now().strftime(
                "%Y-%m-%d_%H.%M.%S") + folder_suffix)
        if enable_save and not self.current_result_path.exists():
            os.makedirs(self.current_result_path)
        self.filename = filename
        self.enable_save = enable_save

    def goal_filename(self, extension):
        return self.current_result_path / (self.filename + extension)

    @execute_if_save_enabled
    def save_to_csv(self, result, header=None):
        if header is None:
            result_df = pd.DataFrame(result)
        else:
            result_df = pd.DataFrame(result)[header]
        goal_path = self.goal_filename('.csv')
        result_df.to_csv(goal_path)
        print(f'Results successfully saved into: {goal_path}')

    @execute_if_save_enabled
    def save_to_npy(self, array: np.ndarray, new_filename=None):
        if new_filename is not None:
            self.filename = new_filename
        np.save(self.goal_filename('.npy'), array)

    @execute_if_save_enabled
    def save_to_npz(self, dictionary: dict, new_filename: str = None):
        if new_filename is not None:
            self.filename = new_filename
        np.savez(self.goal_filename('.npz'), **dictionary)

    @execute_if_save_enabled
    def merge_save_to_npz(self, dictionary: dict, new_filename: str = None):
        # TODO: Merge is not working this way. I average the average of the average
        #  the newer ndarray have a higher weight in the average
        if new_filename is not None:
            self.filename = new_filename

        file_path = self.goal_filename('.npz')

        if os.path.exists(file_path):
            existing_data: dict = np.load(file_path)
            for key, value in dictionary.items():
                dictionary[key] = (existing_data[key] + value) / 2 if key in existing_data else value

        # Save the updated dictionary to the file
        np.savez(file_path, **dictionary)


def merge_npz_files(file1, file2, output_file):
    data1 = np.load(file1)
    data2 = np.load(file2)

    merged_data = {**data1, **data2}

    np.savez(output_file, **merged_data)


class AnalyseResultLoader:
    def __init__(self, trajectory_name, sub_dir=None):
        self.current_result_path: Path = Path('analyse_results') / trajectory_name
        if sub_dir is not None:
            self.current_result_path = self.current_result_path / sub_dir

    def get_load_path(self, filename):
        return self.current_result_path / filename

    def load_npy(self, filename: str) -> np.ndarray:
        load_path = self.get_load_path(filename)
        return np.load(load_path)

    def load_npz(self, filename: str) -> dict:
        load_path = self.get_load_path(filename)
        return dict(np.load(load_path, allow_pickle=True))

    def load_npz_by_filelist(self, filename_list: list) -> dict:
        merged_dict = {}
        for filename in filename_list:
            merged_dict.update(self.load_npz(filename))
        return merged_dict

    def load_npz_files_in_directory(self, directory_name):
        directory_path = self.get_load_path(directory_name)
        filename_list = [file for file in os.listdir(directory_path) if file.endswith(".npz")]
        return self.load_npz_list(directory_name, filename_list)

    def load_npz_list(self, root_dir, filename_list):
        loaded_dict = {}
        for filename in filename_list:
            loaded_dict[filename] = self.load_npz(Path(root_dir) / filename)
        return loaded_dict

    def merge_npz_files(self, goal_dictionary):
        # Step 1: Load all files ending with .npz in the sub_directories of directory_path
        file_dicts = {}

        for root, dirs, files in os.walk(self.current_result_path):
            for file in files:
                if file.endswith(".npz"):
                    full_path = os.path.join(root, file)
                    # Extract the name of the sub-directory to be used as a key
                    sub_directory = os.path.basename(root)
                    if file in file_dicts:
                        file_dicts[file][sub_directory] = np.load(full_path)
                    else:
                        file_dicts[file] = {sub_directory: np.load(full_path)}

        # Step 2: Create merged dictionaries for each file
        save_path = Path(goal_dictionary) / (datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
        os.makedirs(save_path, exist_ok=True)
        for file, sub_directories_dict in file_dicts.items():
            merged_dict = {}
            for key in ['PCA', 'DROPP', 'TICA', 'FastICA']:
                # Calculate the average of the numpy arrays for each key
                try:
                    merged_dict[key] = np.mean([sub_dict[key] for sub_dict in sub_directories_dict.values()], axis=0)
                except ValueError:
                    merged_dict[key] = calculate_mean_along_axis0(
                        [sub_dict[key] for sub_dict in sub_directories_dict.values()])

            # Step 3: Save the merged dictionary into a new .npz file at the goal_dictionary
            np.savez(save_path / file, **merged_dict)
        return save_path


def calculate_mean_along_axis0(list_of_ndarrays):
    # Find the maximum size along axis 1 in the list of arrays
    max_size = max(arr.shape[1] for arr in list_of_ndarrays)

    # Pad the smaller arrays with NaNs along axis 1 to match the maximum size
    padded_arrays = [np.pad(arr, ((0, 0), (0, max_size - arr.shape[1])), mode='constant', constant_values=np.nan) for
                     arr in list_of_ndarrays]

    # Calculate the mean along axis 0, ignoring NaN values
    mean_along_axis0 = np.nanmean(padded_arrays, axis=0)

    return mean_along_axis0
