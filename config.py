import json
import pandas as pd
import os
import warnings
import weather_preprocessing as wp

from trajectory import ProteinTrajectory, SubTrajectoryDecorator, WeatherTrajectory, DataTrajectory
from utils.param_keys import *
from utils.param_keys.analyses import *
from utils.param_keys.kernel_functions import *
from utils.param_keys.model import *
from utils.param_keys.run_options import *
from utils.param_keys.traj_dims import *


def get_run_params(alg_params_json: str) -> dict:
    """
    Loads the running configuration given from a json file or the default dictionary from the code.
    @param alg_params_json: str
        Path to the json data with the running configuration
    @return: dict
        Running Configuration
    """
    if alg_params_json is not None:
        return json.load(open(alg_params_json))
    else:
        warnings.warn('The default run option is used instead of a .json-file.\n'
                      '  Use a configuration from the `config_files`-folder together with the args `-o`.')
        return {
            RUN_OPTION: COMPARE,
            PLOT_TYPE: COLOR_MAP,  # 'heat_map', 'color_map', '3d_map', 'explained_var_plot'
            PLOT_TICS: True,
            CARBON_ATOMS_ONLY: True,
            INTERACTIVE: True,
            N_COMPONENTS: 2,
            BASIS_TRANSFORMATION: False,
            USE_ANGLES: False,
            TRAJECTORY_NAME: '2f4k',
            FILE_ELEMENT: 0,
        }


def get_files_and_kwargs(params: dict):
    """
    This method returns the filename list of the trajectory and generates the kwargs for the ProteinTrajectory.
    The method is individually created for the available data set.
    Add new trajectory options, if different data set are used.
    @param params: dict
        running configuration
    @return: tuple
        list of filenames of the trajectories AND
        kwargs with the important arguments for the classes
    """
    try:
        trajectory_name = params[TRAJECTORY_NAME]
        file_element = params[FILE_ELEMENT]
        data_set: str = params[DATA_SET]
    except KeyError as e:
        raise KeyError(f'Run option parameter is missing the key: `{e}`. This parameter is mandatory.')

    if trajectory_name == '2f4k':
        filename_list = [f'2F4K-0-protein-{i:03d}.dcd' for i in
                         range(0, 62)]  # + ['tr3_unfolded.xtc', 'tr8_folded.xtc']
        kwargs = {FILENAME: filename_list[file_element], TOPOLOGY_FILENAME: '2f4k.pdb', FOLDER_PATH: 'data/2f4k'}
    elif trajectory_name == 'prot2':
        filename_list = ['prod_r1_nojump_prot.xtc', 'prod_r2_nojump_prot.xtc', 'prod_r3_nojump_prot.xtc']
        kwargs = {FILENAME: filename_list[file_element], TOPOLOGY_FILENAME: 'prod_r1_pbc_fit_prot_last.pdb',
                  FOLDER_PATH: 'data/ProtNo2'}
    elif trajectory_name == 'savinase':
        filename_list = ['savinase_1.xtc', 'savinase_2.xtc']
        kwargs = {FILENAME: filename_list[file_element], TOPOLOGY_FILENAME: 'savinase.pdb',
                  FOLDER_PATH: 'data/Savinase'}
    elif trajectory_name == '2wav':
        filename_list = [f'2WAV-0-protein-{i:03d}.dcd' for i in range(0, 69)]
        kwargs = {FILENAME: filename_list[file_element], TOPOLOGY_FILENAME: '2wav.pdb',
                  FOLDER_PATH: 'data/2WAV-0-protein', ATOMS: list(range(710))}
    elif trajectory_name == '5i6x':
        filename_list = ['protein.xtc', 'system.xtc']
        kwargs = {FILENAME: filename_list[file_element], TOPOLOGY_FILENAME: 'protein.pdb',
                  FOLDER_PATH: 'data/ser-tr'}
    elif trajectory_name == 'fs-peptide':
        filename_list = [f'trajectory-{i}.xtc' for i in range(1, 28 + 1)]
        kwargs = {FILENAME: filename_list[file_element], TOPOLOGY_FILENAME: 'fs-peptide.pdb',
                  FOLDER_PATH: 'data/fs-peptide'}
    elif data_set == 'weather':
        country = trajectory_name
        folder_path = f'data/weather_data/{country}/'
        filename_list = [f'weather_{country}_{i}.csv' for i in range(1980, 2019 + 1)]

        if not os.path.isfile(folder_path + filename_list[file_element]):
            raw_data = pd.read_csv('data/weather_data.csv')
            os.makedirs(folder_path, exist_ok=True)
            print('INFO: Created directory ', folder_path)

            wp.get_trajectories_per_year(raw_data, 'utc_timestamp', country)

        kwargs = {FILENAME: filename_list[file_element],
                  FOLDER_PATH: folder_path}
    else:
        raise ValueError(f'No data trajectory was found with the name `{trajectory_name}`.')

    if SUBSET_LIST in params.keys():
        subset_indexes: list = params[SUBSET_LIST]
        if file_element in subset_indexes:
            subset_indexes.remove(file_element)  # file_element already on kwargs
        filename_list = [filename_list[i] for i in subset_indexes if i < len(filename_list)]
    else:
        filename_list.pop(file_element)  # file_element already on kwargs
    kwargs[PARAMS] = params
    return filename_list, kwargs


def get_model_params_list(alg_json_file: str) -> list[dict]:
    """
    Loads the list of model configurations given from a json file or the default list of dictionary from the code.
    @param alg_json_file: str
        Path to the json data with the running configuration
    @return: list[dict]
        list of model configurations
    """
    if alg_json_file is not None:
        return json.load(open(alg_json_file))
    else:
        warnings.warn('The default model parameter list is used instead of a .json-file.\n'
                      '  Use a configuration from the `config_files`-folder together with the args `-a`.')
        return [
            # Old Class-algorithms with parameters, not strings:
            # USE_STD: True, CENTER_OVER_TIME: False (only for tensor),
            # {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, USE_STD: True, CENTER_OVER_TIME: False},

            # Original Algorithms
            # {ALGORITHM_NAME: 'original_pca', NDIM: MATRIX_NDIM},
            # {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, USE_STD: False, ABS_EVAL_SORT: False},
            # {ALGORITHM_NAME: 'original_tica', NDIM: MATRIX_NDIM},
            # {ALGORITHM_NAME: 'tica', LAG_TIME: 10, NDIM: MATRIX_NDIM, USE_STD: False,
            #  ABS_EVAL_SORT: False},

            # raw MATRIX models
            # {ALGORITHM_NAME: 'pca', NDIM: MATRIX_NDIM},
            # {ALGORITHM_NAME: 'tica', NDIM: MATRIX_NDIM, LAG_TIME: params[LAG_TIME]},

            # raw TENSOR models
            # {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM},
            # {ALGORITHM_NAME: 'tica', NDIM: TENSOR_NDIM, LAG_TIME: params[LAG_TIME]},

            # *** Parameters
            # KERNEL: KERNEL_ONLY, KERNEL_DIFFERENCE, KERNEL_MULTIPLICATION
            # KERNEL_TYPE: MY_GAUSSIAN, MY_EXPONENTIAL, MY_LINEAR, MY_EPANECHNIKOV, (GAUSSIAN, EXPONENTIAL, ...)
            # COV_FUNCTION: np.cov, np.corrcoef, utils.matrix_tools.co_mad
            # NTH_EIGENVECTOR: int
            # LAG_TIME: int
            # *** Boolean Parameters:
            # CORR_KERNEL, ONES_ON_KERNEL_DIAG, USE_STD, CENTER_OVER_TIME, EXTRA_DR_LAYER

            {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, KERNEL: KERNEL_ONLY, KERNEL_TYPE: MY_GAUSSIAN},
            # {ALGORITHM_NAME: 'tica', NDIM: TENSOR_NDIM, KERNEL: KERNEL_ONLY, LAG_TIME: params[LAG_TIME]},
        ]


def get_param_grid(param_grid_json_file: str = None) -> list:
    """
    This method loads the parameter grid to use it to find the best parameters for an algorithm.
    @param param_grid_json_file: str
        Path and file location of the json file
    @return: list
        returns the parameter grid
    """
    if param_grid_json_file is not None:
        return json.load(open(param_grid_json_file))
    else:
        warnings.warn('The default parameter grid list is used instead of a .json-file.\n'
                      '  Use a configuration from the `config_files`-folder.')
        return [
            {
                ALGORITHM_NAME: ['pca', 'tica'],
                KERNEL: [None],
            }, {
                ALGORITHM_NAME: ['pca', 'tica', 'kica'],
                LAG_TIME: [10],
                KERNEL: [KERNEL_DIFFERENCE, KERNEL_MULTIPLICATION, KERNEL_ONLY],
                KERNEL_TYPE: [MY_LINEAR, MY_GAUSSIAN, MY_EXPONENTIAL, MY_EPANECHNIKOV],
                ONES_ON_KERNEL_DIAG: [True, False],
                # EXTRA_DR_LAYER: [False, True],
                # EXTRA_LAYER_ON_PROJECTION: [False, True],
                ABS_EVAL_SORT: [False, True]
            }
        ]


def get_data_class(params: dict, kwargs: dict) -> [DataTrajectory, SubTrajectoryDecorator]:
    if DATA_SET in params.keys():
        data_set_name: str = params[DATA_SET]

        if data_set_name.startswith('sub'):
            if QUANTITY in params.keys():
                kwargs[QUANTITY] = params[QUANTITY]
            if TIME_WINDOW_SIZE in params.keys():
                kwargs[TIME_WINDOW_SIZE] = params[TIME_WINDOW_SIZE]
            if PART_COUNT in params.keys():
                kwargs[PART_COUNT] = params[PART_COUNT]

        if data_set_name == "weather":
            return WeatherTrajectory(**kwargs)
        elif data_set_name == "sub_protein":
            return SubTrajectoryDecorator(data_trajectory=ProteinTrajectory(**kwargs), **kwargs)
        else:  # "protein"
            return ProteinTrajectory(**kwargs)
    else:
        return ProteinTrajectory(**kwargs)
