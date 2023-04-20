import json
import warnings

from utils.param_keys.param_key import *
from utils.param_keys.run_options import *


def get_run_params(alg_params_json):
    if alg_params_json is not None:
        return json.load(open(alg_params_json))
    else:
        warnings.warn('The default run option is used instead of a .json-file.\n'
                      '  Use a configuration from the `config_files`-folder together with the args `-o`.')
        return {
            RUN_OPTION: COMPARE,
            PLOT_TYPE: COLOR_MAP,  # 'heat_map', 'color_map', '3d_map', 'explained_var_plot'
            PLOT_TICS: True,  # True, False
            CARBON_ATOMS_ONLY: True,  # True, False
            INTERACTIVE: True,  # True, False
            N_COMPONENTS: 2,
            BASIS_TRANSFORMATION: False,
            USE_ANGLES: False,
            TRAJECTORY_NAME: 'prot2',
            FILE_ELEMENT: 0,
        }


def get_files_and_kwargs(params):
    """
    This method returns the filename list of the trajectory and generates the kwargs for the DataTrajectory.
    The method is individually created for the available data set.
    Add new trajectory options, if different data set are used.
    @param params: run params
    @return:
    """
    try:
        trajectory_name = params[TRAJECTORY_NAME]
        file_element = params[FILE_ELEMENT]
    except KeyError as e:
        raise KeyError(f'Run option parameter is missing the key: `{e}`. This parameter is mandatory.')

    if trajectory_name == '2f4k':
        filename_list = [f'2F4K-0-protein-{i:03d}.dcd' for i in range(0, 62)]  # + ['tr3_unfolded.xtc',
        #   'tr8_folded.xtc']
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
        filename_list = [f'2WAV-0-protein-{i:03d}.dcd' for i in range(0, 136)]
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
    else:
        raise ValueError(f'No data trajectory was found with the name `{trajectory_name}`.')

    filename_list.pop(file_element)
    kwargs[PARAMS] = params
    return filename_list, kwargs


def get_model_params_list(alg_json_file):
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

            {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, KERNEL: KERNEL_ONLY, ANALYSE_PLOT_TYPE: FITTED_KERNEL_CURVES,
             KERNEL_TYPE: MY_GAUSSIAN},
            # {ALGORITHM_NAME: 'tica', NDIM: TENSOR_NDIM, KERNEL: KERNEL_ONLY, LAG_TIME: params[LAG_TIME]},
        ]


def get_param_grid():
    param_grid = [
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
    return param_grid
