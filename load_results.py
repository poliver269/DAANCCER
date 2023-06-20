import os
from pathlib import Path

import numpy as np

from analyse import AnalyseResultLoader
from utils.param_keys import TRAJECTORY_NAME, PLOT_TYPE, N_COMPONENTS, FILENAME, PARAMS, DUMMY_ZERO, INTERACTIVE, \
    PLOT_FOR_PAPER
from utils.param_keys.run_options import *
from plotter import ModelResultPlotter, ArrayPlotter
from utils import pretify_dict_model


def load_list_of_dicts(sub_dir: str, params: dict):
    # directories = ['2023-03-05_23.44.24', '2023-03-05_21.51.49', '2023-03-05_23.46.32']  # prot2
    # sub_dir = 'fit-on-one-transform-on-all_original-pca'
    # sub_dir = 'fit-on-one-transform-on-all_tensor-pca-gaussian-only'
    sub_dir_path = Path('analyse_results') / params[TRAJECTORY_NAME] / sub_dir[DUMMY_ZERO]
    directories = os.listdir(sub_dir_path)
    list_of_list = []
    for directory in directories:
        list_of_list.append(
            AnalyseResultLoader(
                trajectory_name=params[TRAJECTORY_NAME],
                sub_dir=sub_dir
            ).load_npz_files_in_directory(directory)
        )
    ModelResultPlotter(
        interactive=params[INTERACTIVE],
        title_prefix=sub_dir,
        for_paper=params[PLOT_FOR_PAPER]
    ).plot_multi_projections(list_of_list, params[PLOT_TYPE],
                             center_plot=False, sub_parts=False, show_model_properties=False)


def load_analyse_results_dict(result_load_files: list, kwargs: dict):
    """
    Loads and plots existing runs.
    @param result_load_files: list
        contains a list of different .npz files which should be loaded
    @param kwargs: dict
        contains the running parameters and the trajectory file information
    @return:
    """
    from_other_traj = False
    plot_dict = AnalyseResultLoader(kwargs[PARAMS][TRAJECTORY_NAME]).load_npz_by_filelist(result_load_files)

    load_option = MULTI_QUALITATIVE_TRANSFORMATION_ON_SAME_FITTING  # TODO: load_option in run_params kwargs[PARAMS]
    if load_option == MULTI_RE_FIT_TRANSFORMED:
        ArrayPlotter(
            interactive=False,
            title_prefix=f'Compare Kernels ',
            x_label='trajectory Nr',
            y_label='RMSE of the fitting kernel',
            y_range=(0, 1),
            show_grid=True
        ).plot_merged_2ds(plot_dict, statistical_func=np.median)
    elif load_option == MULTI_QUALITATIVE_TRANSFORMATION_ON_SAME_FITTING:
        ArrayPlotter(
            interactive=kwargs[PARAMS][INTERACTIVE],
            title_prefix=f'Reconstruction Error (RE) ' +
                         (f'from {kwargs[FILENAME]}\n' if from_other_traj else '') +
                         f'on {kwargs[PARAMS][N_COMPONENTS]} Principal Components ',
            x_label='number of principal components',
            y_label='median REs of the trajectories',
            y_range=(0, 1),
            for_paper=kwargs[PARAMS][PLOT_FOR_PAPER]
        ).plot_merged_2ds(plot_dict)
