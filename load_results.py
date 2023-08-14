import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from analyse import AnalyseResultLoader
from utils.param_keys import TRAJECTORY_NAME, PLOT_TYPE, N_COMPONENTS, FILENAME, PARAMS, DUMMY_ZERO, INTERACTIVE, \
    PLOT_FOR_PAPER
from utils.param_keys.run_options import *
from plotter import ModelResultPlotter, ArrayPlotter


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
            x_label='Num. Components',
            y_label='Mean Squared Error',
            y_range=(0, 1),
            for_paper=kwargs[PARAMS][PLOT_FOR_PAPER]
        ).plot_merged_2ds(plot_dict)


def load_re_over_component_span(directory_root: str, kwargs: dict):
    npzs = AnalyseResultLoader(kwargs[PARAMS][TRAJECTORY_NAME]).load_npz_files_in_directory(directory_root)
    plot_dict = next(v for k, v in npzs.items() if 'median' in k)
    # a = ['PCA', 'DROPP', 'TICA', 'FastICA']
    # plot_dict = OrderedDict((key, plot_dict[key]) for key in a)
    error_band = next(v for k, v in npzs.items() if 'error_bands' in k)
    from_other_traj = True
    ArrayPlotter(
        interactive=kwargs[PARAMS][INTERACTIVE],
        title_prefix=f'Reconstruction Error (RE) ' +
                     (f'from {kwargs[FILENAME]}\n' if from_other_traj else '') +
                     f'on {kwargs[PARAMS][N_COMPONENTS]} Principal Components ',
        x_label='Num. Components',
        y_label='Mean Squared Error',
        # y_range=(0, 1),
        for_paper=kwargs[PARAMS][PLOT_FOR_PAPER]
    ).plot_merged_2ds(plot_dict, error_band=error_band)


def load_foo_toa_tws(directory_root: str, kwargs: dict):
    npzs = AnalyseResultLoader(kwargs[PARAMS][TRAJECTORY_NAME]).load_npz_files_in_directory(directory_root)
    plot_dict = next(v for k, v in npzs.items() if 'median' in k)
    error_band = next(v for k, v in npzs.items() if 'error_bands' in k)
    time_steps = next(v for k, v in npzs.items() if 'time_steps' in k)
    component_list = next(v for k, v in npzs.items() if 'component_list' in k)
    ArrayPlotter(
        interactive=kwargs[PARAMS][INTERACTIVE],
        title_prefix=f'FooToa on varying time window size',
        x_label='Time Window Size',
        y_label='Mean Squared Error',
        for_paper=kwargs[PARAMS][PLOT_FOR_PAPER],
        y_range=(0, 2)
    ).plot_matrix_in_2d(plot_dict, time_steps['time_steps'], component_list['component_list'], error_band)


def load_eigenvector_similarities(directory_root: str, kwargs: dict):
    npzs = AnalyseResultLoader(kwargs[PARAMS][TRAJECTORY_NAME]).load_npz_files_in_directory(directory_root)
    plot_dict = next(v for k, v in npzs.items() if 'eigenvector' in k)
    error_band = next(v for k, v in npzs.items() if 'error_bands' in k)
    ArrayPlotter(
        interactive=kwargs[PARAMS][INTERACTIVE],
        title_prefix=f'Eigenvector Similarities',
        x_label='Num. Components',
        y_label='Median Cosine Sim.',
        # y_range=(0, 1),
        for_paper=kwargs[PARAMS][PLOT_FOR_PAPER]
    ).plot_merged_2ds(plot_dict, error_band=error_band)


def load_result_and_merge_into_csv(directory_root: str, kwargs: dict):
    npzs = AnalyseResultLoader(kwargs[PARAMS][TRAJECTORY_NAME]).load_npz_files_in_directory(directory_root)
    plot_dict = next(v for k, v in npzs.items() if 'median' in k)
    # Renaming and reordering the keys in the dictionary
    if 'DAANCCER' in plot_dict:
        plot_dict['DROPP'] = plot_dict.pop('DAANCCER')
    if 'FastICA' in plot_dict:
        plot_dict['ICA'] = plot_dict.pop('FastICA')

    new_order = ['DROPP', 'PCA', 'ICA', 'TICA']
    # for algorithm_name in new_order:
    #
    #     if algorithm_name not in plot_dict.keys() and next(
    #             v for k, v in plot_dict.items() if algorithm_name in k) is not None:
    #         plot_dict[algorithm_name] = plot_dict.pop(next(k for k, v in plot_dict.items() if algorithm_name in k))

    plot_dict = {key: plot_dict[key] for key in new_order if key in plot_dict}

    # Extract relevant values from the plot_dict based on extraction_list
    extraction_list = [2, 5, 10, 15, 30]

    extracted_values = np.asarray([plot_dict[key][extraction_list] for key in plot_dict])
    reshaped_array = extracted_values.reshape(-1, order='F')

    goal_filename = 'analyse_results/FooToa-merged.csv'
    goal_path = Path(goal_filename)
    if not goal_path.exists():
        data_frame = pd.DataFrame()
    else:
        data_frame = pd.read_csv(goal_filename)

    # If the 'Method' column doesn't exist, set it with repeated values
    if 'Method' not in data_frame.columns:
        num_rows = len(extraction_list)
        data_frame['Method'] = np.tile(new_order, num_rows)

    # Append the reshaped_array as a new column to the DataFrame
    data_frame[kwargs[PARAMS][TRAJECTORY_NAME]] = reshaped_array

    data_frame.to_csv(goal_filename, index=False)


def load_merge_average(directory_root: str, kwargs: dict):
    loader = AnalyseResultLoader(kwargs[PARAMS][TRAJECTORY_NAME], directory_root)
    loader.merge_npz_files(f'analyse_results/{kwargs[PARAMS][TRAJECTORY_NAME]}/')
