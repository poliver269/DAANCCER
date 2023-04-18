import os
from pathlib import Path

import numpy as np

from analyse import AnalyseResultLoader
from main import MULTI_QUALITATIVE_TRANSFORMATION_ON_SAME_FITTING, MULTI_RE_FIT_TRANSFORMED
from plotter import ModelResultPlotter, ArrayPlotter
from utils import pretify_dict_model
from utils.param_key import TRAJECTORY_NAME, PLOT_TYPE, N_COMPONENTS


def load_list_of_dicts(params):
    # directories = ['2023-03-05_23.44.24', '2023-03-05_21.51.49', '2023-03-05_23.46.32']  # prot2
    sub_dir = 'fit-on-one-transform-on-all_original-pca'
    # sub_dir = 'fit-on-one-transform-on-all_tensor-pca-gaussian-only'
    sub_dir_path = Path('analyse_results') / params[TRAJECTORY_NAME] / sub_dir
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
        interactive=True,
        title_prefix=sub_dir,
        for_paper=True
    ).plot_multi_projections(list_of_list, params[PLOT_TYPE], center_plot=False)


def load_analyse_results_dict(kwargs, params):
    from_other_traj = False
    plot_dict = AnalyseResultLoader(params[TRAJECTORY_NAME]).load_npz(
        '2023-03-07_12.35.46_RE_fit-transform_pca+tensor-pca-gaussian-kernels/'
        'median_RE_over_trajectories_on_same.npz'
        # '2023-03-11_08.03.19_RE_footoa-all_pca+tensor-pca-gaussian/median_RE_over_trajectories_on_FooToa.npz'

        # '2023-03-08_13.21.12_RE_footoa-all_tica+tensor-tica-gaussian/median_RE_over_trajectories_on_FooToa.npz'
        # '2023-03-08_00.58.57_RE_fit-transform_tica+tensor-tica-gaussian/median_RE_over_trajectories_on_same.npz'
    )
    update_dict = False
    if update_dict:
        plot_dict.update(AnalyseResultLoader(params[TRAJECTORY_NAME]).load_npz(
            '2023-03-17_01.18.58_RE_fit-transform_evs/median_RE_over_trajectories_on_fit-transform.npz'

            # '2023-02-25_06.01.36_RE_diff_traj/median_RE_over_trajectories_on_other.npz'

            # '2023-03-01_22.19.05_RE-same_my-tica-models/median_RE_over_trajectories_on_other.npz'
            # '2023-03-02_01.20.26_RE-diff_my-tica-models/median_RE_over_trajectories_on_other.npz'
            # '2023-03-03_23.53.55_RMSE-kernel_compare/compare_rmse_kernel.npz'
        ))
    filter_by_indices = True
    if filter_by_indices:
        indices = [
            '[PCA, output dimension = 105]      ',
            # '[TICA, lag = 10; max. output dim. = 105]',
            'Tensor-pca, my_gaussian-only       ',
            'Tensor-pca, my_gaussian-diff       ',
            'Tensor-pca, my_gaussian-multi      ',
            # 'Tensor-pca, my_gaussian-only-3rd_ev_eevd',
            # 'Tensor-pca, my_gaussian-only-2nd_layer_eevd',
            # 'Tensor-tica, my_gaussian-only      ',
            # 'Tensor-tica, my_gaussian-diff      ',
            # 'Tensor-tica, my_gaussian-multi     ',
            # BIN 'Tensor-tica, my_gaussian-only-3rd_ev_eevd',
            # BIN 'Tensor-tica, my_gaussian-only-2nd_layer_eevd'
            # 'my_gaussian', 'my_exponential', 'my_epanechnikov'
        ]
        plot_dict = {k: plot_dict[k] for k in indices}
    pretyfied_dict = pretify_dict_model(plot_dict)
    load_option = MULTI_QUALITATIVE_TRANSFORMATION_ON_SAME_FITTING
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
            interactive=True,
            title_prefix=f'Reconstruction Error (RE) ' +
                         (f'from {kwargs["filename"]}\n' if from_other_traj else '') +
                         f'on {params[N_COMPONENTS]} Principal Components ',
            x_label='number of principal components',
            y_label='median REs of the trajectories',
            y_range=(0, 1),
            for_paper=True
        ).plot_merged_2ds(pretyfied_dict)
