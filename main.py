import json
from datetime import datetime

import numpy as np

from my_tsne import TrajectoryTSNE
from plotter import TrajectoryPlotter, ArrayPlotter
from trajectory import DataTrajectory, TopologyConverter
from analyse import MultiTrajectoryAnalyser, SingleTrajectoryAnalyser, AnalyseResultLoader
from utils.param_key import *

COMPARE = 'compare'
COMPARE_TRANSFORMATION_ON_SAME_FITTING = 'compare_transformation_on_same_fitting'
MULTI_COMPARE_ALL_PCS = 'multi_compare_all_pcs'
MULTI_COMPARE_COMBO_PCS = 'multi_compare_combo_pcs'
MULTI_COMPARE_SOME_PCS = 'multi_compare_some_pcs'
COMPARE_WITH_TLTSNE = 'compare_with_tltsne'
PLOT_WITH_SLIDER = 'plot_with_slider'
PLOT_RECONSTRUCTED_WITH_SLIDER = 'plot_reconstructed_with_slider'
COMPARE_WITH_CA_ATOMS = 'compare_with_carbon_alpha_atoms'
BASE_TRANSFORMATION = 'base_transformation'
CALCULATE_PEARSON_CORRELATION_COEFFICIENT = 'calculate_pcc'
PARAMETER_GRID_SEARCH = 'parameter_grid_search'
LOAD_ANALYSE_RESULTS_DICT = 'load_analyse_result_dict'
MULTI_GRID_SEARCH = 'multi_parameter_grid_search'
MULTI_RECONSTRUCT_WITH_DIFFERENT_EV = 'multi_reconstruct_with_different_eigenvector'
MULTI_MEDIAN_RECONSTRUCTION_SCORES_ON_DIFF_FITTED = 'multi_median_reconstruction_scores'
MULTI_KERNEL_COMPARE = 'multi_kernel_compare'
MULTI_RECONSTRUCTION_ERROR_ON_SAME_TRAJ = 'multi_reconstruction_error_on_same_trajectory'
MULTI_MEDIAN_RECONSTRUCTION_ERROR_ON_SAME_TRAJ = 'multi_median_reconstruction_error_on_same_trajectory'


def main():
    print(f'Starting time: {datetime.now()}')
    # TODO: Argsparser for options
    run_params_json = None  # NotYetImplemented
    alg_params_json = None
    # alg_params_json = 'config_files/algorithm/pca+gaussian_kernels.json'  # None or filename
    # alg_params_json = 'config_files/algorithm/algorithm_parameters_list.json'
    # alg_params_json = 'config_files/algorithm/tica_models.json'
    # alg_params_json = 'config_files/algorithm/pca+tica+all_kernels.json'  # None or filename
    # alg_params_json = 'config_files/algorithm/all_my_kernels_only.json'  # None or filename

    result_load_file = None  # '2023-02-26_23.02.56_RE_diff_traj_evs/median_RE_over_trajectories_on_other.npz'
    run_option = COMPARE

    run_params = {
        PLOT_TYPE: COLOR_MAP,  # 'heat_map', 'color_map', '3d_map', 'explained_var_plot'
        PLOT_TICS: True,  # True, False
        STANDARDIZED_PLOT: True,  # True, False
        CARBON_ATOMS_ONLY: True,  # True, False
        INTERACTIVE: True,  # True, False
        N_COMPONENTS: 2,
        LAG_TIME: 10,
        TRUNCATION_VALUE: 0,  # deprecated
        BASIS_TRANSFORMATION: False,
        USE_ANGLES: False,
        TRAJECTORY_NAME: '2f4k',
        FILE_ELEMENT: 64,
    }

    filename_list, kwargs = get_files_and_kwargs(run_params)
    model_params_list = get_model_params_list(alg_params_json, run_params)
    param_grid = get_param_grid()
    run(run_option, kwargs, run_params, model_params_list, filename_list, param_grid, result_load_file)
    print(f'Finishing time: {datetime.now()}')


def get_files_and_kwargs(params):
    trajectory_name = params[TRAJECTORY_NAME]
    file_element = params[FILE_ELEMENT]
    if trajectory_name == '2f4k':
        filename_list = [f'2F4K-0-protein-{i:03d}.dcd' for i in range(0, 62 + 1)] + ['tr3_unfolded.xtc',
                                                                                     'tr8_folded.xtc']
        kwargs = {'filename': filename_list[file_element], 'topology_filename': '2f4k.pdb', 'folder_path': 'data/2f4k',
                  'params': params}
    elif trajectory_name == 'prot2':
        filename_list = ['prod_r1_nojump_prot.xtc', 'prod_r2_nojump_prot.xtc', 'prod_r3_nojump_prot.xtc']
        kwargs = {'filename': filename_list[file_element], 'topology_filename': 'prod_r1_pbc_fit_prot_last.pdb',
                  'folder_path': 'data/ProtNo2', 'params': params}
    elif trajectory_name == 'savinase':
        filename_list = ['savinase_1.xtc', 'savinase_2.xtc']
        kwargs = {'filename': filename_list[file_element], 'topology_filename': 'savinase.pdb',
                  'folder_path': 'data/Savinase', 'params': params}
    elif trajectory_name == '2wav':
        filename_list = [f'2WAV-0-protein-{i:03d}.dcd' for i in range(36, 100)]
        kwargs = {'filename': filename_list[file_element], 'topology_filename': '2wav.pdb',
                  'folder_path': 'data/2WAV-0-protein', 'params': params, 'atoms': list(range(710))}
    elif trajectory_name == '5i6x':
        filename_list = ['protein.xtc', 'system.xtc']
        kwargs = {'filename': filename_list[file_element], 'topology_filename': 'protein.pdb',
                  'folder_path': 'data/ser-tr', 'params': params}
    elif trajectory_name == 'fs-peptide':
        filename_list = [f'trajectory-{i}.xtc' for i in range(1, 28 + 1)]
        kwargs = {'filename': filename_list[file_element], 'topology_filename': 'fs-peptide.pdb',
                  'folder_path': 'data/fs-peptide', 'params': params}
    else:
        raise ValueError(f'No data trajectory was found with the name `{trajectory_name}`.')
    filename_list.pop(file_element)
    return filename_list, kwargs


def get_model_params_list(alg_json_file, params):
    if alg_json_file is not None:
        return json.load(open(alg_json_file))
        # return json.load(open('algorithm_parameters_list.json'))
    else:
        return [
            # Old Class-algorithms with parameters, not strings:
            # USE_STD: True, CENTER_OVER_TIME: False (only for tensor),
            # {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, USE_STD: True, CENTER_OVER_TIME: False},

            # Original Algorithms
            # {ALGORITHM_NAME: 'original_pca', NDIM: MATRIX_NDIM},
            # {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, USE_STD: False, ABS_EVAL_SORT: False},
            # {ALGORITHM_NAME: 'original_tica', NDIM: MATRIX_NDIM},
            # {ALGORITHM_NAME: 'tica', LAG_TIME: params[LAG_TIME], NDIM: MATRIX_NDIM, USE_STD: False,
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

            {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, KERNEL: KERNEL_ONLY,
             ANALYSE_PLOT_TYPE: EIGENVECTOR_MATRIX_ANALYSE},
            # {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, KERNEL: KERNEL_ONLY, ANALYSE_PLOT_TYPE: 'CMM', USE_STD: False},
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


def run(run_option, kwargs, params, model_params_list, filename_list, param_grid, result_load_file):
    if run_option == 'covert_to_pdb':
        kwargs = {'filename': 'protein.xtc', 'topology_filename': 'protein.gro',
                  'goal_filename': 'protein.pdb', 'folder_path': 'data/ser-tr'}
        tc = TopologyConverter(**kwargs)
        tc.convert()
    elif run_option == COMPARE_WITH_TLTSNE:
        tr = TrajectoryTSNE(**kwargs)
        tr.compare('tsne')
    elif run_option == PLOT_WITH_SLIDER:
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr).data_with_timestep_slider(min_max=None)  # [0, 1000]
    elif run_option == PLOT_RECONSTRUCTED_WITH_SLIDER:
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr, reconstruct_params=model_params_list[1]).data_with_timestep_slider()
    elif run_option == COMPARE:
        if kwargs['params'][N_COMPONENTS] != 2:
            raise ValueError(f'The parameter `{N_COMPONENTS}` has to be 2')
        tr = DataTrajectory(**kwargs)
        SingleTrajectoryAnalyser(tr).compare(model_params_list)
    elif run_option == COMPARE_WITH_CA_ATOMS:
        tr = DataTrajectory(**kwargs)
        SingleTrajectoryAnalyser(tr).compare_with_carbon_alpha_and_all_atoms('pca')
    elif run_option == BASE_TRANSFORMATION:
        tr = DataTrajectory(**kwargs)
        SingleTrajectoryAnalyser(tr).compare_with_basis_transformation(['tica'])
    elif run_option == CALCULATE_PEARSON_CORRELATION_COEFFICIENT:
        tr = DataTrajectory(**kwargs)
        SingleTrajectoryAnalyser(tr).calculate_pearson_correlation_coefficient()
    elif run_option == PARAMETER_GRID_SEARCH:
        tr = DataTrajectory(**kwargs)
        SingleTrajectoryAnalyser(tr).grid_search(param_grid)
    elif run_option == LOAD_ANALYSE_RESULTS_DICT:
        from_other_traj = False
        plot_dict = AnalyseResultLoader(params[TRAJECTORY_NAME]).load_npz(
            # '2023-03-01_02.45.30_RE-same_all-models/median_RE_over_trajectories_on_same.npz'
            '2023-02-26_20.31.59_RE_diff_pca+tica/median_RE_over_trajectories_on_other.npz'
        )
        update_dict = True
        if update_dict:
            plot_dict.update(AnalyseResultLoader(params[TRAJECTORY_NAME]).load_npz(
                '2023-02-27_03.04.39_RE_diff-evs_mean-ax0_use-original-mean/median_RE_over_trajectories_on_other.npz'
                # '2023-02-25_06.01.36_RE_diff_traj/median_RE_over_trajectories_on_same.npz'
                # '2023-03-01_22.19.05_RE-same_my-tica-models/median_RE_over_trajectories_on_same.npz'
                # '2023-03-03_23.53.55_RMSE-kernel_compare/compare_rmse_kernel.npz'
            ))

        filter_by_indices = True
        if filter_by_indices:
            indices = [
                '[PCA, output dimension = 105]      ',
                '[TICA, lag = 10; max. output dim. = 105]',
                'Tensor-pca, my_gaussian-only       ',
                # 'Tensor-pca, my_gaussian-diff       ',
                # 'Tensor-pca, my_gaussian-multi      ',
                'Tensor-pca, my_gaussian-only-3rd_ev_eevd',
                'Tensor-pca, my_gaussian-only-2nd_layer_eevd',
                # 'Tensor-tica, my_gaussian-only      ',
                # 'Tensor-tica, my_gaussian-diff      ',
                # 'Tensor-tica, my_gaussian-multi     ',
                # BIN 'Tensor-tica, my_gaussian-only-3rd_ev_eevd',
                # BIN 'Tensor-tica, my_gaussian-only-2nd_layer_eevd'
                # 'my_gaussian', 'my_exponential', 'my_epanechnikov'
            ]
            plot_dict = {k: plot_dict[k] for k in indices}

        load_option = MULTI_KERNEL_COMPARE
        if load_option == MULTI_RECONSTRUCTION_ERROR_ON_SAME_TRAJ:
            ArrayPlotter(
                interactive=False,
                title_prefix=f'Compare Kernels ',
                x_label='trajectory Nr',
                y_label='RMSE of the fitting kernel',
                y_range=(0, 0.2),
            ).plot_merged_2ds(plot_dict, statistical_func=np.median)
        elif MULTI_RECONSTRUCTION_ERROR_ON_SAME_TRAJ:
            ArrayPlotter(
                interactive=False,
                title_prefix=f'Reconstruction Error (RE) ' +
                             (f'from {filename_list[0]}\n' if from_other_traj else '') +
                             f'on {params[N_COMPONENTS]} Principal Components ',
                x_label='number of principal components',
                y_label='median REs of the trajectories',
                y_range=(0, 1)
            ).plot_merged_2ds(plot_dict)
    elif run_option.startswith('multi'):
        kwargs_list = [kwargs]
        if result_load_file is None:
            for filename in filename_list:
                new_kwargs = kwargs.copy()
                new_kwargs['filename'] = filename
                kwargs_list.append(new_kwargs)

        if run_option == 'multi_trajectory':
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_pcs(model_params_list)
        elif run_option == MULTI_COMPARE_COMBO_PCS:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_trajectory_combos(traj_nrs=[3, 8, 63, 64], model_params_list=model_params_list,
                                          pc_nr_list=[2, 9, 30])
        elif run_option == MULTI_COMPARE_ALL_PCS:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_all_trajectory_eigenvectors(traj_nrs=None, model_params_list=model_params_list,
                                                    pc_nr_list=None, merged_plot=True)
        elif run_option == MULTI_COMPARE_SOME_PCS:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_all_trajectory_eigenvectors(traj_nrs=None, model_params_list=model_params_list,
                                                    pc_nr_list=[2, 9, 30])
        elif run_option == MULTI_GRID_SEARCH:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.grid_search(param_grid)
        elif run_option == MULTI_RECONSTRUCT_WITH_DIFFERENT_EV:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_reconstruction_scores(model_params_list, other_traj_index=params[FILE_ELEMENT])
        elif run_option == MULTI_MEDIAN_RECONSTRUCTION_SCORES_ON_DIFF_FITTED:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_median_reconstruction_scores(model_params_list, other_traj_index=params[FILE_ELEMENT])
        elif run_option == MULTI_KERNEL_COMPARE:
            kernel_names = [MY_GAUSSIAN, MY_EXPONENTIAL, MY_EPANECHNIKOV]
            model_params = {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, ANALYSE_PLOT_TYPE: 'something'}
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_kernel_fitting_scores(kernel_names, model_params)
        elif run_option == MULTI_RECONSTRUCTION_ERROR_ON_SAME_TRAJ:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_reconstruction_scores(model_params_list)
        elif run_option == MULTI_MEDIAN_RECONSTRUCTION_ERROR_ON_SAME_TRAJ:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_median_reconstruction_scores(model_params_list)


if __name__ == '__main__':
    main()
