from datetime import datetime

import config
import load_results
from plotter import TrajectoryPlotter
from trajectory import DataTrajectory, TopologyConverter
from analyse import MultiTrajectoryAnalyser, SingleTrajectoryAnalyser
from utils.default_argparse import ArgParser
from utils.param_keys.param_key import *
from utils.param_keys.run_options import *


def main():
    print(f'Starting time: {datetime.now()}')

    args = ArgParser().parse('DAANCCER main')
    run_params_json = args.run_params_json
    alg_params_json = args.alg_params_json
    result_load_files = args.result_load_files

    run_params = config.get_run_params(run_params_json)
    filename_list, kwargs = config.get_files_and_kwargs(run_params)

    if result_load_files is None:
        model_params_list = config.get_model_params_list(alg_params_json)
        run(kwargs, run_params, model_params_list, filename_list)
    else:
        load(result_load_files, kwargs, run_params)

    print(f'Finishing time: {datetime.now()}')


def run(kwargs, params, model_params_list, filename_list):
    run_option = params[RUN_OPTION]
    if run_option == 'covert_to_pdb':
        kwargs = {FILENAME: 'protein.xtc', TOPOLOGY_FILENAME: 'protein.gro',
                  GOAL_FILENAME: 'protein.pdb', FOLDER_PATH: 'data/ser-tr'}
        tc = TopologyConverter(**kwargs)
        tc.convert()
    elif run_option == PLOT_WITH_SLIDER:
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr, standardize=True).data_with_timestep_slider(min_max=None)  # [0, 1000]
    elif run_option == PLOT_RECONSTRUCTED_WITH_SLIDER:
        tr = DataTrajectory(**kwargs)
        TrajectoryPlotter(tr, reconstruct_params=model_params_list[1]).data_with_timestep_slider()
    elif run_option == COMPARE:
        if kwargs[PARAMS][N_COMPONENTS] != 2:
            raise ValueError(f'The parameter `{N_COMPONENTS}` has to be 2, but it\'s {kwargs[PARAMS][N_COMPONENTS]}.')
        tr = DataTrajectory(**kwargs)
        SingleTrajectoryAnalyser(tr, params).compare(model_params_list)
    elif run_option == COMPARE_WITH_CA_ATOMS:
        tr = DataTrajectory(**kwargs)
        SingleTrajectoryAnalyser(tr, params).compare_with_carbon_alpha_and_all_atoms(model_params_list)
    elif run_option == BASE_TRANSFORMATION:
        tr = DataTrajectory(**kwargs)
        SingleTrajectoryAnalyser(tr, params).compare_with_basis_transformation(model_params_list)
    elif run_option == PARAMETER_GRID_SEARCH:
        param_grid = config.get_param_grid()
        tr = DataTrajectory(**kwargs)
        SingleTrajectoryAnalyser(tr, params).grid_search(param_grid)
    elif run_option.startswith(MULTI):
        kwargs_list = [kwargs]
        for filename in filename_list:
            new_kwargs = kwargs.copy()
            new_kwargs[FILENAME] = filename
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
                                                    pc_nr_list=[2, 30, 104])
        elif run_option == MULTI_GRID_SEARCH:
            param_grid = config.get_param_grid()
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.grid_search(param_grid)
        elif run_option == MULTI_RE_FIT_ON_ONE_TRANSFORM_ON_ALL:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_reconstruction_scores(model_params_list, fit_transform_re=False)
        elif run_option == MULTI_MEDIAN_RE_FIT_ON_ONE_TRANSFORM_ON_ALL:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_median_reconstruction_scores(model_params_list, fit_transform_re=False)
        elif run_option == MULTI_KERNEL_COMPARE:
            kernel_names = [MY_GAUSSIAN, MY_EXPONENTIAL, MY_EPANECHNIKOV]
            model_params = {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, ANALYSE_PLOT_TYPE: KERNEL_COMPARE}
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_kernel_fitting_scores(kernel_names, model_params)
        elif run_option == MULTI_RE_FIT_TRANSFORMED:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_reconstruction_scores(model_params_list)
        elif run_option == MULTI_MEDIAN_RE_FIT_TRANSFORMED:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_median_reconstruction_scores(model_params_list)
        elif run_option == MULTI_QUALITATIVE_TRANSFORMATION_ON_SAME_FITTING:
            mtr = MultiTrajectoryAnalyser(kwargs_list, params)
            mtr.compare_results_on_same_fitting(model_params_list[0], 0)


def load(result_load_files, kwargs, params):
    run_option = params[RUN_OPTION]
    if run_option == LOAD_ANALYSE_RESULTS_DICT:
        load_results.load_analyse_results_dict(kwargs, params)
    elif run_option == LOAD_LIST_OF_DICTS:
        load_results.load_list_of_dicts(params)


if __name__ == '__main__':
    main()
