from datetime import datetime

from trajectory import ProteinTopologyConverter
from plotter import ProteinPlotter
from analyse import MultiTrajectoryAnalyser, SingleTrajectoryAnalyser, SingleProteinTrajectoryAnalyser, \
    MultiSubTrajectoryAnalyser
from utils.default_argparse import ArgParser
from utils.errors import InvalidRunningOptionError
from utils.param_keys import *
from utils.param_keys.analyses import ANALYSE_PLOT_TYPE, KERNEL_COMPARE
from utils.param_keys.kernel_functions import MY_GAUSSIAN, MY_EPANECHNIKOV, MY_EXPONENTIAL
from utils.param_keys.model import ALGORITHM_NAME, NDIM
from utils.param_keys.run_options import *
import config
import load_results


def main():
    print(f'Starting time: {datetime.now()}')

    args = ArgParser().parse('DAANCCER main')

    run_params = config.get_run_params(args.run_params_json)
    filename_list, kwargs = config.get_files_and_kwargs(run_params)

    if args.result_load_files is not None:
        load(args.result_load_files, kwargs)
    elif args.loading_directory is not None:
        load_directory(args.loading_directory, kwargs)
    else:
        model_params_list = config.get_model_params_list(args.alg_params_json)
        run(kwargs, model_params_list, filename_list)

    print(f'Finishing time: {datetime.now()}')


def run(kwargs: dict, model_params_list: list, filename_list: list):
    """
    This function chooses the running option for the program.
    @param kwargs: dict
        contains the running parameters and the trajectory file information
    @param model_params_list: list
        contains a list of model parameters, which are used to analyse this different models.
    @param filename_list: list
        contains the list of the filenames to load multiple trajectories
    @return:
    """
    params = kwargs[PARAMS]
    run_option = params[RUN_OPTION]
    if run_option.startswith(MULTI):
        run_multi_analyse(filename_list, model_params_list, kwargs)
    else:
        data_class = config.get_data_class(params, kwargs)
        if run_option == 'covert_to_pdb':
            kwargs = {FILENAME: 'protein.xtc', TOPOLOGY_FILENAME: 'protein.gro',
                      GOAL_FILENAME: 'protein.pdb', FOLDER_PATH: 'data/ser-tr'}
            tc = ProteinTopologyConverter(**kwargs)
            tc.convert()
        elif run_option == PLOT_WITH_SLIDER:
            ProteinPlotter(data_class, standardize=True).data_with_timestep_slider(min_max=None)  # [0, 1000]
        elif run_option == PLOT_RECONSTRUCTED_WITH_SLIDER:
            ProteinPlotter(data_class, reconstruct_params=model_params_list[1]).data_with_timestep_slider()
        elif run_option == COMPARE:
            if params[N_COMPONENTS] != 2:
                raise ValueError(f'The parameter `{N_COMPONENTS}` has to be 2, but it\'s {params[N_COMPONENTS]}.')
            SingleTrajectoryAnalyser(data_class, params).compare(model_params_list)
        elif run_option == 'plot_kernels':
            SingleTrajectoryAnalyser(data_class, params).compare(model_params_list, plot_results=False)
        elif run_option == COMPARE_WITH_CA_ATOMS:
            SingleProteinTrajectoryAnalyser(data_class, params).compare_with_carbon_alpha_and_all_atoms(model_params_list)
        elif run_option == BASE_TRANSFORMATION:
            SingleProteinTrajectoryAnalyser(data_class, params).compare_with_basis_transformation(model_params_list)
        elif run_option == PARAMETER_GRID_SEARCH:
            param_grid = config.get_param_grid()
            SingleTrajectoryAnalyser(data_class, params).grid_search(param_grid)
        elif run_option == TRAJECTORY_SUBSET_ANALYSIS:
            SingleTrajectoryAnalyser(data_class, params).compare_trajectory_subsets(model_params_list)
        else:
            raise InvalidRunningOptionError(f'The run_option: `{run_option}` in the (json) configuration '
                                            f'does not exists or it is not a loading option.\n')


def run_multi_analyse(filename_list, model_params_list, kwargs):
    run_option = kwargs[PARAMS][RUN_OPTION]
    kwargs_list = [kwargs]
    for filename in filename_list:
        new_kwargs = kwargs.copy()
        new_kwargs[FILENAME] = filename
        kwargs_list.append(new_kwargs)

    if run_option == 'multi_trajectory':
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_pcs(model_params_list)
    elif run_option == MULTI_COMPARE_COMBO_PCS:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_trajectory_combos(traj_nrs=[3, 8, 63, 64], model_params_list=model_params_list,
                                      pc_nr_list=[2, 9, 30])
    elif run_option == MULTI_COMPARE_ALL_PCS:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_all_trajectory_eigenvectors(traj_nrs=None, model_params_list=model_params_list,
                                                pc_nr_list=None, merged_plot=True)
    elif run_option == MULTI_COMPARE_SOME_PCS:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_all_trajectory_eigenvectors(traj_nrs=None, model_params_list=model_params_list,
                                                pc_nr_list=[2, 30, 104])
    elif run_option == MULTI_GRID_SEARCH:
        param_grid = config.get_param_grid()
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.grid_search(param_grid)
    elif run_option == MULTI_RE_FIT_ON_ONE_TRANSFORM_ON_ALL:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_reconstruction_scores(model_params_list, fit_transform_re=False)
    elif run_option == MULTI_MEDIAN_RE_FIT_ON_ONE_TRANSFORM_ON_ALL:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_median_reconstruction_scores(model_params_list, fit_transform_re=False)
    elif run_option == MULTI_KERNEL_COMPARE + '_on_same_model':
        kernel_names = [MY_GAUSSIAN, MY_EXPONENTIAL, MY_EPANECHNIKOV]
        model_params = {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM, ANALYSE_PLOT_TYPE: KERNEL_COMPARE}
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_kernel_fitting_scores_on_same_model(kernel_names, model_params)
    elif run_option == MULTI_KERNEL_COMPARE:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_kernel_fitting_scores(model_params_list)
    elif run_option == MULTI_RE_FIT_TRANSFORMED:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_reconstruction_scores(model_params_list)
    elif run_option == MULTI_MEDIAN_RE_FIT_TRANSFORMED:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_median_reconstruction_scores(model_params_list)
    elif run_option == MULTI_QUALITATIVE_TRANSFORMATION_ON_SAME_FITTING:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_results_on_same_fitting(model_params_list[DUMMY_ZERO], DUMMY_ZERO)
    elif run_option == MULTI_QUALITATIVE_PROJECTION_MATRIX:
        mtr = MultiTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mtr.compare_projection_matrix(model_params_list)
    elif run_option == 'multi_compare_re_on_small_parts':
        mst = MultiSubTrajectoryAnalyser(kwargs_list, kwargs[PARAMS])
        mst.compare_re_on_small_parts(model_params_list)
    else:
        raise InvalidRunningOptionError(f'The \"run_option\": \"{run_option}\" in the (json) configuration '
                                        f'does not exists or it is not a loading option.\n')


def load(result_load_files: list, kwargs: dict):
    """
    This function chooses the loading option of the files, how they should be loaded.
    @param result_load_files: list
        a list of file destinations
    @param kwargs: dict
        contains the running parameters and the trajectory file information
    """
    run_option = kwargs[PARAMS][RUN_OPTION]
    if run_option == LOAD_ANALYSE_RESULTS_DICT:
        load_results.load_analyse_results_dict(result_load_files, kwargs)
    else:
        raise InvalidRunningOptionError(f'The loading files are set to: {result_load_files},\n'
                                        f'but the run_option: `{run_option}` in the (json) configuration '
                                        f'does not exists or it is not a loading option.\n'
                                        f'Make sure, that the input arguments are '
                                        f'set correctly in combination with the run_option.')


def load_directory(directory_root: str, kwargs: dict):
    run_option = kwargs[PARAMS][RUN_OPTION]
    if run_option == LOAD_RE_OVER_COMPONENT_SPAN:
        load_results.load_re_over_component_span(directory_root, kwargs)
    elif run_option == LOAD_LIST_OF_DICTS:
        load_results.load_list_of_dicts(sub_dir=directory_root, params=kwargs[PARAMS])
    elif run_option == LOAD_FOOTOA_TWS:
        load_results.load_foo_toa_tws(directory_root, kwargs)
    elif run_option == LOAD_EIGENVECTOR_SIMILARITIES:
        load_results.load_eigenvector_similarities(directory_root, kwargs)
    else:
        raise InvalidRunningOptionError(f'The loading directory is set to: {directory_root},\n'
                                        f'but the run_option: `{run_option}` in the (json) configuration '
                                        f'does not exists or it is not a directory loading option.\n'
                                        f'Make sure, that the input arguments are '
                                        f'set correctly in combination with the run_option.')


if __name__ == '__main__':
    main()
