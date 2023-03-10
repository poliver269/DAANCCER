import os
import warnings
from datetime import datetime
from itertools import combinations
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from plotter import ArrayPlotter, MultiTrajectoryPlotter, ModelResultPlotter
from trajectory import DataTrajectory
from utils import statistical_zero
from utils.algorithms.tensor_dim_reductions import ParameterModel
from utils.errors import InvalidReconstructionException
from utils.matrix_tools import calculate_symmetrical_kernel_matrix, reconstruct_matrix
from utils.param_key import *


class SingleTrajectoryAnalyser:
    def __init__(self, trajectory):
        self.trajectory: DataTrajectory = trajectory

    def compare(self, model_parameter_list: list[dict, str], plot_results: bool = True) -> list[dict]:
        """
        Compares different models, with different input-parameters
        :param model_parameter_list: dict - model parameter dict, (str - specific algorithm name)
            List of different model-parameters
        :param plot_results: bool
        :return: The results of the models {MODEL, PROJECTION, EXPLAINED_VAR, INPUT_PARAMS}
        """
        model_results = []
        for model_parameters in model_parameter_list:
            try:
                model_results.append(self.trajectory.get_model_result(model_parameters))
            except np.linalg.LinAlgError as e:
                warnings.warn(f'Eigenvalue decomposition for model `{model_parameters}` could not be calculated:\n {e}')
            except AssertionError as e:
                warnings.warn(f'{e}')

        if plot_results:
            self.compare_with_plot(model_results)

        return model_results

    def compare_with_carbon_alpha_and_all_atoms(self, model_names):
        model_results = self.trajectory.get_model_results_with_changing_param(model_names, CARBON_ATOMS_ONLY)
        self.compare_with_plot(model_results)

    def compare_with_basis_transformation(self, model_params_list):
        model_results = self.trajectory.get_model_results_with_changing_param(model_params_list, BASIS_TRANSFORMATION)
        self.compare_with_plot(model_results)

    def compare_with_plot(self, model_results_list):
        ModelResultPlotter(self.trajectory).plot_models(
            model_results_list,
            plot_type=self.trajectory.params[PLOT_TYPE],
            plot_tics=self.trajectory.params[PLOT_TICS],
            components=self.trajectory.params[N_COMPONENTS]
        )

    def calculate_pearson_correlation_coefficient(self):
        mode = 'calculate_coefficients_than_mean_dimensions'
        coefficient_mean = self.trajectory.determine_coefficient_mean(mode)
        print('Fit Kernel on data...')
        d_matrix = calculate_symmetrical_kernel_matrix(coefficient_mean,
                                                       analyse_mode=self.trajectory.params[TRAJECTORY_NAME])
        weighted_alpha_coeff_matrix = coefficient_mean - d_matrix

        title_prefix = (f'{"Angles" if self.trajectory.params[USE_ANGLES] else "Coordinates"}. '
                        f'Pearson Coefficient. {mode}')
        ArrayPlotter(
            title_prefix=title_prefix,
            x_label='number of correlations',
            y_label='number of correlations'
        ).matrix_plot(weighted_alpha_coeff_matrix, as_surface=self.trajectory.params[PLOT_TYPE])

    def grid_search(self, param_grid):
        print('Searching for best model...')
        model = ParameterModel()
        inp = self.trajectory.data_input()  # Cannot train for different ndim at once
        cv = [(slice(None), slice(None))]  # get rid of cross validation
        grid = GridSearchCV(model, param_grid, cv=cv, verbose=1)
        grid.fit(inp, n_components=self.trajectory.params[N_COMPONENTS])
        AnalyseResultsSaver(trajectory_name=self.trajectory.params[TRAJECTORY_NAME],
                            filename=f'grid_search_{self.trajectory.filename[:-4]}').save_to_csv(grid.cv_results_)


class MultiTrajectoryAnalyser:
    def __init__(self, kwargs_list, params):
        self.trajectories: list[DataTrajectory] = [DataTrajectory(**kwargs) for kwargs in kwargs_list]
        print(f'Trajectories loaded time: {datetime.now()}')
        self.params: dict = params

    def compare_pcs(self, model_params_list: list):
        for model_parameters in model_params_list:
            principal_components = []
            for trajectory in self.trajectories:
                res = trajectory.get_model_result(model_parameters)
                principal_components.append(res['model'].eigenvectors)
            pcs = np.asarray(principal_components)
            MultiTrajectoryPlotter(interactive=False).plot_principal_components(model_parameters, pcs,
                                                                                self.params[N_COMPONENTS])

    def _get_trajectories_by_index(self, traj_nrs: [list[int], None]):
        if traj_nrs is None:  # take all the trajectories
            return self.trajectories
        else:
            sorted_traj_nrs = sorted(i for i in traj_nrs if i < len(self.trajectories))
            return list(itemgetter(*sorted_traj_nrs)(self.trajectories))

    @staticmethod
    def _get_trajectory_pairs(trajectories: list[DataTrajectory], model_params: dict) -> list:
        traj_results = []
        for trajectory in trajectories:
            res = trajectory.get_model_result(model_params)
            res.update({'traj': trajectory})
            traj_results.append(res)
        return list(combinations(traj_results, 2))

    @staticmethod
    def _get_all_similarities_from_trajectory_ev_pairs(trajectory_pairs: list[tuple],
                                                       pc_nr_list: list = None,
                                                       plot: bool = False):
        """
        Calculates and returns the similarity between trajectory pairs
        :param trajectory_pairs:
        :param pc_nr_list:
        :param plot:
        :return:
        """
        if plot and pc_nr_list is None:
            raise ValueError('Trajectories can not be compared to each other, because the `pc_nr_list` is not given.')

        all_similarities = []
        for trajectory_pair in trajectory_pairs:
            pc_0_matrix = trajectory_pair[0][MODEL].eigenvectors.T
            pc_1_matrix = trajectory_pair[1][MODEL].eigenvectors.T
            cos_matrix = cosine_similarity(np.real(pc_0_matrix), np.real(pc_1_matrix))
            sorted_similarity_indexes = linear_sum_assignment(-np.abs(cos_matrix))[1]
            assert len(sorted_similarity_indexes) == len(
                set(sorted_similarity_indexes)), "Not all eigenvectors have a unique most similar eigenvector pair."
            sorted_cos_matrix = cos_matrix[:, sorted_similarity_indexes]
            combo_pc_similarity = np.diag(sorted_cos_matrix)
            combo_sim_of_n_pcs = np.asarray([np.mean(np.abs(combo_pc_similarity[:nr_of_pcs]))
                                             for nr_of_pcs in range(len(combo_pc_similarity))])
            if plot:
                sorted_pc_nr_list = sorted(i for i in pc_nr_list if i < len(combo_sim_of_n_pcs))
                selected_sim_vals = {
                    nr_of_pcs: combo_sim_of_n_pcs[nr_of_pcs] if len(combo_pc_similarity) > nr_of_pcs
                    else "PC nr. not found" for nr_of_pcs in sorted_pc_nr_list}
                combo_similarity = np.mean(np.abs(combo_pc_similarity))
                sim_text = f'Similarity values: All: {combo_similarity},\n{selected_sim_vals}'
                print(sim_text)
                ArrayPlotter(
                    interactive=False,
                    title_prefix=f'{trajectory_pair[0]["model"]}\n'
                                 f'{trajectory_pair[0]["traj"].filename} & {trajectory_pair[1]["traj"].filename}\n'
                                 f'PC Similarity',
                    x_label='Principal Component Number',
                    y_label='Principal Component Number',
                    bottom_text=sim_text
                ).matrix_plot(cos_matrix)
            all_similarities.append(combo_sim_of_n_pcs)
        return np.asarray(all_similarities)

    def compare_all_trajectory_eigenvectors(
            self,
            traj_nrs: [list[int], None],
            model_params_list: list[dict],
            pc_nr_list: [list[int], None],
            merged_plot=False
    ):
        """
        This function compares different models.
        For each model, the similarity of the eigenvectors are calculated,
        which are fitted from different trajectories.
        :param traj_nrs:
            if None compare all the trajectories
            compare only the trajectories in a given list,
        :param model_params_list:
            different model parameters, which should be compared with each other
        :param pc_nr_list:
            gives a list of how many principal components should be compared with each other.
            If None, then all the principal components are compared
        :param merged_plot:
        :return:
        """
        trajectories = self._get_trajectories_by_index(traj_nrs)

        model_similarities = {}
        for model_params in model_params_list:
            trajectory_pairs = self._get_trajectory_pairs(trajectories, model_params)
            all_sim_matrix = self._get_all_similarities_from_trajectory_ev_pairs(trajectory_pairs)

            if merged_plot:
                model_similarities[str(trajectory_pairs[0][0][MODEL])] = np.mean(all_sim_matrix, axis=0)
            else:
                if pc_nr_list is None:
                    ArrayPlotter(
                        interactive=False,
                        title_prefix=f'{self.params[TRAJECTORY_NAME]}\n{model_params}\n'
                                     'Similarity value of all trajectories',
                        x_label='Principal component number',
                        y_label='Similarity value',
                    ).plot_2d(np.mean(all_sim_matrix, axis=0))
                else:
                    for pc_index in pc_nr_list:
                        tria = np.zeros((len(trajectories), len(trajectories)))
                        sim_text = f'Similarity of all {np.mean(all_sim_matrix[:, pc_index])}'
                        print(sim_text)
                        tria[np.triu_indices(len(trajectories), 1)] = all_sim_matrix[:, pc_index]
                        tria = tria + tria.T
                        ArrayPlotter(
                            interactive=False,
                            title_prefix=f'{self.params[TRAJECTORY_NAME]}\n{model_params}\n'
                                         f'Trajectory Similarities for {pc_index}-Components',
                            x_label='Trajectory number',
                            y_label='Trajectory number',
                            bottom_text=sim_text
                        ).matrix_plot(tria)
        if merged_plot:
            ArrayPlotter(
                interactive=False,
                title_prefix=f'Eigenvector Similarities',
                x_label='Principal Component Number',
                y_label='Similarity value',
                y_range=(0, 1)
            ).plot_merged_2ds(model_similarities)

    def compare_trajectory_combos(self, traj_nrs, model_params_list, pc_nr_list):
        """
        Compare the trajectory combos with each other
        :param traj_nrs:
        :param model_params_list:
        :param pc_nr_list:
        :return:
        """
        trajectories = self._get_trajectories_by_index(traj_nrs)

        for model_params in model_params_list:
            trajectory_pairs = self._get_trajectory_pairs(trajectories, model_params)
            self._get_all_similarities_from_trajectory_ev_pairs(trajectory_pairs, pc_nr_list, plot=True)

    def grid_search(self, param_grid):
        print('Searching for best model...')
        train_trajectories = self.trajectories[:int(len(self.trajectories) * .8)]
        # test_trajectories = self.trajectories[int(len(self.trajectories) * .8):]
        inp = train_trajectories[0].data_input()
        for trajectory in train_trajectories[1:]:
            np.concatenate((inp, trajectory.data_input()), axis=0)
        model = ParameterModel()
        cv = len(train_trajectories)
        grid = GridSearchCV(model, param_grid, cv=cv, verbose=1)
        grid.fit(inp, n_components=self.params[N_COMPONENTS])
        AnalyseResultsSaver(
            trajectory_name=self.params[TRAJECTORY_NAME],
            filename='grid_search_all'
        ).save_to_csv(grid.cv_results_, header=['params', 'mean_test_score', 'std_test_score', 'rank_test_score'])

    def compare_reconstruction_scores(self, model_params_list: list, fit_transform_re: bool = True):
        """
        Calculate the reconstruction error of the trajectories
        if from_other_traj is True than reconstruct from the model fitted on specific trajectory
        (0th-element in self trajectories)
        :param fit_transform_re:
        :param model_params_list:
        """

        # st = SingleTrajectoryAnalyser(self.trajectories[other_traj_index])
        # model_results: list = st.compare(model_params_list, plot_results=False)

        model_scores = {}
        for model_index, model_params in enumerate(model_params_list):
            # model: [ParameterModel, StreamingEstimationTransformer] = model_dict[MODEL]
            print(f'Calculating reconstruction errors ({model_params})...')
            model_dict_list = self._get_model_result_list(model_params)
            model_description = str(model_dict_list[DUMMY_ZERO][MODEL].describe())
            score_list = []
            for traj_index, trajectory in enumerate(self.trajectories):
                if fit_transform_re:
                    model_dict = model_dict_list[traj_index]
                    model = model_dict[MODEL]
                    matrix_projection = model_dict[PROJECTION]
                else:
                    model_dict = model_dict_list[DUMMY_ZERO]
                    model = model_dict[MODEL]
                    matrix_projection = None
                input_data = trajectory.data_input(model_dict[INPUT_PARAMS])
                score = self._get_reconstruction_score(model, input_data, matrix_projection)
                score_list.append(score)
            score_list = np.asarray(score_list)
            model_scores[f'{model_description:35}'] = score_list

        ArrayPlotter(
            interactive=False,
            title_prefix='Reconstruction Error (RE) ' + (
                '' if fit_transform_re else f'fit-on-one-transform-on-all\n'),
            x_label='trajectories',
            y_label='score',
            y_range=(0, 1)
        ).plot_merged_2ds(model_scores, np.median)

    @staticmethod
    def _get_reconstruction_score(model, input_data, data_projection=None, component=None):
        if isinstance(model, ParameterModel):
            if component is None:
                return model.score(input_data, data_projection)
            else:
                if data_projection is None:
                    data_projection = model.transform(input_data)
                reconstructed_tensor = model.reconstruct(data_projection[:, :component], component)
                input_data = model.convert_to_matrix(input_data)
                reconstructed_data = model.convert_to_matrix(reconstructed_tensor)
        else:
            if data_projection is None:
                data_projection = model.transform(input_data)
            if component is None:
                component = model.dim
            reconstructed_data = reconstruct_matrix(data_projection, model.eigenvectors, component,
                                                    mean=model.mean)
        return mean_squared_error(input_data, reconstructed_data, squared=False)

    def compare_median_reconstruction_scores(self, model_params_list: list[dict], fit_transform_re: bool = True):
        model_median_scores = self._calculate_median_scores(model_params_list, fit_transform_re)
        ArrayPlotter(
            interactive=False,
            title_prefix=f'Reconstruction Error (RE) ' +
                         ('' if fit_transform_re
                          else f'fit-on-one-transform-on-all\n') +
                         f'on {self.trajectories[DUMMY_ZERO].max_components} Principal Components ',
            x_label='number of principal components',
            y_label='median REs of the trajectories',
            y_range=(0, 1),
            xtick_start=1
        ).plot_merged_2ds(model_median_scores)

    def _calculate_median_scores(self, model_params_list: list[dict], fit_transform_re: bool = True):
        saver = AnalyseResultsSaver(trajectory_name=self.params[TRAJECTORY_NAME])

        model_median_scores = {}
        component_wise_scores = {}
        for model_params in model_params_list:
            print(f'Calculating median reconstruction errors ({model_params})...')
            model_dict_list = self._get_model_result_list(model_params)
            model_description = str(model_dict_list[DUMMY_ZERO][MODEL].describe())

            median_list = self._get_median_over_components_for_trajectories(model_dict_list,
                                                                            component_wise_scores,
                                                                            model_description,
                                                                            fit_transform_re)
            model_median_scores[f'{model_description:35}'] = np.asarray(median_list)
            saver.save_to_npz(model_median_scores, 'median_RE_over_trajectories_on_' +
                              ('fit-transform' if fit_transform_re else 'FooToa'))
            saver.save_to_npz(component_wise_scores, 'component_wise_RE_on_' +
                              ('fit-transform' if fit_transform_re else 'FooToa') + '_traj')
        return model_median_scores

    def _get_model_result_list(self, model_params: dict):
        model_dict_list = []
        for trajectory in self.trajectories:
            model_dict_list.append(trajectory.get_model_result(model_params, log=False))
        return model_dict_list

    def _get_median_over_components_for_trajectories(self,
                                                     model_dict_list: list[dict],
                                                     component_wise_scores: dict,
                                                     model_description: str,
                                                     fit_transform_re: bool = True) -> list:
        median_list = []
        for component in tqdm(range(1, self.trajectories[DUMMY_ZERO].max_components + 1)):
            score_list = []
            if str(component) not in component_wise_scores:
                component_wise_scores[str(component)] = {}
            try:
                for traj_index, fitted_trajectory in enumerate(self.trajectories):
                    model_dict = model_dict_list[traj_index]
                    model = model_dict[MODEL]
                    if fit_transform_re:
                        input_data = fitted_trajectory.data_input(model_dict[INPUT_PARAMS])
                        matrix_projection = model_dict[PROJECTION]
                        score = self._get_reconstruction_score(model, input_data, matrix_projection, component)
                    else:
                        transform_score = []
                        for transform_trajectory in self.trajectories:
                            input_data = transform_trajectory.data_input(model_dict[INPUT_PARAMS])
                            matrix_projection = model.transform(input_data)

                            transform_score.append(
                                self._get_reconstruction_score(model, input_data, matrix_projection, component))
                        score = np.median(transform_score)
                    score_list.append(score)
            except InvalidReconstructionException as e:
                warnings.warn(str(e))
                break
            component_wise_scores[str(component)][f'{model_description:35}'] = score_list
            median_list.append(np.median(score_list))
        return median_list

    def compare_kernel_fitting_scores(self, kernel_names, model_params, load_filename: [str, None] = None):
        if load_filename is None:
            kernel_accuracies = self._calculate_kernel_accuracies(kernel_names, model_params)
        else:
            kernel_accuracies = AnalyseResultLoader(self.params[TRAJECTORY_NAME]).load_npz(load_filename)
        ArrayPlotter(
            interactive=False,
            title_prefix=f'Compare Kernels ',
            x_label='trajectory Nr',
            y_label='RMSE of the fitting kernel',
            y_range=(0, 0.2),
        ).plot_merged_2ds(kernel_accuracies, statistical_func=np.median)

    def _calculate_kernel_accuracies(self, kernel_names, model_params):
        kernel_accuracies = {kernel_name: [] for kernel_name in kernel_names}
        for trajectory in self.trajectories:
            model, _ = trajectory.get_model_and_projection(model_params)
            for kernel_name in kernel_names:
                matrix = model.get_combined_covariance_matrix()
                variance = calculate_symmetrical_kernel_matrix(
                    matrix, statistical_zero, kernel_name,
                    analyse_mode=KERNEL_COMPARE)
                kernel_accuracies[kernel_name].append(variance)
        AnalyseResultsSaver(self.params[TRAJECTORY_NAME], filename='compare_rmse_kernel').save_to_npz(kernel_accuracies)
        return kernel_accuracies

    def compare_results_on_same_fitting(self, model_params, traj_index):
        fitting_trajectory = self.trajectories[traj_index]
        fitting_results = fitting_trajectory.get_model_result(model_params)
        model_results_list = []
        for trajectory_nr, trajectory in enumerate(self.trajectories):
            if trajectory_nr == traj_index:
                model_results_list.append(fitting_results)
            else:
                transform_results = fitting_results.copy()
                transform_results[PROJECTION] = fitting_results[MODEL].transform(trajectory.data_input(model_params))
                model_results_list.append(transform_results)
        st = SingleTrajectoryAnalyser(fitting_trajectory)
        st.compare_with_plot(model_results_list)
        saver = AnalyseResultsSaver(self.params[TRAJECTORY_NAME])
        for result_index, model_result in enumerate(model_results_list):
            saver.save_to_npz(
                model_result,
                new_filename=f'{self.trajectories[result_index].filename[:-4]}_transformed_on_{fitting_trajectory.filename[:-4]}'
            )


class AnalyseResultsSaver:
    def __init__(self, trajectory_name, filename=''):
        self.current_result_path: Path = Path('analyse_results') / trajectory_name / datetime.now().strftime(
            "%Y-%m-%d_%H.%M.%S")
        if not self.current_result_path.exists():
            os.makedirs(self.current_result_path)
        self.filename = filename

    def goal_filename(self, extension):
        return self.current_result_path / (self.filename + extension)

    def save_to_csv(self, result, header=None):
        if header is None:
            result_df = pd.DataFrame(result)
        else:
            result_df = pd.DataFrame(result)[header]
        goal_path = self.goal_filename('.csv')
        result_df.to_csv(goal_path)
        print(f'Results successfully saved into: {goal_path}')

    def save_to_npy(self, array: np.ndarray, new_filename=None):
        if new_filename is not None:
            self.filename = new_filename
        np.save(self.goal_filename('.npy'), array)

    def save_to_npz(self, dictionary: dict, new_filename=None):
        if new_filename is not None:
            self.filename = new_filename
        np.savez(self.goal_filename('.npz'), **dictionary)


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

    def load_npz_files_in_directory(self, directory_name):
        directory_path = self.get_load_path(directory_name)
        filename_list = os.listdir(directory_path)
        return self.load_npz_list(directory_name, filename_list)

    def load_npz_list(self, root_dir, filename_list):
        loaded_list = []
        for filename in filename_list:
            loaded_list.append(self.load_npz(Path(root_dir) / filename))
        return loaded_list
