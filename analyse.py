import os
import warnings
from datetime import datetime
from itertools import combinations
from operator import itemgetter
from pathlib import Path
import numpy as np
import pandas as pd
from pyemma.coordinates.data._base.transformer import StreamingEstimationTransformer
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV

from plotter import ArrayPlotter, MultiTrajectoryPlotter, TrajectoryPlotter
from trajectory import DataTrajectory
from utils.algorithms.tensor_dim_reductions import ParameterModel
from utils.matrix_tools import calculate_symmetrical_kernel_from_matrix
from utils.param_key import *


class SingleTrajectory:
    def __init__(self, trajectory):
        self.trajectory: DataTrajectory = trajectory

    def compare(self, model_parameter_list: list[dict, str], plot_results: bool = True) -> list[dict]:
        """
        Compares different models, with different input-parameters.
        :param model_parameter_list: dict - model parameter dict, (str - specific algorithm name)
            List of different model-parameters
        :param plot_results: bool
        :return: The results of the models {MODEL, PROJECTION, EXPLAINED_VAR}
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

    def compare_with_plot(self, model_projection_list):
        TrajectoryPlotter(self.trajectory).plot_models(
            model_projection_list,
            data_elements=[0],  # [0, 1, 2]
            plot_type=self.trajectory.params[PLOT_TYPE],
            plot_tics=self.trajectory.params[PLOT_TICS],
            components=self.trajectory.params[N_COMPONENTS]
        )

    def calculate_pearson_correlation_coefficient(self):
        mode = 'calculate_coefficients_than_mean_dimensions'
        coefficient_mean = self.trajectory.determine_coefficient_mean(mode)
        print('Fit Kernel on data...')
        d_matrix = calculate_symmetrical_kernel_from_matrix(coefficient_mean,
                                                            trajectory_name=self.trajectory.params[TRAJECTORY_NAME])
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
        inp = self.trajectory.data_input()  # Cannot train for
        cv = [(slice(None), slice(None))]  # get rid of cross validation
        grid = GridSearchCV(model, param_grid, cv=cv, verbose=1)
        grid.fit(inp, n_components=self.trajectory.params[N_COMPONENTS])
        _save_tuning_results(grid.cv_results_, name=self.trajectory.filename[:-4])


class MultiTrajectory:
    def __init__(self, kwargs_list, params):
        self.trajectories: list[DataTrajectory] = [DataTrajectory(**kwargs) for kwargs in kwargs_list]
        self.params: dict = params

    def compare_pcs(self, algorithms):
        for algorithm in algorithms:
            principal_components = []
            for trajectory in self.trajectories:
                res = trajectory.get_model_result(algorithm)
                principal_components.append(res['model'].eigenvectors)
            pcs = np.asarray(principal_components)
            MultiTrajectoryPlotter(interactive=False).plot_principal_components(algorithm, pcs,
                                                                                self.params[N_COMPONENTS])

    def _get_trajectories_by_index(self, traj_nrs: [list[int], None]):
        if traj_nrs is None:  # take all the trajectories
            return self.trajectories
        else:
            sorted_traj_nrs = sorted(i for i in traj_nrs if i < len(self.trajectories))
            return list(itemgetter(*sorted_traj_nrs)(self.trajectories))

    @staticmethod
    def _get_trajectory_combos(trajectories, model_params):
        traj_results = []
        for trajectory in trajectories:
            res = trajectory.get_model_result(model_params)
            res.update({'traj': trajectory})
            traj_results.append(res)
        return list(combinations(traj_results, 2))

    @staticmethod
    def _get_all_similarities_from_eigenvector_combos(combos, pc_nr_list=None, plot=False):
        if plot and pc_nr_list is None:
            raise ValueError('Trajectories can not be compared to each other, because the `pc_nr_list` is not given.')

        all_similarities = []
        for combi in combos:
            pc_0_matrix = combi[0][MODEL].eigenvectors.T
            pc_1_matrix = combi[1][MODEL].eigenvectors.T
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
                    title_prefix=f'{combi[0]["model"]}\n'
                                 f'{combi[0]["traj"].filename} & {combi[1]["traj"].filename}\n'
                                 f'PC Similarity',
                    x_label='Principal Component Number',
                    y_label='Principal Component Number',
                    bottom_text=sim_text
                ).matrix_plot(cos_matrix)
            all_similarities.append(combo_sim_of_n_pcs)
        return np.asarray(all_similarities)

    def compare_all_trajectory_eigenvectors(self,
                                            traj_nrs: [list[int], None],
                                            model_params_list: list[dict],
                                            pc_nr_list: [list[int], None]):
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
        :return:
        """
        trajectories = self._get_trajectories_by_index(traj_nrs)

        for model_params in model_params_list:
            eigenvector_combos = self._get_trajectory_combos(trajectories, model_params)
            all_sim_matrix = self._get_all_similarities_from_eigenvector_combos(eigenvector_combos)

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

    def compare_trajectory_combos(self, traj_nrs, model_params_list, pc_nr_list):
        trajectories = self._get_trajectories_by_index(traj_nrs)

        for model_params in model_params_list:
            result_combos = self._get_trajectory_combos(trajectories, model_params)
            self._get_all_similarities_from_eigenvector_combos(result_combos, pc_nr_list, plot=True)

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
        _save_tuning_results(grid.cv_results_, self.params[TRAJECTORY_NAME],
                             header=['params', 'mean_test_score', 'std_test_score', 'rank_test_score'])

    def reconstruct_with_different_eigenvector(self, model_params_list):
        """
        Calculate the reconstruction error of the trajectories
        given a list of eigenvalues of different models on a specific trajectory
        (0th-element in self.trajectories).
        :param model_params_list:
        :return:
        """

        st = SingleTrajectory(self.trajectories[0])
        model_results: list = st.compare(model_params_list, plot_results=False)

        model_scores = {}
        for model_dict in model_results:
            model: [ParameterModel, StreamingEstimationTransformer] = model_dict[MODEL]
            print(f'Calculating reconstruction errors ({model})...')
            score_list = []
            for trajectory in self.trajectories:
                input_data = trajectory.data_input(model_dict[INPUT_PARAMS])
                score = self._get_reconstruction_score(model, input_data)
                score_list.append(score)
            score_list = np.asarray(score_list)
            model_scores[f'{str(model):25}'] = score_list
        ArrayPlotter(
            interactive=False,
            title_prefix=f'Reconstruction Error from {self.trajectories[0].filename}\n',
            x_label='trajectories',
            y_label='score'
        ).plot_merged_2ds(model_scores, np.median)

    @staticmethod
    def _get_reconstruction_score(model, input_data):
        try:
            return model.score(input_data)
        except AttributeError:  # Only for Original Algorithms
            mu = np.mean(input_data, axis=0)
            data_projection = model.transform(input_data - mu)
            reconstructed_data = np.dot(data_projection, model.eigenvectors[:, :model.dim].T) + mu
            return mean_squared_error(input_data, reconstructed_data, squared=False)


def _save_tuning_results(result, name, header=None):
    if header is None:
        result_df = pd.DataFrame(result)
    else:
        result_df = pd.DataFrame(result)[header]
    current_result_path = Path('analyse_results') / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.mkdir(current_result_path)
    goal_path = current_result_path / (name + '.csv')
    result_df.to_csv(goal_path)
    print(f'Grid search results successfully saved into: {goal_path}')
