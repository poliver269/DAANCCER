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
from tqdm import tqdm

from plotter import ArrayPlotter, MultiTrajectoryPlotter, TrajectoryPlotter
from trajectory import DataTrajectory
from utils import statistical_zero
from utils.algorithms.tensor_dim_reductions import ParameterModel
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
        inp = self.trajectory.data_input()  # Cannot train for
        cv = [(slice(None), slice(None))]  # get rid of cross validation
        grid = GridSearchCV(model, param_grid, cv=cv, verbose=1)
        grid.fit(inp, n_components=self.trajectory.params[N_COMPONENTS])
        AnalyseResultsSaver(trajectory_name=self.trajectory.params[TRAJECTORY_NAME],
                            filename=f'grid_search_{self.trajectory.filename[:-4]}').save_to_csv(grid.cv_results_)


class MultiTrajectoryAnalyser:
    def __init__(self, kwargs_list, params):
        self.trajectories: list[DataTrajectory] = [DataTrajectory(**kwargs) for kwargs in kwargs_list]
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

    def compare_reconstruction_scores(self, model_params_list, from_other_traj=True):
        """
        Calculate the reconstruction error of the trajectories
        if from_other_traj is True than reconstruct from the model fitted on specific trajectory
        (0th-element in self trajectories)
        :param from_other_traj:
        :param model_params_list:
        """

        st = SingleTrajectoryAnalyser(self.trajectories[0])
        model_results: list = st.compare(model_params_list, plot_results=False)

        model_scores = {}
        for model_index, model_dict in enumerate(model_results):
            model: [ParameterModel, StreamingEstimationTransformer] = model_dict[MODEL]
            print(f'Calculating reconstruction errors ({model})...')
            score_list = []
            for trajectory in self.trajectories:
                if not from_other_traj:
                    model_dict = trajectory.get_model_result(model_params_list[model_index])
                    model = model_dict[MODEL]
                input_data = trajectory.data_input(model_parameters=model_dict[INPUT_PARAMS])
                score = self._get_reconstruction_score(model, input_data)
                score_list.append(score)
            score_list = np.asarray(score_list)
            model_scores[f'{str(model):25}'] = score_list

        ArrayPlotter(
            interactive=False,
            title_prefix=(f'Reconstruction Error from {self.trajectories[0].filename}\n' if from_other_traj
                          else 'Reconstruction Error (RE) '),
            x_label='trajectories',
            y_label='score',
            y_range=(0, 1)
        ).plot_merged_2ds(model_scores, np.median)

    @staticmethod
    def _get_reconstruction_score(model, input_data):
        if isinstance(model, ParameterModel):
            return model.score(input_data)
        else:  # Only for Original Algorithms
            data_projection = model.transform(input_data)  # pca subtracts mean, at transformation step.
            reconstructed_data = reconstruct_matrix(data_projection, model.eigenvectors, model.dim,
                                                    mean=np.mean(input_data, axis=0))
            return mean_squared_error(input_data, reconstructed_data, squared=False)

    def calculate_median_scores(self, model_params_list: list[dict], from_other_traj: bool = True):
        st = SingleTrajectoryAnalyser(self.trajectories[0])
        model_results: list = st.compare(model_params_list, plot_results=False)

        saver = AnalyseResultsSaver(trajectory_name=self.params[TRAJECTORY_NAME])
        model_median_scores = {}
        component_wise_scores = {}
        for model_index, model_dict in enumerate(model_results):
            model: [ParameterModel, StreamingEstimationTransformer] = model_dict[MODEL]
            print(f'Calculating median reconstruction errors ({model.describe()})...')
            model_dict_list = [model_dict]
            if not from_other_traj:
                for trajectory in self.trajectories[1:]:
                    model_dict_list.append(trajectory.get_model_result(model_params_list[model_index], log=False))

            median_list = []
            for component in tqdm(range(1, self.params[N_COMPONENTS] + 1)):
                score_list = []
                component_wise_scores[str(component)] = {}
                try:
                    for traj_index, trajectory in enumerate(self.trajectories):
                        if not from_other_traj:
                            model_dict = model_dict_list[traj_index]
                            model = model_dict[MODEL]
                            input_data = trajectory.data_input(model_dict[INPUT_PARAMS])
                            matrix_projection = model_dict[PROJECTION][0]
                        else:
                            input_data = trajectory.data_input(model_dict[INPUT_PARAMS])
                            matrix_projection = model.transform(input_data)

                        if isinstance(model, ParameterModel):
                            reconstructed_tensor = model.reconstruct(matrix_projection[:, :component], component)
                            input_data = model.convert_to_matrix(input_data)
                            reconstructed_data = model.convert_to_matrix(reconstructed_tensor)
                        else:
                            reconstructed_data = reconstruct_matrix(matrix_projection, model.eigenvectors, component,
                                                                    mean=np.mean(input_data, axis=0))
                        score = mean_squared_error(input_data, reconstructed_data, squared=False)
                        score_list.append(score)
                except IndexError as e:
                    warnings.warn(str(e))
                    break
                component_wise_scores[str(component)][f'{str(model.describe()):35}'] = score_list
                median_list.append(np.median(score_list))
            saver.save_to_npz(component_wise_scores, 'component_wise_RE_on_' +
                              ('other' if from_other_traj else 'same') + '_traj')
            model_median_scores[f'{str(model.describe()):35}'] = np.asarray(median_list)
        saver.save_to_npz(model_median_scores, 'median_RE_over_trajectories_on_' +
                                               'other' if from_other_traj else 'same')
        return model_median_scores

    def compare_median_reconstruction_scores(self, model_params_list: list[dict], from_other_traj: bool = True,
                                             load_filename: [str, None] = None):
        if load_filename is None:
            model_median_scores = self.calculate_median_scores(model_params_list, from_other_traj)
        else:
            model_median_scores = AnalyseResultLoader().load_npz(load_filename)

        ArrayPlotter(
            interactive=False,
            title_prefix=f'Reconstruction Error (RE) ' +
                         (f'from {self.trajectories[0].filename}\n' if from_other_traj else '') +
                         f'on {self.params[N_COMPONENTS]} Principal Components ',
            x_label='number of principal components',
            y_label='median REs of the trajectories',
            y_range=(0, 1)
        ).plot_merged_2ds(model_median_scores)

    def calculate_kernel_accuracies(self, kernel_names, model_params):
        kernel_accuracies = {kernel_name: [] for kernel_name in kernel_names}
        for trajectory in self.trajectories:
            model, _ = trajectory.get_model_and_projection(model_params)
            for kernel_name in kernel_names:
                matrix = model.get_combined_covariance_matrix()
                variance = calculate_symmetrical_kernel_matrix(
                    matrix, statistical_zero, kernel_name,
                    analyse_mode=KERNEL_COMPARE)
                kernel_accuracies[kernel_name].append(variance)
        return kernel_accuracies

    def compare_kernel_fitting_scores(self, kernel_names, model_params, load_filename: [str, None] = None):
        if load_filename is None:
            kernel_accuracies = self.calculate_kernel_accuracies(kernel_names, model_params)
        else:
            kernel_accuracies = AnalyseResultLoader().load_npz(load_filename)
        ArrayPlotter(
            interactive=False,
            title_prefix=f'Compare Kernels',
            x_label='trajectory Nr',
            y_range=(0, 0.2),
        ).plot_merged_2ds(kernel_accuracies, statistical_func=np.mean)


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
    @staticmethod
    def load_npy(filename: str) -> np.ndarray:
        return np.load(filename)

    @staticmethod
    def load_npz(filename: str) -> dict:
        return np.load(filename)
