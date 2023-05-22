import os
import warnings
from datetime import datetime
from itertools import combinations
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from plotter import ArrayPlotter, MultiTrajectoryPlotter, ModelResultPlotter
from trajectory import DataTrajectory, TrajectorySubset
from utils import statistical_zero, pretify_dict_model
from utils.algorithms.tensor_dim_reductions.daanccer import DAANCCER
from utils.errors import InvalidReconstructionException, InvalidDataTrajectory
from utils.matrix_tools import calculate_symmetrical_kernel_matrix, reconstruct_matrix
from utils.param_keys import *
from utils.param_keys.analyses import COLOR_MAP, KERNEL_COMPARE
from utils.param_keys.model_result import MODEL, PROJECTION, INPUT_PARAMS


class SingleTrajectoryAnalyser:
    def __init__(self, trajectory, params=None):
        self.trajectory: DataTrajectory = trajectory
        self.params: dict = {
            PLOT_TYPE: params.get(PLOT_TYPE, COLOR_MAP),
            PLOT_TICS: params.get(PLOT_TICS, True),
            INTERACTIVE: params.get(INTERACTIVE, True),
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            PLOT_FOR_PAPER: params.get(PLOT_FOR_PAPER, False)
        }

    def compare(self, model_parameter_list: list[dict], plot_results: bool = True) -> list[dict]:
        """
        Calculates the result of different models (fitted and transformed the data of the trajectory)
        given by different parameters and returns the result list
        :param model_parameter_list: list[dict]
            Different model input parameters, saved in a list
        :param plot_results: bool
            Plots the components of the different models (default: True)
            The function can be calculated
        :return: list[dict]
            The results of the models {MODEL, PROJECTION, EXPLAINED_VAR, INPUT_PARAMS}
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

    def compare_with_carbon_alpha_and_all_atoms(self, model_params_list):
        """
        Compares the same trajectory with using only carbon atoms
        and using all the atoms of different model_params.
        @param model_params_list: list[dict]
            Different model input parameters, saved in a list.
        """
        model_results = self.trajectory.get_model_results_with_changing_trajectory_parameter(model_params_list,
                                                                                             CARBON_ATOMS_ONLY)
        self.compare_with_plot(model_results)

    def compare_with_basis_transformation(self, model_params_list):
        """
        Compares same trajectory with using and not using basis transformation of different model_params.
        @param model_params_list: list[dict]
            Different model input parameters, saved in a list.
        """
        model_results = self.trajectory.get_model_results_with_changing_trajectory_parameter(model_params_list,
                                                                                             BASIS_TRANSFORMATION)
        self.compare_with_plot(model_results)

    def compare_with_plot(self, model_results_list):
        """
        This method is used to compare the results in a plot, after fit_transforming different models.
        @param model_results_list: list[dict]
            Different model input parameters, saved in a list.
        """
        ModelResultPlotter().plot_models(
            model_results_list,
            plot_type=self.params[PLOT_TYPE],
            plot_tics=self.params[PLOT_TICS],
            components=self.params[N_COMPONENTS]
        )

    def plot_eigenvalues(self, model_parameter_list):
        """
        Plots the values of the eigenvalues.
        @param model_parameter_list: list[dict]
            Different model input parameters, saved in a list.
        """
        model_results_list = self.compare(model_parameter_list, plot_results=False)
        for model_result in model_results_list:
            model = model_result[MODEL]
            ArrayPlotter(
                title_prefix=f'Eigenvalues of\n{model}',
                x_label='Principal Component Number',
                y_label='Eigenvalue',
                for_paper=self.params[PLOT_FOR_PAPER]
            ).plot_2d(ndarray_data=model.explained_variance_)

    def compare_trajectory_subsets(self, model_params_list):
        if isinstance(self.trajectory, TrajectorySubset):
            results = []
            for model_params in model_params_list:
                total_result = [self.trajectory.get_model_result(model_params)]
                total_result = total_result + self.trajectory.get_sub_results(model_params)
                results.append(total_result)
            ModelResultPlotter().plot_multi_projections(results, self.params[PLOT_TYPE], center_plot=False)
        else:
            raise InvalidDataTrajectory(f'Data trajectory is invalid. Use a subset trajectory to ')

    def grid_search(self, param_grid: list[dict]):
        """
        Runs a grid search, to find the best input for the DAANCCER algorithm.
        @param param_grid: list[dict]
            List of different parameters, which sets the search space.
        """
        print('Searching for best model...')
        model = DAANCCER()
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
        self.params: dict = {
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            TRAJECTORY_NAME: params.get(TRAJECTORY_NAME, 'Not Found'),
            PLOT_TYPE: params.get(PLOT_TYPE, COLOR_MAP),
            PLOT_TICS: params.get(PLOT_TICS, True),
            INTERACTIVE: params.get(INTERACTIVE, True),
            PLOT_FOR_PAPER: params.get(PLOT_FOR_PAPER, False)
        }

    def compare_pcs(self, model_params_list: list[dict]):
        """
        Method is used to analyse principal components (=eigenvectors).
        It saves the eigenvectors of the different trajectories into an array and plots it for different models.
        @param model_params_list: list[dict]
            List of different parameters, which sets the search space.
        """
        for model_parameters in model_params_list:
            principal_components = []
            for trajectory in self.trajectories:
                res = trajectory.get_model_result(model_parameters)
                principal_components.append(res['model'].components_)
            pcs = np.asarray(principal_components)
            MultiTrajectoryPlotter(
                interactive=self.params[INTERACTIVE]
            ).plot_principal_components(model_parameters, pcs, self.params[N_COMPONENTS])

    def _get_trajectories_by_index(self, traj_nrs: [list[int], None]):
        """
        Return a subset of trajectories given their index number.
        @param traj_nrs: list[int] or None
            if None all the trajectories are returned
            else: compare only the trajectories in a given list,
        @return:
        """
        if traj_nrs is None:  # take all the trajectories
            return self.trajectories
        else:
            sorted_traj_nrs = sorted(i for i in traj_nrs if i < len(self.trajectories))
            return list(itemgetter(*sorted_traj_nrs)(self.trajectories))

    @staticmethod
    def _get_trajectory_result_pairs(trajectories: list[DataTrajectory], model_params: dict) -> list:
        """
        Returns all the different combination-pairs of the results of the given trajectory list.
        The results of the models are calculated on the basis of the model parameters.
        @param trajectories: list[DataTrajectory]
            The subset of trajectories to use the models for this step.
        @param model_params: dict
            The model parameters for the models.
        @return:
        """
        traj_results = []
        for trajectory in trajectories:
            res = trajectory.get_model_result(model_params)
            res.update({'traj': trajectory})
            traj_results.append(res)
        return list(combinations(traj_results, 2))

    def _get_all_similarities_from_trajectory_ev_pairs(self, trajectory_result_pairs: list[tuple],
                                                       pc_nr_list: list = None,
                                                       plot: bool = False):
        """
        Calculates and returns the similarity of eigenvectors between the results between trajectory pairs.
        The similarity is calculated via cosine similarity and then compared the most similar ones
        to calculate the mean of all together
        :param trajectory_result_pairs: list[tuple]
            list of pairwise trajectory results after a model was fitted and transformed on the trajectory
        :param plot: bool
            Sets if the similarity values should be plotted or returned and used for further analysis. (default: False)
        :param pc_nr_list: list (default None)
            subset of the indexes of the principal components.
            If plot == False, this parameter is ignored.
        :return:
        """
        if plot and pc_nr_list is None:
            raise ValueError('Trajectories can not be compared to each other, because the `pc_nr_list` is not given.')

        all_similarities = []
        for trajectory_pair in trajectory_result_pairs:
            pc_0_matrix = trajectory_pair[0][MODEL].components_
            pc_1_matrix = trajectory_pair[1][MODEL].components_
            # if isinstance(trajectory_pair[0][MODEL], DAANCCER):
            #     pc_0_matrix = pc_0_matrix.T[:self.params[N_COMPONENTS]]
            #     pc_1_matrix = pc_1_matrix.T[:self.params[N_COMPONENTS]]
            cos_matrix = cosine_similarity(np.real(pc_0_matrix), np.real(pc_1_matrix))
            sorted_similarity_indexes = scipy.optimize.linear_sum_assignment(-np.abs(cos_matrix))[DUMMY_ONE]
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
                    interactive=self.params[INTERACTIVE],
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
        which are fitted and transformed from different trajectories.
        :param traj_nrs:
            if None compare all the trajectories
            compare only the trajectories in a given list,
        :param model_params_list: list[dict]
            Different model input parameters, saved in a list, which should be compared with each other
        :param pc_nr_list:
            gives a list of how many principal components should be compared with each other.
            If None, then all the principal components are compared
        :param merged_plot: bool
            The parameter makes the
        :return:
        """
        # TODO: Implement the Save Similarities and Load them
        trajectories = self._get_trajectories_by_index(traj_nrs)

        model_similarities = {}
        similarity_error_bands = {}
        for model_params in model_params_list:
            result_pairs = self._get_trajectory_result_pairs(trajectories, model_params)
            all_sim_matrix = self._get_all_similarities_from_trajectory_ev_pairs(result_pairs)

            if merged_plot:
                model_similarities[str(result_pairs[0][0][MODEL])] = np.mean(all_sim_matrix, axis=0)
                similarity_error_bands[str(result_pairs[0][0][MODEL])] = np.vstack((np.min(all_sim_matrix, axis=0),
                                                                                    np.max(all_sim_matrix, axis=0)))
            else:
                if pc_nr_list is None:
                    ArrayPlotter(
                        interactive=self.params[INTERACTIVE],
                        title_prefix=f'{self.params[TRAJECTORY_NAME]}\n{model_params}\n'
                                     'Similarity value of all trajectories',
                        x_label='Principal component number',
                        y_label='Similarity value',
                        for_paper=self.params[PLOT_FOR_PAPER]
                    ).plot_2d(np.mean(all_sim_matrix, axis=0))
                else:
                    for pc_index in pc_nr_list:
                        tria = np.zeros((len(trajectories), len(trajectories)))
                        sim_text = f'Similarity of all {np.mean(all_sim_matrix[:, pc_index])}'
                        print(sim_text)
                        tria[np.triu_indices(len(trajectories), 1)] = all_sim_matrix[:, pc_index]
                        tria = tria + tria.T
                        ArrayPlotter(
                            interactive=self.params[INTERACTIVE],
                            title_prefix=f'{self.params[TRAJECTORY_NAME]}\n{model_params}\n'
                                         f'Trajectory Similarities for {pc_index}-Components',
                            x_label='Trajectory number',
                            y_label='Trajectory number',
                            bottom_text=sim_text,
                            for_paper=self.params[PLOT_FOR_PAPER]
                        ).matrix_plot(tria)
        if merged_plot:
            AnalyseResultsSaver(self.params[TRAJECTORY_NAME]).save_to_npz(
                model_similarities, 'eigenvector_similarities')
            ArrayPlotter(
                interactive=self.params[INTERACTIVE],
                title_prefix=f'Eigenvector Similarities',
                x_label='Principal Component Number',
                y_label='Similarity value',
                y_range=(0.2, 1),
                for_paper=self.params[PLOT_FOR_PAPER]
            ).plot_merged_2ds(model_similarities, error_band=similarity_error_bands)

    def compare_trajectory_combos(self, traj_nrs, model_params_list, pc_nr_list):
        """
        Compare the trajectory combos with each other
        :param traj_nrs:
        :param model_params_list: list[dict]
            Different model input parameters, saved in a list.
        :param pc_nr_list: TODO
        :return:
        """
        trajectories = self._get_trajectories_by_index(traj_nrs)

        for model_params in model_params_list:
            trajectory_pairs = self._get_trajectory_result_pairs(trajectories, model_params)
            self._get_all_similarities_from_trajectory_ev_pairs(trajectory_pairs, pc_nr_list, plot=True)

    def grid_search(self, param_grid):
        """
        Runs a grid search for multiple trajectories, to find the best input for the DAANCCER algorithm.
        @param param_grid: list[dict]
            List of different parameters, which sets the search space.
        """
        print('Searching for best model...')
        train_trajectories = self.trajectories[:int(len(self.trajectories) * .8)]
        # test_trajectories = self.trajectories[int(len(self.trajectories) * .8):]
        inp = train_trajectories[0].data_input()  # Works only for TENSOR_DIM_INPUT
        for trajectory in train_trajectories[1:]:
            np.concatenate((inp, trajectory.data_input()), axis=0)
        model = DAANCCER()
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
        :param fit_transform_re: Fit-transform reconstruction error or Fit-on-all-Transform-on-one
        :param model_params_list: list[dict]
            Different model input parameters, saved in a list.
        """

        model_scores = {}
        for model_index, model_params in enumerate(model_params_list):
            # model: [ParameterModel, StreamingEstimationTransformer] = model_dict[MODEL]
            print(f'Calculating reconstruction errors ({model_params})...')
            model_dict_list = self._get_model_result_list(model_params)
            model_description = str(model_dict_list[DUMMY_ZERO][MODEL]).split('(')[DUMMY_ZERO]
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
            model_scores[model_description] = score_list

        # model_scores = pretify_dict_model(model_scores)
        ArrayPlotter(
            interactive=self.params[INTERACTIVE],
            title_prefix='Reconstruction Error (RE) ' + (
                '' if fit_transform_re else f'fit-on-one-transform-on-all\n'),
            x_label='trajectories',
            y_label='score',
            # y_range=(0, 1),
            for_paper=self.params[PLOT_FOR_PAPER]
        ).plot_merged_2ds(model_scores, np.median)

    @staticmethod
    def _get_reconstruction_score(model, input_data: np.ndarray, data_projection: [np.ndarray, None] = None,
                                  component: int = None):
        """
        Calculates the reconstruction score of the model and the input data.
        @param model: obj
            Model with parent class TransformerMixin, BaseEstimator of sklearn
        @param input_data: np.ndarray
            original data before transforming
        @param data_projection: np.ndarray (optional)
            the transformed data. If this is not calculated, then the transformation step is used to calculate it.
        @param component: int (default: None)
            This parameter is used to set how many components of the model should be used
            to reconstruct the original data. If it's None, all the components of the model are used.
        @return: Root Mean Squared Error of the reconstruction data and the original one
        """
        if isinstance(model, DAANCCER):
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
                component = model.n_components
            reconstructed_data = reconstruct_matrix(data_projection, model.components_, component,
                                                    mean=model.mean_)
        return mean_squared_error(input_data, reconstructed_data, squared=False)

    def compare_median_reconstruction_scores(self, model_params_list: list[dict], fit_transform_re: bool = True):
        """
        Used for analysing the reconstruction scores of different models.
        (See self._calculate_median_scores() for more information)
        @param model_params_list: list[dict]
            Different model input parameters, saved in a list.
        @param fit_transform_re: bool (default = True)
            Should be calculated the fit_transform on the same data
            or the transformation steps of trajectories should be applied
            to models fitted on a different trajectory.
        @return:
        """
        model_median_scores = self._calculate_median_scores(model_params_list, fit_transform_re)
        ArrayPlotter(
            interactive=self.params[INTERACTIVE],
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
        """
        Calculates the median reconstruction scores for different models.
        @param model_params_list: list[dict]
            Different model input parameters, saved in a list.
        @param fit_transform_re: bool (default = True)
            Should be calculated the fit_transform on the same data
            or the transformation steps of trajectories should be applied
            to models fitted on a different trajectory.
        @return:
        """
        saver = AnalyseResultsSaver(trajectory_name=self.params[TRAJECTORY_NAME])

        model_median_scores = {}
        component_wise_scores = {}
        for model_params in model_params_list:
            print(f'Calculating median reconstruction errors ({model_params})...')
            model_dict_list = self._get_model_result_list(model_params)
            model_description = str(model_dict_list[DUMMY_ZERO][MODEL]).split('(')[DUMMY_ZERO]

            median_list = self._get_median_over_components_for_trajectories(model_dict_list,
                                                                            component_wise_scores,
                                                                            model_description,
                                                                            fit_transform_re)
            model_median_scores[model_description] = np.asarray(median_list)
            saver.save_to_npz(model_median_scores, 'median_RE_over_trajectories_on_' +
                              ('fit-transform' if fit_transform_re else 'FooToa'))
            saver.save_to_npz(component_wise_scores, 'component_wise_RE_on_' +
                              ('fit-transform' if fit_transform_re else 'FooToa') + '_traj')
        return model_median_scores

    def _get_model_result_list(self, model_params: dict):
        """
        Get the results of a model for all the trajectories.
        @param model_params: dict
            Parameters for the model.
        @return: results of models
        """
        model_dict_list = []
        for trajectory in self.trajectories:
            model_dict_list.append(trajectory.get_model_result(model_params, log=False))
        return model_dict_list

    def _get_median_over_components_for_trajectories(self,
                                                     model_dict_list: list[dict],
                                                     component_wise_scores: dict,
                                                     model_description: str,
                                                     fit_transform_re: bool = True) -> list:
        """
        Calculates the median reconstruction errors of the trajectories over the component span.
        @param model_dict_list:
        @param component_wise_scores: dict
            This dictionary is updated without returning it.
        @param model_description:
        @param fit_transform_re:
        @return:
        """
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
                    else:  # fit on one transform on all
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
            component_wise_scores[str(component)][model_description] = score_list
            median_list.append(np.median(score_list))
        return median_list

    def compare_kernel_fitting_scores(self, kernel_names, model_params, load_filename: [str, None] = None):
        if load_filename is None:
            kernel_accuracies = self._calculate_kernel_accuracies(kernel_names, model_params)
        else:
            kernel_accuracies = AnalyseResultLoader(self.params[TRAJECTORY_NAME]).load_npz(load_filename)
        ArrayPlotter(
            interactive=self.params[INTERACTIVE],
            title_prefix=f'Compare Kernels ',
            x_label='Trajectory Number',
            y_label='RMSE of the fitting kernel',
            y_range=(0, 0.2),
            for_paper=self.params[PLOT_FOR_PAPER]
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
                new_filename=f'{self.trajectories[result_index].filename[:-4]}'
                             f'_transformed_on_{fitting_trajectory.filename[:-4]}'
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

    def load_npz_by_filelist(self, filename_list: list) -> dict:
        merged_dict = {}
        for filename in filename_list:
            merged_dict.update(self.load_npz(filename))
        return merged_dict

    def load_npz_files_in_directory(self, directory_name):
        directory_path = self.get_load_path(directory_name)
        filename_list = os.listdir(directory_path)
        return self.load_npz_list(directory_name, filename_list)

    def load_npz_list(self, root_dir, filename_list):
        loaded_list = []
        for filename in filename_list:
            loaded_list.append(self.load_npz(Path(root_dir) / filename))
        return loaded_list
