import warnings
from datetime import datetime
from itertools import combinations
from operator import itemgetter

import numpy as np
import scipy.optimize
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from research_evaluations.plotter import ArrayPlotter, MultiTrajectoryPlotter, ModelResultPlotter
from preprocessing.config import get_data_class
from research_evaluations.file_operations import AnalyseResultsSaver, AnalyseResultLoader
from trajectory import ProteinTrajectory, DataTrajectory, SubTrajectoryDecorator
from utils import statistical_zero, get_algorithm_name
from utils.algorithms.dropp import DROPP
from utils.errors import InvalidReconstructionException, InvalidProteinTrajectory
from utils.matrix_tools import calculate_symmetrical_kernel_matrix, reconstruct_matrix
from utils.param_keys import *
from utils.param_keys.analyses import COLOR_MAP, KERNEL_COMPARE
from utils.param_keys.model import KERNEL_FUNCTION, USE_ORIGINAL_DATA, ALGORITHM_NAME
from utils.param_keys.model_result import MODEL, PROJECTION, INPUT_PARAMS, TITLE_PREFIX, FITTED_ON
from utils.param_keys.traj_dims import TIME_FRAMES


class SingleTrajectoryAnalyser:
    def __init__(self, trajectory, params=None):
        self.trajectory: DataTrajectory = trajectory
        self.params: dict = {
            PLOT_TYPE: params.get(PLOT_TYPE, COLOR_MAP),
            PLOT_TICS: params.get(PLOT_TICS, True),
            INTERACTIVE: params.get(INTERACTIVE, True),
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            PLOT_FOR_PAPER: params.get(PLOT_FOR_PAPER, False),
            ENABLE_SAVE: params.get(ENABLE_SAVE, False)
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

    def compare_with_plot(self, model_results_list):
        """
        This method is used to compare the results in a plot, after fit_transforming different models.
        @param model_results_list: list[dict]
            Different model input parameters, saved in a list.
        """
        ModelResultPlotter(
            interactive=self.params[INTERACTIVE],
            for_paper=self.params[PLOT_FOR_PAPER]
        ).plot_models(
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
                x_label='Num. Components',
                y_label='Eigenvalue',
                for_paper=self.params[PLOT_FOR_PAPER]
            ).plot_2d(ndarray_data=model.explained_variance_)

    def compare_trajectory_subsets(self, model_params_list):
        if isinstance(self.trajectory, SubTrajectoryDecorator):
            results = []
            for model_params in model_params_list:
                total_result = [self.trajectory.get_model_result(model_params)]
                total_result = total_result + self.trajectory.get_sub_results(model_params)
                results.append(total_result)
            ModelResultPlotter().plot_multi_projections(
                results, self.params[PLOT_TYPE], center_plot=False, sub_parts=True, show_model_properties=False)
        else:
            raise InvalidProteinTrajectory(f'Protein trajectory is invalid. Use a subset trajectory to ')

    def grid_search(self, param_grid: list[dict]):
        """
        Runs a grid search, to find the best input for the DAANCCER algorithm.
        @param param_grid: list[dict]
            List of different parameters, which sets the search space.
        """
        print('Searching for best model...')
        model = DROPP()
        inp = self.trajectory.data_input()  # Cannot train for different ndim at once
        cv = [(slice(None), slice(None))]  # get rid of cross validation
        grid = GridSearchCV(model, param_grid, cv=cv, verbose=1)
        grid.fit(inp, n_components=self.trajectory.params[N_COMPONENTS])
        AnalyseResultsSaver(
            trajectory_name=self.trajectory.params[TRAJECTORY_NAME],
            filename=f'grid_search_{self.trajectory.filename[:-4]}',
            enable_save=self.params[ENABLE_SAVE]
        ).save_to_csv(grid.cv_results_)


class SingleProteinTrajectoryAnalyser(SingleTrajectoryAnalyser):
    def __init__(self, trajectory, params=None):
        super().__init__(trajectory, params)
        self.trajectory: ProteinTrajectory = trajectory

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


class MultiTrajectoryAnalyser:
    def __init__(self, kwargs_list: list, params: dict, set_trajectories=True):
        if set_trajectories:
            self.trajectories: list[DataTrajectory] = [get_data_class(params, kwargs) for kwargs in kwargs_list]
        print(f'Trajectories loaded time: {datetime.now()}')
        self.params: dict = {
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            TRAJECTORY_NAME: params.get(TRAJECTORY_NAME, 'Not Found'),
            PLOT_TYPE: params.get(PLOT_TYPE, COLOR_MAP),
            PLOT_TICS: params.get(PLOT_TICS, True),
            INTERACTIVE: params.get(INTERACTIVE, True),
            PLOT_FOR_PAPER: params.get(PLOT_FOR_PAPER, False),
            TRANSFORM_ON_WHOLE: params.get(TRANSFORM_ON_WHOLE, False),
            ENABLE_SAVE: params.get(ENABLE_SAVE, False)
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
                interactive=self.params[INTERACTIVE],
                for_paper=self.params[PLOT_FOR_PAPER]
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
    def _get_trajectory_result_pairs(trajectories: list[ProteinTrajectory], model_params: dict) -> list:
        """
        Returns all the different combination-pairs of the results of the given trajectory list.
        The results of the models are calculated on the basis of the model parameters.
        @param trajectories: list[ProteinTrajectory]
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
                                                       plot: bool = False) -> np.ndarray:
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

            if pc_0_matrix.shape != pc_1_matrix.shape:
                continue  # TODO weather data PL and FI causes TICA errors
            # if isinstance(trajectory_pair[0][MODEL], DAANCCER):
            #     pc_0_matrix = pc_0_matrix.T[:self.params[N_COMPONENTS]]
            #     pc_1_matrix = pc_1_matrix.T[:self.params[N_COMPONENTS]]
            cos_matrix = cosine_similarity(np.real(pc_0_matrix), np.real(pc_1_matrix))
            # noinspection PyUnresolvedReferences
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
                    x_label='Num. Components',
                    y_label='Num. Components',
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
        trajectories = self._get_trajectories_by_index(traj_nrs)

        model_similarities = {}
        similarity_error_bands = {}
        for model_params in model_params_list:
            result_pairs = self._get_trajectory_result_pairs(trajectories, model_params)
            all_sim_matrix = self._get_all_similarities_from_trajectory_ev_pairs(result_pairs)

            if merged_plot:
                algorithm_name = get_algorithm_name(result_pairs[0][0][MODEL])
                model_similarities[algorithm_name] = np.mean(all_sim_matrix, axis=0)[1:]
                similarity_error_bands[algorithm_name] = np.vstack(
                    (np.min(all_sim_matrix, axis=0), np.max(all_sim_matrix, axis=0)))[:, 1:]
            else:
                if pc_nr_list is None:
                    ArrayPlotter(
                        interactive=self.params[INTERACTIVE],
                        title_prefix=f'{self.params[TRAJECTORY_NAME]}\n{model_params}\n'
                                     'Similarity value of all trajectories',
                        x_label='Num. Components',
                        y_label='Median Cosine Sim.',
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
                            # x_label='Trajectory number',
                            # y_label='Trajectory number',
                            # bottom_text=sim_text,
                            for_paper=self.params[PLOT_FOR_PAPER]
                        ).matrix_plot(tria)
        if merged_plot:
            saver = AnalyseResultsSaver(self.params[TRAJECTORY_NAME],
                                        enable_save=self.params[ENABLE_SAVE],
                                        folder_suffix='_EV-similarities')
            saver.save_to_npz(model_similarities, 'eigenvector_similarities')
            saver.save_to_npz(similarity_error_bands, 'error_bands_similarity')
            ArrayPlotter(
                interactive=self.params[INTERACTIVE],
                title_prefix=f'Eigenvector Similarities',
                x_label='Num. Components',
                y_label='Median Cosine Sim.',
                # y_range=(0.2, 1),
                for_paper=self.params[PLOT_FOR_PAPER]
            ).plot_merged_2ds(model_similarities, error_band=similarity_error_bands)

    def compare_trajectory_combos(self, traj_nrs, model_params_list, pc_nr_list):
        """
        Compare the trajectory combos with each other
        :param traj_nrs:
        :param model_params_list: list[dict]
            Different model input parameters, saved in a list.
        :param pc_nr_list: list
            Subset of principal components, which should be
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
        model = DROPP()
        cv = len(train_trajectories)
        grid = GridSearchCV(model, param_grid, cv=cv, verbose=1)
        grid.fit(inp, n_components=self.params[N_COMPONENTS])
        AnalyseResultsSaver(
            trajectory_name=self.params[TRAJECTORY_NAME],
            filename='grid_search_all',
            enable_save=self.params[ENABLE_SAVE]
        ).save_to_csv(grid.cv_results_, header=['params', 'mean_test_score', 'std_test_score', 'rank_test_score'])

    def compare_reconstruction_scores(self, model_params_list: list, fit_transform_re: bool = True):
        """
        Calculate the reconstruction error over the trajectory span for different model_params.
        If from_other_traj is True than reconstruct from the model fitted on specific trajectory set
        :param model_params_list: list[dict]
            Different model input parameters, saved in a list
        :param fit_transform_re: bool
            Fit-transform (Default: True) or Fit-on-all-Transform-on-one (False) reconstruction error
        """
        model_scores = {}
        for model_params in model_params_list:
            # model: [ParameterModel, StreamingEstimationTransformer] = model_dict[MODEL]
            print(f'Calculating reconstruction errors ({model_params})...')
            model_dict_list = self._get_model_result_list(model_params)
            model_description = get_algorithm_name(model_dict_list[DUMMY_ZERO][MODEL])
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
            score_ndarray = np.asarray(score_list)
            model_scores[model_description] = score_ndarray

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
        if isinstance(model, DROPP):
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
        model_median_scores, re_error_bands = self._calculate_median_scores(model_params_list, fit_transform_re)
        ArrayPlotter(
            interactive=self.params[INTERACTIVE],
            for_paper=self.params[PLOT_FOR_PAPER],
            title_prefix=f'Reconstruction Error (RE) ' +
                         ('' if fit_transform_re
                          else f'fit-on-one-transform-on-all\n') +
                         f'on {self.trajectories[DUMMY_ZERO].max_components} Principal Components ',
            x_label='Num. Components',
            y_label='Mean Squared Error',
            # y_range=(0, 1),
            xtick_start=1
        ).plot_merged_2ds(model_median_scores, error_band=re_error_bands)

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
        saver = AnalyseResultsSaver(
            trajectory_name=self.params[TRAJECTORY_NAME],
            enable_save=self.params[ENABLE_SAVE],
            folder_suffix='_' + ("Ftoa" if fit_transform_re else "FooToa")
        )

        model_median_scores = {}
        model_mean_scores = {}
        component_wise_scores = {}
        re_error_bands = {}
        for model_params in model_params_list:
            print(f'Calculating median reconstruction errors ({model_params})...')
            model_dict_list = self._get_model_result_list(model_params)
            model_description = get_algorithm_name(model_dict_list[DUMMY_ZERO][MODEL])

            scores_on_component_span = self._get_reconstruction_score_of_component_span(model_dict_list,
                                                                                        fit_transform_re)
            component_wise_scores[model_description] = scores_on_component_span
            model_median_scores[model_description] = np.median(np.array(scores_on_component_span), axis=1)
            model_mean_scores[model_description] = np.mean(np.array(scores_on_component_span), axis=1)
            re_error_bands[model_description] = np.vstack((np.min(scores_on_component_span, axis=1),
                                                           np.max(scores_on_component_span, axis=1)))
            saver.save_to_npz(model_median_scores, 'median_RE_' +
                              ('fit-transform' if fit_transform_re else 'FooToa'))
            saver.save_to_npz(model_mean_scores, 'mean_RE_' +
                              ('fit-transform' if fit_transform_re else 'FooToa'))
            saver.save_to_npz(component_wise_scores, 'component_wise_RE_on_' +
                              ('fit-transform' if fit_transform_re else 'FooToa'))
            saver.save_to_npz(re_error_bands, 'error_bands_' +
                              ('fit-transform' if fit_transform_re else 'FooToa'))
        return model_median_scores, re_error_bands

    def _get_model_result_list(self, model_params: dict):
        """
        Get the results of a model for all the trajectories.
        @param model_params: dict
            Parameters for the model.
        @return: results of models
        """
        model_dict_list = []
        for trajectory in self.trajectories:
            if isinstance(trajectory, SubTrajectoryDecorator) and trajectory.part_count is None:
                model_dict_list = model_dict_list + trajectory.get_sub_results(model_params)
            model_dict_list.append(trajectory.get_model_result(model_params, log=False))
        return model_dict_list

    def _get_reconstruction_score_of_component_span(self,
                                                    model_dict_list: list[dict],
                                                    fit_transform_re: bool = True) -> np.ndarray:
        """
        Calculates the reconstruction errors of the trajectories over the component span.
        @param model_dict_list: list
            model results dict in a list
        @param fit_transform_re: bool (default = True)
            Should be calculated the fit_transform on the same data
            or the transformation steps of trajectories should be applied
            to models fitted on a different trajectory.
        @return:
        """
        scores_on_component_span: list = []
        for component in tqdm(range(1, self.trajectories[DUMMY_ZERO].max_components + 1)):
            try:
                scores_on_component_span.append(
                    self._models_re_for_component(component, fit_transform_re, model_dict_list))
            except InvalidReconstructionException as e:
                warnings.warn(str(e))
                break
        return np.array(scores_on_component_span)

    def _models_re_for_component(self, component: int, fit_transform_re: bool, model_dict_list: list) -> list:
        all_models_reconstruction_scores: list = []
        for traj_index, fitted_trajectory in enumerate(self.trajectories):
            model_dict = model_dict_list[traj_index]
            model = model_dict[MODEL]
            if fit_transform_re:
                model_reconstruction_score = self._reconstruction_score_ftoa(component, fitted_trajectory, model,
                                                                             model_dict)
            else:  # fit on one transform on all
                model_reconstruction_score = self._reconstruction_score_footoa(component, model,
                                                                               model_dict[INPUT_PARAMS])
            all_models_reconstruction_scores.append(model_reconstruction_score)
        return all_models_reconstruction_scores

    def _reconstruction_score_ftoa(self, component: int, fitted_trajectory: DataTrajectory, model,
                                   model_result_dict: dict):
        """
        Calculate the reconstruction score of a model, while using the
        reconstruction of the same dataset after fit transforming it.
        :param component:
        :param fitted_trajectory:
        :param model:
            Model with parent class TransformerMixin, BaseEstimator of sklearn
        :param model_result_dict:
        :return:
        """
        if (self.params[TRANSFORM_ON_WHOLE] and
                isinstance(fitted_trajectory, SubTrajectoryDecorator)):
            with fitted_trajectory.use_full_input():
                input_data = fitted_trajectory.data_input(model_result_dict[INPUT_PARAMS])
        else:
            input_data = fitted_trajectory.data_input(model_result_dict[INPUT_PARAMS])
        matrix_ndim_projection = model_result_dict[PROJECTION]
        return self._get_reconstruction_score(model, input_data, matrix_ndim_projection, component)

    def _reconstruction_score_footoa(self, component: int, model, model_params: dict):
        """
        Calculate the reconstruction score of a model, while using the
        Fit on one Transform (FooToa) approach.
        In this approach the model is already fitted,
        and the trajectories of the dataset are transformed on this model.
        The calculation is for a specific number of component
        :param component: int
            number of principal components used
        :param model: model
            with parent class TransformerMixin, BaseEstimator of sklearn
        :param model_params: dict
            used to determine the data_input of the trajectories
        :return: float
            median of the reconstruction errors of the trajectories of a model
        """
        transform_score = []
        for transform_trajectory in self.trajectories:
            if (self.params[TRANSFORM_ON_WHOLE] and
                    isinstance(transform_trajectory, SubTrajectoryDecorator)):
                with transform_trajectory.use_full_input():
                    input_data = transform_trajectory.data_input(model_params)
            else:
                input_data = transform_trajectory.data_input(model_params)

            matrix_projection = model.transform(input_data)

            transform_score.append(
                self._get_reconstruction_score(model, input_data, matrix_projection, component))

        return np.median(transform_score)

    def compare_kernel_fitting_scores_on_same_model(self, kernel_names, model_params,
                                                    load_filename: [str, None] = None):
        if load_filename is None:
            kernel_accuracies = self._calculate_kernel_accuracies_same_model(kernel_names, model_params)
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

    def compare_kernel_fitting_scores(self, model_params_list):
        kernel_accuracies = self._calculate_kernel_accuracies(model_params_list)
        ArrayPlotter(
            interactive=self.params[INTERACTIVE],
            title_prefix=f'Compare Kernels ',
            x_label='Trajectory Number',
            y_label='RMSE of the fitting kernel',
            y_range=(0, 0.2),
            for_paper=self.params[PLOT_FOR_PAPER]
        ).plot_merged_2ds(kernel_accuracies, statistical_func=np.median)

    def _calculate_kernel_accuracies(self, model_params_list: list[dict]):
        kernel_accuracies = {}

        for model_params in model_params_list:

            kernel_description = f'{model_params[KERNEL_FUNCTION]}'
            if USE_ORIGINAL_DATA in model_params.keys():
                if model_params[USE_ORIGINAL_DATA]:
                    kernel_description += "use original data"
                else:
                    kernel_description += "with rescaled data"

            if kernel_description not in kernel_accuracies.keys():
                kernel_accuracies[kernel_description] = []

            for trajectory in self.trajectories:
                model, _ = trajectory.get_model_and_projection(model_params)
                matrix = model.get_combined_covariance_matrix()
                variance = calculate_symmetrical_kernel_matrix(
                    matrix, statistical_zero, model_params[KERNEL_FUNCTION],
                    analyse_mode=KERNEL_COMPARE)
                kernel_accuracies[kernel_description].append(variance)
        return kernel_accuracies

    def _calculate_kernel_accuracies_same_model(self, kernel_names, model_params):
        kernel_accuracies = {kernel_name: [] for kernel_name in kernel_names}
        for trajectory in self.trajectories:
            model, _ = trajectory.get_model_and_projection(model_params)
            for kernel_name in kernel_names:
                matrix = model.get_combined_covariance_matrix()
                variance = calculate_symmetrical_kernel_matrix(
                    matrix, statistical_zero, kernel_name,
                    analyse_mode=KERNEL_COMPARE)
                kernel_accuracies[kernel_name].append(variance)
        AnalyseResultsSaver(
            self.params[TRAJECTORY_NAME],
            filename='compare_rmse_kernel',
            enable_save=self.params[ENABLE_SAVE]
        ).save_to_npz(kernel_accuracies)
        return kernel_accuracies

    def compare_results_on_same_fitting(self, model_params, traj_index, plot=True):
        fitting_trajectory = self.trajectories[traj_index]
        fitting_results = fitting_trajectory.get_model_result(model_params)
        fitting_results[FITTED_ON] = fitting_trajectory.filename[:-4]
        model_results_list = []
        for trajectory_nr, trajectory in enumerate(self.trajectories):
            if trajectory_nr == traj_index:
                fitting_results[TITLE_PREFIX] = trajectory.filename[:-4]
                model_results_list.append(fitting_results)
            elif model_params[ALGORITHM_NAME] == "original_tsne":
                fitting_results[TITLE_PREFIX] = fitting_trajectory.filename[:-4]
                fitting_results[FITTED_ON] = trajectory.filename[:-4]
                model_results_list.append(fitting_results)
            else:
                transform_results = fitting_results.copy()
                transform_results[PROJECTION] = fitting_results[MODEL].transform(trajectory.data_input(model_params))
                transform_results[TITLE_PREFIX] = trajectory.filename[:-4]
                model_results_list.append(transform_results)

        if plot:
            SingleTrajectoryAnalyser(
                trajectory=fitting_trajectory,
                params=self.params
            ).compare_with_plot(model_results_list)

        saver = AnalyseResultsSaver(self.params[TRAJECTORY_NAME], enable_save=self.params[ENABLE_SAVE])
        for result_index, model_result in enumerate(model_results_list):
            saver.save_to_npz(
                model_result,
                new_filename=f'{self.trajectories[result_index].filename[:-4]}'
                             f'_transformed_on_{fitting_trajectory.filename[:-4]}'
            )

        return model_results_list

    def compare_projection_matrix(self, model_params_list):
        for model_params in model_params_list:  # create matrix for every model parameter
            list_of_list = []
            for traj_index in range(len(self.trajectories)):
                fitted_traj_row = self.compare_results_on_same_fitting(model_params, traj_index, plot=False)
                list_of_list.append(fitted_traj_row)

            if model_params[ALGORITHM_NAME] == 'original_tsne':
                list_of_list = list(zip(*list_of_list))  # flip rows and columns

            ModelResultPlotter(
                interactive=self.params[INTERACTIVE],
                for_paper=self.params[PLOT_FOR_PAPER]
            ).plot_multi_projections(
                model_result_list_in_list=list_of_list,
                plot_type=self.params[PLOT_TYPE],
                center_plot=False,
                sub_parts=False,
                show_model_properties=False
            )


class MultiSubTrajectoryAnalyser(MultiTrajectoryAnalyser):
    def __init__(self, kwargs_list: list, params: dict):
        super().__init__(kwargs_list, params, set_trajectories=False)
        self.trajectories: list[SubTrajectoryDecorator] = [get_data_class(params, kwargs) for kwargs in kwargs_list]

    def compare_re_on_small_parts(self, model_params_list):
        saver = AnalyseResultsSaver(
            trajectory_name=self.params[TRAJECTORY_NAME],
            enable_save=self.params[ENABLE_SAVE],
            folder_suffix='_FooToaTws'
        )
        max_time_steps = self.trajectories[DUMMY_ZERO].dim[TIME_FRAMES]  # e.g. 10000
        time_steps = np.geomspace(self.trajectories[DUMMY_ZERO].max_components, max_time_steps, num=15, dtype=int)
        print(f'Time window sizes: {time_steps}')
        component_list = np.asarray([2, 5, 10, 25, 50])
        saver.save_to_npz({'time_steps': time_steps}, 'time_steps_FooToaTws')
        saver.save_to_npz({'component_list': component_list}, 'component_list_FooToaTws')

        re_error_bands = {}
        model_median_scores = {}  # {'PCA': {'1': }, 'DROPP', 'TICA'}
        for model_params in model_params_list:
            print(f'Calculating reconstruction errors ({model_params})...')

            for time_index, time_window_size in enumerate(tqdm(time_steps)):
                self.change_time_window_sizes(time_window_size)
                model_dict_list = self._get_model_result_list(model_params)
                model_description = get_algorithm_name(model_dict_list[DUMMY_ZERO][MODEL])
                if model_description not in model_median_scores.keys():
                    model_median_scores[model_description] = np.zeros((time_steps.size, component_list.size))
                    re_error_bands[model_description] = np.zeros((time_steps.size, component_list.size, 2))

                for component_index, component in enumerate(component_list):
                    models_re_for_component: list = self._models_re_for_component(component, fit_transform_re=False,
                                                                                  model_dict_list=model_dict_list)
                    model_median_scores[model_description][time_index, component_index] = np.median(
                        models_re_for_component)
                    re_error_bands[model_description][time_index, component_index] = (
                        np.min(models_re_for_component),
                        np.max(models_re_for_component)
                    )
                    saver.save_to_npz(model_median_scores, 'median_RE_FooToaTws')
                    saver.save_to_npz(re_error_bands, 'error_bands_FooToaTws')

        ArrayPlotter(
            interactive=self.params[INTERACTIVE],
            title_prefix=f'FooToa on varying time window size',
            x_label='Time window size',
            y_label='Mean Squared Error',
            for_paper=self.params[PLOT_FOR_PAPER],
            y_range=(0, 2)
        ).plot_matrix_in_2d(model_median_scores, time_steps, component_list, re_error_bands)

    def change_time_window_sizes(self, new_time_window_size):
        self.trajectories = [trajectory.change_time_window_size(new_time_window_size) for trajectory in
                             self.trajectories]
