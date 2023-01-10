import os
import warnings
from datetime import datetime
from itertools import combinations
from operator import itemgetter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV

from plotter import ArrayPlotter, MultiTrajectoryPlotter, TrajectoryPlotter
from trajectory import DataTrajectory
from utils.algorithms.tensor_dim_reductions import ParameterModel
from utils.param_key import TRAJECTORY_NAME, N_COMPONENTS, CARBON_ATOMS_ONLY, BASIS_TRANSFORMATION, PLOT_TICS, PLOT_TYPE


class SingleTrajectory:
    def __init__(self, trajectory):
        self.trajectory: DataTrajectory = trajectory

    def compare(self, model_parameter_list: list[dict, str]):
        model_results = []
        for model_parameters in model_parameter_list:
            try:
                model_results.append(self.trajectory.get_model_result(model_parameters))
            except np.linalg.LinAlgError as e:
                warnings.warn(f'Eigenvalue decomposition for model `{model_parameters}` could not be calculated:\n {e}')
            except AssertionError as e:
                warnings.warn(f'{e}')
        self.compare_with_plot(model_results)

    def compare_with_carbon_alpha_and_all_atoms(self, model_names):
        model_results = self.trajectory.get_model_results_with_different_param(model_names, CARBON_ATOMS_ONLY)
        self.compare_with_plot(model_results)

    def compare_with_basis_transformation(self, model_params_list):
        model_results = self.trajectory.get_model_results_with_different_param(model_params_list, BASIS_TRANSFORMATION)
        self.compare_with_plot(model_results)

    def compare_with_plot(self, model_projection_list):
        TrajectoryPlotter(self).plot_models(
            model_projection_list,
            data_elements=[0],  # [0, 1, 2]
            plot_type=self.trajectory.params[PLOT_TYPE],
            plot_tics=self.trajectory.params[PLOT_TICS],
            components=self.trajectory.params[N_COMPONENTS]
        )


class MultiTrajectory:
    def __init__(self, kwargs_list, params):
        self.trajectories: list[DataTrajectory] = [DataTrajectory(**kwargs) for kwargs in kwargs_list]
        self.params = params

    def compare_pcs(self, algorithms):
        for algorithm in algorithms:
            principal_components = []
            for trajectory in self.trajectories:
                res = trajectory.get_model_result(algorithm)
                principal_components.append(res['model'].eigenvectors)
            pcs = np.asarray(principal_components)
            MultiTrajectoryPlotter(interactive=False).plot_principal_components(algorithm, pcs,
                                                                                self.params[N_COMPONENTS])

    def get_trajectories_by_index(self, traj_nrs: [list[int], None]):
        if traj_nrs is None:  # take all the trajectories
            return self.trajectories
        else:
            sorted_traj_nrs = sorted(i for i in traj_nrs if i < len(self.trajectories))
            return list(itemgetter(*sorted_traj_nrs)(self.trajectories))

    @staticmethod
    def get_trajectory_combos(trajectories, model_params):
        traj_results = []
        for trajectory in trajectories:
            res = trajectory.get_model_result(model_params)
            res.update({'traj': trajectory})
            traj_results.append(res)
        return list(combinations(traj_results, 2))

    @staticmethod
    def get_all_similarities_from_combos(combos, pc_nr_list=None, plot=False):
        if plot and pc_nr_list is None:
            raise ValueError('Trajectories can not be compared to each other, because the `pc_nr_list` is not given.')

        all_similarities = []
        for combi in combos:
            pc_0_matrix = combi[0]['model'].eigenvectors.T
            pc_1_matrix = combi[1]['model'].eigenvectors.T
            cos_matrix = cosine_similarity(np.real(pc_0_matrix), np.real(pc_1_matrix))
            sorted_similarity_indexes = np.argmax(np.abs(cos_matrix), axis=1)
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
                ArrayPlotter(interactive=False).matrix_plot(
                    cos_matrix,
                    title_prefix=f'{combi[0]["model"]}\n'
                                 f'{combi[0]["traj"].filename} & {combi[1]["traj"].filename}\n'
                                 f'PC Similarity',
                    bottom_text=sim_text,
                    xy_label='Principal Component Number'
                )
            all_similarities.append(combo_sim_of_n_pcs)
        return np.asarray(all_similarities)

    def compare_all_trajectories(self, traj_nrs: [list[int], None], model_params_list: list[dict],
                                 pc_nr_list: [list[int], None]):
        trajectories = self.get_trajectories_by_index(traj_nrs)

        for model_params in model_params_list:
            result_combos = self.get_trajectory_combos(trajectories, model_params)
            all_sim_matrix = self.get_all_similarities_from_combos(result_combos)

            if pc_nr_list is None:
                ArrayPlotter(interactive=False).plot_2d(
                    np.mean(all_sim_matrix, axis=0),
                    title_prefix=f'{self.params[TRAJECTORY_NAME]}\n{model_params}\n'
                                 'Similarity value of all trajectories',
                    xlabel='Principal component number',
                    ylabel='Similarity value',
                )
            else:
                for pc_index in pc_nr_list:
                    tria = np.zeros((len(trajectories), len(trajectories)))
                    sim_text = f'Similarity of all {np.mean(all_sim_matrix[:, pc_index])}'
                    print(sim_text)
                    tria[np.triu_indices(len(trajectories), 1)] = all_sim_matrix[:, pc_index]
                    tria = tria + tria.T
                    ArrayPlotter(interactive=False).matrix_plot(
                        tria,
                        title_prefix=f'{self.params[TRAJECTORY_NAME]}\n{model_params}\n'
                                     f'Trajectory Similarities for {pc_index}-Components',
                        bottom_text=sim_text,
                        xy_label='Trajectory number'
                    )

    def compare_trajectory_combos(self, traj_nrs, model_params_list, pc_nr_list):
        trajectories = self.get_trajectories_by_index(traj_nrs)

        for model_params in model_params_list:
            result_combos = self.get_trajectory_combos(trajectories, model_params)
            self.get_all_similarities_from_combos(result_combos, pc_nr_list, plot=True)


class GridSearchTrajectory:
    def __init__(self, trajectory):
        self.trajectory: DataTrajectory = trajectory

    def estimate(self, grid_param):
        model = ParameterModel()
        inp = self.trajectory.data_input()
        cv = [(slice(None), slice(None))]  # get rid of cross validation
        clf = GridSearchCV(model, grid_param, cv=cv, verbose=1)
        print('Searching for best model...')
        clf.fit(inp, n_components=self.trajectory.params[N_COMPONENTS])
        result_df = pd.DataFrame(clf.cv_results_)
        current_result_path = Path('analyse_results') / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        os.mkdir(current_result_path)
        goal_path = current_result_path / (self.trajectory.filename[:-4] + '.csv')
        result_df.to_csv(goal_path)
        print(f'Grid search results successfully saved into: {goal_path}')
