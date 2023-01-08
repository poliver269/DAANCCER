import warnings
from itertools import combinations
from operator import itemgetter
from pathlib import Path

import mdtraj as md
import numpy as np
import pyemma.coordinates as coor
from mdtraj.utils import deprecated
from sklearn.metrics.pairwise import cosine_similarity

from plotter import ArrayPlotter, TrajectoryPlotter, MultiTrajectoryPlotter
from utils.algorithms.pca import MyPCA, TruncatedPCA, KernelFromCovPCA
from utils.algorithms.tensor_dim_reductions import ParameterModel
from utils.algorithms.tensor_dim_reductions.pca import (TensorPCA, TensorPearsonCovPCA, TensorKernelOnPearsonCovPCA,
                                                        TensorKernelOnCovPCA, TensorKernelFromCovPCA,
                                                        TensorKernelFromComadPCA)
from utils.algorithms.tensor_dim_reductions.tica import (TensorTICA, TensorKernelOnCovTICA,
                                                         TensorKernelOnPearsonCovTICA,
                                                         TensorKernelFromCovTICA, TensorKernelFromCoMadTICA,
                                                         TensorKernelOnCoMadTICA)
from utils.algorithms.tica import MyTICA, TruncatedTICA, KernelFromCovTICA
from utils.math import basis_transform, explained_variance
from utils.matrix_tools import calculate_pearson_correlations, calculate_symmetrical_kernel_from_matrix
from utils.param_key import *


class TrajectoryFile:
    def __init__(self, filename, topology_filename, folder_path):
        self.root_path = Path(folder_path)
        self.filename = filename
        self.topology_filename = topology_filename

    @property
    def filepath(self):
        return str(self.root_path / self.filename)

    @property
    def topology_path(self):
        return str(self.root_path / self.topology_filename)


class DataTrajectory(TrajectoryFile):
    def __init__(self, filename, topology_filename, folder_path='data/2f4k', params=None):
        super().__init__(filename, topology_filename, folder_path)
        try:
            print(f"Loading trajectory {filename}...")
            if str(self.filename).endswith('dcd'):
                self.traj = md.load_dcd(self.filepath, top=self.topology_path)
            else:
                self.traj = md.load(self.filepath, top=self.topology_path)
            self.traj = self.traj.superpose(self.traj).center_coordinates(mass_weighted=True)
            self.dim = {TIME_FRAMES: self.traj.xyz.shape[TIME_DIM],
                        ATOMS: self.traj.xyz.shape[ATOM_DIM],
                        COORDINATES: self.traj.xyz.shape[COORDINATE_DIM]}
            self.phi = md.compute_phi(self.traj)
            self.psi = md.compute_psi(self.traj)
        except IOError:
            raise FileNotFoundError(f"Cannot load {self.filepath} or {self.topology_path}.")
        else:
            print(f"{self.traj} successfully loaded.")

        if params is None:
            params = {}
        self.params = {
            PLOT_TYPE: params.get(PLOT_TYPE, COLOR_MAP),
            PLOT_TICS: params.get(PLOT_TICS, True),
            STANDARDIZED_PLOT: params.get(STANDARDIZED_PLOT, False),
            CARBON_ATOMS_ONLY: params.get(CARBON_ATOMS_ONLY, True),
            INTERACTIVE: params.get(INTERACTIVE, True),
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            LAG_TIME: params.get(LAG_TIME, 10),
            TRUNCATION_VALUE: params.get(TRUNCATION_VALUE, 0),
            BASIS_TRANSFORMATION: params.get(BASIS_TRANSFORMATION, False),
            RANDOM_SEED: params.get(RANDOM_SEED, 30),
            USE_ANGLES: params.get(USE_ANGLES, True),
            TRAJECTORY_NAME: params.get(TRAJECTORY_NAME, 'Not Found')
        }

        self.x_coordinates = self._filter_coordinates_by_coordinates(0)
        self.y_coordinates = self._filter_coordinates_by_coordinates(1)
        self.z_coordinates = self._filter_coordinates_by_coordinates(2)
        self.coordinate_mins = {X: self.x_coordinates.min(), Y: self.y_coordinates.min(), Z: self.z_coordinates.min()}
        self.coordinate_maxs = {X: self.x_coordinates.max(), Y: self.y_coordinates.max(), Z: self.z_coordinates.max()}

    @property
    def atom_coordinates(self):
        if self.params[BASIS_TRANSFORMATION]:
            return self.basis_transformed_coordinates
        else:
            return self.traj.xyz

    @property
    def flattened_coordinates(self):
        if self.params[CARBON_ATOMS_ONLY]:
            return self.alpha_carbon_coordinates.reshape(self.dim[TIME_FRAMES],
                                                         len(self.carbon_alpha_indexes) * self.dim[COORDINATES])
        else:
            return self.atom_coordinates.reshape(self.dim[TIME_FRAMES], self.dim[ATOMS] * self.dim[COORDINATES])

    @property
    def carbon_alpha_indexes(self):
        return [a.index for a in self.traj.topology.atoms if a.name == 'CA']

    @property
    def alpha_carbon_coordinates(self):
        return self._filter_coordinates_by_atom_index(self.carbon_alpha_indexes)

    @property
    def basis_transformed_coordinates(self):
        np.random.seed(self.params[RANDOM_SEED])
        return basis_transform(self.traj.xyz, self.dim[COORDINATES])

    def _filter_coordinates_by_time_frames(self, element_list, ac_only=False):
        coordinates_dict = self.alpha_carbon_coordinates if ac_only else self.atom_coordinates
        return coordinates_dict[element_list, :, :]

    def _filter_coordinates_by_atom_index(self, element_list, ac_only=False):
        coordinates_dict = self.alpha_carbon_coordinates if ac_only else self.atom_coordinates
        return coordinates_dict[:, element_list, :]

    def _filter_coordinates_by_coordinates(self, element_list, ac_only=False):
        coordinates_dict = self.alpha_carbon_coordinates if ac_only else self.atom_coordinates
        return coordinates_dict[:, :, element_list]

    @deprecated
    def get_model_and_projection_by_name(self, model_name: str, inp: np.ndarray = None):
        if inp is None:
            inp = self._determine_input(model_name)
        if model_name == 'pca':
            pca = coor.pca(data=inp, dim=self.params[N_COMPONENTS])
            return pca, pca.get_output()
        elif model_name == 'mypca':
            pca = MyPCA()
            return pca, [pca.fit_transform(inp, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'trunc_pca':
            pca = TruncatedPCA(self.params[TRUNCATION_VALUE])
            return pca, [pca.fit_transform(inp, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'pca_kernel_only':
            pca = KernelFromCovPCA()
            return pca, [pca.fit_transform(inp, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_pca':
            pca = TensorPCA()
            return pca, [pca.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_pearson_pca':
            ppca = TensorPearsonCovPCA()
            return ppca, [ppca.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_pearson_kernel_pca':
            pkpca = TensorKernelOnPearsonCovPCA()
            return pkpca, [pkpca.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_kernel_pca':
            ckpca = TensorKernelOnCovPCA()
            return ckpca, [ckpca.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_ko_pca':
            ko_pca = TensorKernelFromCovPCA()
            return ko_pca, [ko_pca.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_kernel_mad_pca':
            ko_med_pca = TensorKernelFromComadPCA()
            return ko_med_pca, [ko_med_pca.fit_transform(self.alpha_carbon_coordinates,
                                                         n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tica':
            tica = coor.tica(data=inp, lag=self.params[LAG_TIME], dim=self.params[N_COMPONENTS])
            return tica, tica.get_output()
        elif model_name == 'mytica':
            tica = MyTICA(lag_time=self.params[LAG_TIME])
            return tica, [tica.fit_transform(inp, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'trunc_tica':
            tica = TruncatedTICA(lag_time=self.params[LAG_TIME], trunc_value=self.params[TRUNCATION_VALUE])
            return tica, [tica.fit_transform(inp, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'kernel_only_tica':
            tica = KernelFromCovTICA(lag_time=self.params[LAG_TIME])
            return tica, [tica.fit_transform(inp, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_tica':
            tica = TensorTICA(lag_time=self.params[LAG_TIME])
            return tica, [tica.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_kernel_tica':
            tica = TensorKernelOnCovTICA(lag_time=self.params[LAG_TIME])
            return tica, [tica.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_kp_tica':
            tica = TensorKernelOnPearsonCovTICA(lag_time=self.params[LAG_TIME])
            return tica, [tica.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_ko_tica':
            tica = TensorKernelFromCovTICA(lag_time=self.params[LAG_TIME])
            return tica, [tica.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_comad_tica':
            tica = TensorKernelFromCoMadTICA(lag_time=self.params[LAG_TIME])
            return tica, [tica.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'tensor_comad_kernel_tica':
            tica = TensorKernelOnCoMadTICA(lag_time=self.params[LAG_TIME])
            return tica, [tica.fit_transform(self.alpha_carbon_coordinates, n_components=self.params[N_COMPONENTS])]
        else:
            raise ValueError(f'Model with name \"{model_name}\" does not exists.')

    def get_model_and_projection(self, model_parameters: dict, inp: np.ndarray = None):
        print(f'Running {model_parameters}...')
        if inp is None:
            inp = self._determine_input(model_parameters)
        if model_parameters[ALGORITHM_NAME].startswith('original'):
            try:
                if model_parameters[ALGORITHM_NAME] == 'original_pca':
                    pca = coor.pca(data=inp, dim=self.params[N_COMPONENTS])
                    return pca, pca.get_output()
                elif model_parameters[ALGORITHM_NAME] == 'original_tica':
                    tica = coor.tica(data=inp, lag=self.params[LAG_TIME], dim=self.params[N_COMPONENTS])
                    return tica, tica.get_output()
                else:
                    warnings.warn(f'No original algorithm was found with name: {model_parameters[ALGORITHM_NAME]}')
            except TypeError:
                raise TypeError(f'Input data of the function is not correct. '
                                f'Original algorithms take only 2-n-dimensional ndarray')
        else:
            model = ParameterModel(**model_parameters)
            return model, [model.fit_transform(inp, n_components=self.params[N_COMPONENTS])]

    def compare(self, model_parameter_list: list[dict, str]):
        model_results = []
        for model_parameters in model_parameter_list:
            try:
                model_results.append(self.get_model_result(model_parameters))
            except np.linalg.LinAlgError as e:
                warnings.warn(f'Eigenvalue decomposition for model `{model_parameters}` could not be calculated:\n {e}')
            except AssertionError as e:
                warnings.warn(f'{e}')
        self.compare_with_plot(model_results)

    def get_model_result(self, model_parameters: [str, dict]) -> dict:
        if isinstance(model_parameters, str):
            model, projection = self.get_model_and_projection_by_name(model_parameters)
        else:
            model, projection = self.get_model_and_projection(model_parameters)
        ex_var = explained_variance(model.eigenvalues, self.params[N_COMPONENTS])
        # TODO Add explained variance to the models, and if they don't have a parameter, than calculate here
        if self.params[PLOT_TYPE] == EXPL_VAR_PLOT:
            ArrayPlotter(interactive=False).plot_2d(
                ndarray_data=model.eigenvalues,
                title_prefix=f'Eigenvalues of\n{model}',
                xlabel='ComponentNr',
                ylabel='Eigenvalue'
            )
        return {MODEL: model, PROJECTION: projection, EXPLAINED_VAR: ex_var}

    def _determine_input(self, model_parameters: [str, dict]) -> np.ndarray:
        try:
            n_dim = MATRIX_NDIM if isinstance(model_parameters, str) else model_parameters['ndim']
        except KeyError as e:
            raise KeyError(f'Model-parameter-dict needs the key: {e}. Set to ´2´ or ´3´.')

        if self.params[USE_ANGLES]:
            if n_dim == MATRIX_NDIM:
                return np.concatenate([self.phi[DIHEDRAL_ANGEL_VALUES], self.psi[DIHEDRAL_ANGEL_VALUES]], axis=1)
            else:
                return np.asarray([self.phi[DIHEDRAL_ANGEL_VALUES], self.psi[DIHEDRAL_ANGEL_VALUES]])
        else:
            if n_dim == MATRIX_NDIM:

                return self.flattened_coordinates
            else:
                if self.params[CARBON_ATOMS_ONLY]:
                    return self.alpha_carbon_coordinates
                else:
                    return self.atom_coordinates

    def compare_with_carbon_alpha_and_all_atoms(self, model_names):
        model_results = self.get_model_results_with_different_param(model_names, CARBON_ATOMS_ONLY)
        self.compare_with_plot(model_results)

    def compare_with_basis_transformation(self, model_params_list):
        model_results = self.get_model_results_with_different_param(model_params_list, BASIS_TRANSFORMATION)
        self.compare_with_plot(model_results)

    def get_model_results_with_different_param(self, model_names, parameter):
        model_results = []
        for model_name in model_names:
            model, projection = self.get_model_and_projection_by_name(model_name)
            model_results.append({MODEL: model, PROJECTION: projection,
                                  TITLE_PREFIX: f'{parameter}: {self.params[parameter]}\n'})
            self.params[parameter] = not self.params[parameter]
            model, projection = self.get_model_and_projection_by_name(model_name)
            model_results.append({MODEL: model, PROJECTION: projection,
                                  TITLE_PREFIX: f'{parameter}: {self.params[parameter]}\n'})
            self.params[parameter] = not self.params[parameter]
        return model_results

    def compare_with_plot(self, model_projection_list):
        TrajectoryPlotter(self).plot_models(
            model_projection_list,
            data_elements=[0],  # [0, 1, 2]
            plot_type=self.params[PLOT_TYPE],
            plot_tics=self.params[PLOT_TICS],
            components=self.params[N_COMPONENTS]
        )

    def calculate_pearson_correlation_coefficient(self):
        mode = 'calculate_coefficients_than_mean_dimensions'
        if mode == 'coordinates_mean_first':  # Not used currently
            coordinates_mean = np.mean(self.atom_coordinates, axis=2)
            coefficient_mean = np.corrcoef(coordinates_mean.T)
        elif mode == 'calculate_mean_of_dimensions_to_matrix_than_correlation_coefficient':
            print('Calculate mean matrix...')
            if self.params[USE_ANGLES]:
                mean_matrix = np.mean(np.array([self.phi[DIHEDRAL_ANGEL_VALUES], self.psi[DIHEDRAL_ANGEL_VALUES]]),
                                      axis=0)
            else:
                mean_matrix = np.mean(self.alpha_carbon_coordinates, axis=2)
            print('Calculate correlation coefficient...')
            coefficient_mean = np.corrcoef(mean_matrix.T)
        elif mode == 'calculate_coefficients_than_mean_dimensions':
            if self.params[USE_ANGLES]:
                input_list = [self.phi[DIHEDRAL_ANGEL_VALUES], self.psi[DIHEDRAL_ANGEL_VALUES]]
            else:
                input_list = [self._filter_coordinates_by_coordinates(c, ac_only=self.params[CARBON_ATOMS_ONLY])
                              for c in range(len(self.dim))]
            print('Calculate correlation coefficient...')
            coefficient_mean = calculate_pearson_correlations(input_list, np.mean)
        else:
            raise ValueError('Invalid mode string was given')

        print('Fit Kernel on data...')
        d_matrix = calculate_symmetrical_kernel_from_matrix(coefficient_mean,
                                                            trajectory_name=self.params[TRAJECTORY_NAME])
        weighted_alpha_coeff_matrix = coefficient_mean - d_matrix

        title_prefix = ('Angles' if self.params[USE_ANGLES] else 'Coordinates') + f'. Pearson Coefficient. {mode}'
        ArrayPlotter().matrix_plot(weighted_alpha_coeff_matrix,
                                   title_prefix=title_prefix,
                                   xy_label='number of correlations',
                                   as_surface=self.params[PLOT_TYPE])


class TopologyConverter(TrajectoryFile):
    def __init__(self, filename, topology_filename, goal_filename, folder_path='data/2f4k'):
        super().__init__(filename, topology_filename, folder_path)
        self.goal_filename = goal_filename

    @property
    def goal_filepath(self):
        return str(self.root_path / self.goal_filename)

    def convert(self):
        import MDAnalysis
        print(f'Convert Topology {self.topology_filename} to {self.goal_filename}...')
        # noinspection PyProtectedMember
        if self.topology_filename.split('.')[-1].upper() in MDAnalysis._PARSERS:
            universe = MDAnalysis.Universe(self.topology_path)
            with MDAnalysis.Writer(self.goal_filepath) as writer:
                writer.write(universe)
        else:
            raise NotImplementedError(f'{self.topology_filename} cannot converted into {self.filename}')
        print('Convert successful.')


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
