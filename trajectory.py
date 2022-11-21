from pathlib import Path

import mdtraj as md
import numpy as np
import pyemma.coordinates as coor

from plotter import ArrayPlotter, TrajectoryPlotter, MultiTrajectoryPlotter
from utils.algorithms.pca import MyPCA, TruncatedPCA, KernelFromCovPCA
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
                self.traj = md.load_dcd(self.filepath, top=self.topology_path, atom_indices=range(0, 710))
            else:
                self.traj = md.load(self.filepath, top=self.topology_path)
            self.dim = {TIME_FRAMES: self.traj.xyz.shape[0],
                        ATOMS: self.traj.xyz.shape[1],
                        COORDINATES: self.traj.xyz.shape[2]}
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
            TRAJECTORY_NAME: params.get(TRAJECTORY_NAME, 'Not Found')  # TODO: Should raise Error?
        }

        self.x_coordinates = self.filter_coordinates_by_coordinates(0)
        self.y_coordinates = self.filter_coordinates_by_coordinates(1)
        self.z_coordinates = self.filter_coordinates_by_coordinates(2)
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
        return self.filter_coordinates_by_atom_index(self.carbon_alpha_indexes)

    @property
    def basis_transformed_coordinates(self):
        np.random.seed(self.params[RANDOM_SEED])
        return basis_transform(self.traj.xyz, self.dim[COORDINATES])

    def filter_coordinates_by_time_frames(self, element_list, ac_only=False):
        coordinates_dict = self.alpha_carbon_coordinates if ac_only else self.atom_coordinates
        return coordinates_dict[element_list, :, :]

    def filter_coordinates_by_atom_index(self, element_list, ac_only=False):
        coordinates_dict = self.alpha_carbon_coordinates if ac_only else self.atom_coordinates
        return coordinates_dict[:, element_list, :]

    def filter_coordinates_by_coordinates(self, element_list, ac_only=False):
        coordinates_dict = self.alpha_carbon_coordinates if ac_only else self.atom_coordinates
        return coordinates_dict[:, :, element_list]

    def get_model_and_projection(self, model_name, inp=None):
        # TODO: This function is too long, and not good, rewrite models to more changeability with different parameters
        #  get input, component_nr, matrix/tensor, kernel on/from covariance/pearson, pca/tica,
        #  truncation, comad
        print(f'Running {model_name}...')
        if inp is None:
            inp = self.flattened_coordinates
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
            # this and following tensor models works only for alpha carbon coordinates, not for angles -> TODO
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

    def compare(self, model_names):
        model_results = []
        for model_name in model_names:
            try:
                model_results.append(self.get_model_result(model_name))
            except np.linalg.LinAlgError as e:
                print(f'Eigenvalue decomposition for model `{model_name}` couldn\'t be calculated:\n {e}')
            except AssertionError as e:
                print(f'{e}')
        self.compare_with_plot(model_results)

    def get_model_result(self, model_name):
        if self.params[USE_ANGLES]:
            flattened_angles = np.concatenate([self.phi[1], self.psi[1]], axis=1)
            model, projection = self.get_model_and_projection(model_name, flattened_angles)
            ex_var = explained_variance(model.eigenvalues, self.params[N_COMPONENTS])
            return {MODEL: model, PROJECTION: projection, TITLE_PREFIX: 'Flattened Angles',
                    EXPLAINED_VAR: f'\nExplained var: {ex_var}'}
        else:
            model, projection = self.get_model_and_projection(model_name)
            ex_var = explained_variance(model.eigenvalues, self.params[N_COMPONENTS])
            return {MODEL: model, PROJECTION: projection, EXPLAINED_VAR: f'\nExplained var: {ex_var}'}

    def compare_with_carbon_alpha_and_all_atoms(self, model_names):
        model_results = self.get_model_results_with_different_param(model_names, CARBON_ATOMS_ONLY)
        self.compare_with_plot(model_results)

    def compare_with_basis_transformation(self, model_names):
        model_results = self.get_model_results_with_different_param(model_names, BASIS_TRANSFORMATION)
        self.compare_with_plot(model_results)

    def get_model_results_with_different_param(self, model_names, parameter):
        model_results = []
        for model_name in model_names:
            model, projection = self.get_model_and_projection(model_name)
            model_results.append({MODEL: model, PROJECTION: projection,
                                  TITLE_PREFIX: f'{parameter}: {self.params[parameter]}\n'})
            self.params[parameter] = not self.params[parameter]
            model, projection = self.get_model_and_projection(model_name)
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
                input_list = [self.filter_coordinates_by_coordinates(c, ac_only=self.params[CARBON_ATOMS_ONLY])
                              for c in range(len(self.dim))]
                # input_list = list(
                #     map(lambda c: self.filter_coordinates_by_coordinates(c, ac_only=self.params[CARBON_ATOMS_ONLY]),
                #         range(len(self.dim))))
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
            from pymol import cmd
            cmd.load(self.filepath)
            cmd.save(self.goal_filepath)
            cmd.delete('*')
            raise NotImplementedError(f'{self.topology_filename} cannot converted into {self.filename}')
        print('Convert successful.')


class MultiTrajectory:
    def __init__(self, kwargs_list, params):
        self.trajectories = [DataTrajectory(**kwargs) for kwargs in kwargs_list]
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
