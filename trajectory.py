import warnings
from pathlib import Path

import mdtraj as md
import numpy as np
import pyemma.coordinates as coor
from mdtraj.utils import deprecated

from plotter import ArrayPlotter
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
