import warnings
from pathlib import Path

import mdtraj as md
import numpy as np
import pyemma.coordinates as coor
from mdtraj import Trajectory

from utils.algorithms.tensor_dim_reductions.daanccer import DAANCCER
from utils.algorithms.tsne import MyTSNE, MyTimeLaggedTSNE
from utils.math import basis_transform, explained_variance
from utils.matrix_tools import reconstruct_matrix
from utils.param_keys.param_key import *


class TrajectoryFile:
    def __init__(self, filename, topology_filename, folder_path):
        self.root_path: Path = Path(folder_path)
        self.filename: str = filename
        self.topology_filename: str = topology_filename

    @property
    def filepath(self) -> str:
        return str(self.root_path / self.filename)

    @property
    def topology_path(self) -> str:
        return str(self.root_path / self.topology_filename)


class DataTrajectory(TrajectoryFile):
    def __init__(self, filename, topology_filename, folder_path='data/2f4k', params=None, atoms=None):
        super().__init__(filename, topology_filename, folder_path)
        try:
            print(f"Loading trajectory {self.filename}...")
            if str(self.filename).endswith('dcd'):
                self.traj: Trajectory = md.load_dcd(self.filepath, top=self.topology_path, atom_indices=atoms)
            else:
                self.traj: Trajectory = md.load(self.filepath, top=self.topology_path)
            # TODO: Should the superposing not be done especially for the Alpha Carbon Atoms only?
            self.traj: Trajectory = self.traj.superpose(self.traj).center_coordinates(mass_weighted=True)
            # TODO:
            # 1. Range of xyz-coordinates
            # TODO: 2. Normalize again?
            self.traj.xyz = (self.traj.xyz - np.mean(self.traj.xyz, axis=0)[np.newaxis, :, :]) / np.std(self.traj.xyz,
                                                                                                        axis=0)
            self.dim: dict = {TIME_FRAMES: self.traj.xyz.shape[TIME_DIM],
                              ATOMS: self.traj.xyz.shape[ATOM_DIM],
                              COORDINATES: self.traj.xyz.shape[COORDINATE_DIM]}
            self.phi: np.ndarray = md.compute_phi(self.traj)
            self.psi: np.ndarray = md.compute_psi(self.traj)
        except IOError:
            raise FileNotFoundError(f"Cannot load {self.filepath} or {self.topology_path}.")
        else:
            print(f"Trajectory `{self.traj}` successfully loaded.")

        if params is None:
            params = {}
        self.params: dict = {
            CARBON_ATOMS_ONLY: params.get(CARBON_ATOMS_ONLY, True),
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            BASIS_TRANSFORMATION: params.get(BASIS_TRANSFORMATION, False),
            RANDOM_SEED: params.get(RANDOM_SEED, 42),
            USE_ANGLES: params.get(USE_ANGLES, True),
            TRAJECTORY_NAME: params.get(TRAJECTORY_NAME, 'Not Found')
        }

        self.x_coordinates = self._filter_coordinates_by_coordinates(0)
        self.y_coordinates = self._filter_coordinates_by_coordinates(1)
        self.z_coordinates = self._filter_coordinates_by_coordinates(2)
        self.coordinate_mins = {X: self.x_coordinates.min(), Y: self.y_coordinates.min(), Z: self.z_coordinates.min()}
        self.coordinate_maxs = {X: self.x_coordinates.max(), Y: self.y_coordinates.max(), Z: self.z_coordinates.max()}
        self.__check_init_params()

    def __check_init_params(self):
        if self.params[N_COMPONENTS] is None:
            self.params[N_COMPONENTS] = self.max_components

    @property
    def atom_coordinates(self) -> np.ndarray:
        if self.params[BASIS_TRANSFORMATION]:
            return self.__basis_transformed_coordinates()
        else:
            return self.traj.xyz

    @property
    def flattened_coordinates(self) -> np.ndarray:
        if self.params[CARBON_ATOMS_ONLY]:
            self.dim[ATOMS] = len(self.alpha_carbon_indexes)
            return self.alpha_carbon_coordinates.reshape(self.dim[TIME_FRAMES],
                                                         self.dim[ATOMS] * self.dim[COORDINATES])
        else:
            return self.atom_coordinates.reshape(self.dim[TIME_FRAMES], self.dim[ATOMS] * self.dim[COORDINATES])

    @property
    def max_components(self) -> int:
        if self.params[USE_ANGLES]:
            return self.phi[ANGLE_INDICES].shape[0] * 2
        else:
            if self.params[CARBON_ATOMS_ONLY]:
                self.dim[ATOMS] = len(self.alpha_carbon_indexes)
            return self.dim[ATOMS] * self.dim[COORDINATES]

    @property
    def alpha_carbon_indexes(self) -> list:
        return [a.index for a in self.traj.topology.atoms if a.name == 'CA']

    @property
    def alpha_carbon_coordinates(self) -> np.ndarray:
        return self._filter_coordinates_by_atom_index(self.alpha_carbon_indexes)

    def __basis_transformed_coordinates(self) -> np.ndarray:
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

    def get_model_result(self, model_parameters: dict, log: bool = True) -> dict:
        """
        Returns a dict of all the important result values. Used for analysing the different models
        :param model_parameters: dict
            The input parameters for the model
        :param log: bool
            Enables the log output while running the program (default: True)
        :return: dict of the results: {MODEL, PROJECTION, EXPLAINED_VAR, INPUT_PARAMS}
        """
        model, projection = self.get_model_and_projection(model_parameters, log=log)
        ex_var = explained_variance(model.eigenvalues, self.params[N_COMPONENTS])
        return {MODEL: model, PROJECTION: projection, EXPLAINED_VAR: ex_var, INPUT_PARAMS: model_parameters}

    def get_model_and_projection(self, model_parameters: dict, inp: np.ndarray = None, log: bool = True):
        """
        This method is fitting a model with the given parameters :model_parameters: and
        the inp(ut) data is transformed on the model.
        @param model_parameters: dict
            The input parameters for the model.
        @param inp: np.ndarray
            Input data for the model (optional), (default: None -> calculated on the basis of the model_parameters)
        @param log: bool
            Enables the log output while running the program (default: True)
        @return: fitted model and transformed data
        """
        if log:
            print(f'Running {model_parameters}...')

        if inp is None:
            inp = self.data_input(model_parameters)

        if ALGORITHM_NAME not in model_parameters.keys():
            raise KeyError(f'{ALGORITHM_NAME} is a mandatory model parameter.')

        if model_parameters[ALGORITHM_NAME].startswith('original'):
            try:
                if model_parameters[ALGORITHM_NAME] == 'original_pca':
                    from sklearn.decomposition import PCA
                    pca = coor.pca(data=inp, dim=self.params[N_COMPONENTS])
                    return pca, pca.get_output()[0]
                elif model_parameters[ALGORITHM_NAME] == 'original_tica':
                    tica = coor.tica(data=inp, lag=model_parameters[LAG_TIME], dim=self.params[N_COMPONENTS])
                    return tica, tica.get_output()[0]
                elif model_parameters[ALGORITHM_NAME] == 'original_tsne':
                    tsne = MyTSNE(n_components=self.params[N_COMPONENTS])
                    return tsne, tsne.fit_transform(inp)
                elif model_parameters[ALGORITHM_NAME] == 'original-tl-tsne':
                    tsne = MyTimeLaggedTSNE(lag_time=model_parameters[LAG_TIME], n_components=self.params[N_COMPONENTS])
                    return tsne, tsne.fit_transform(inp)
                else:
                    warnings.warn(f'No original algorithm was found with name: {model_parameters[ALGORITHM_NAME]}')
            except TypeError:
                raise TypeError(f'Input data of the function is not correct. '
                                f'Original algorithms take only 2-n-dimensional ndarray')
        else:
            model = DAANCCER(**model_parameters)
            return model, model.fit_transform(inp, n_components=self.params[N_COMPONENTS])

    def data_input(self, model_parameters: dict = None) -> np.ndarray:
        """
        Determines the input data for the model on the basis of the model_parameters and the trajectory parameters.
        If the model_parameters are unknown, then the input for TENSOR_NDIM is used.
        @param model_parameters: dict
            The input parameters for the model.
        @return: np.ndarray
            The input data for the model
        """
        try:
            if model_parameters is None:
                n_dim = TENSOR_NDIM
            else:
                n_dim = model_parameters[NDIM]
        except KeyError as e:
            raise KeyError(f'Model-parameter-dict needs the key: {e}. Set to ´2´ or ´3´.')

        if self.params[USE_ANGLES]:
            if n_dim == MATRIX_NDIM:
                return np.concatenate([self.phi[DIHEDRAL_ANGLE_VALUES], self.psi[DIHEDRAL_ANGLE_VALUES]], axis=1)
            else:
                return np.concatenate([self.phi[DIHEDRAL_ANGLE_VALUES][:, :, np.newaxis],
                                       self.psi[DIHEDRAL_ANGLE_VALUES][:, :, np.newaxis]], axis=2)
        else:
            if n_dim == MATRIX_NDIM:
                return self.flattened_coordinates
            else:
                if self.params[CARBON_ATOMS_ONLY]:
                    return self.alpha_carbon_coordinates
                else:
                    return self.atom_coordinates

    def get_model_results_with_changing_trajectory_parameter(
            self,
            model_params_list: list[dict],
            trajectory_key_parameter: str
    ) -> list[dict]:
        """
        This method helps to compare the same model with different input of the same trajectory
        (e.g. with and without basis-transformation).
        @param model_params_list: list[dict]
            Different model input parameters, saved in a list.
        @param trajectory_key_parameter: str
            key of the trajectory parameter which should be used to determine the different input.
        @return:
        """
        model_results = []
        for model_params in model_params_list:
            model, projection = self.get_model_and_projection(model_params)
            model_results.append({MODEL: model, PROJECTION: projection,
                                  TITLE_PREFIX: f'{trajectory_key_parameter}: '
                                                f'{self.params[trajectory_key_parameter]}\n'})
            self.params[trajectory_key_parameter] = not self.params[trajectory_key_parameter]
            model, projection = self.get_model_and_projection(model_params)
            model_results.append({MODEL: model, PROJECTION: projection,
                                  TITLE_PREFIX: f'{trajectory_key_parameter}: '
                                                f'{self.params[trajectory_key_parameter]}\n'})
            self.params[trajectory_key_parameter] = not self.params[trajectory_key_parameter]
        return model_results

    def get_reconstructed_traj(self, model_parameters: dict) -> np.ndarray:
        """
        Calculate the reconstructed trajectory data which would be constructed, with the given model parameters.
        @param model_parameters: dict
            The input parameters for the model.
        @return: np.ndarray
            reconstructed trajectory coordinates
        """
        model, projection = self.get_model_and_projection(model_parameters)
        if isinstance(model, DAANCCER):
            return model.reconstruct(projection[0])
        else:
            eigenvectors = model.eigenvectors
            input_data = self.data_input(model_parameters)
            reconstructed_matrix = reconstruct_matrix(projection[0], eigenvectors, self.params[N_COMPONENTS],
                                                      mean=np.mean(input_data, axis=0))
            return reconstructed_matrix.reshape((self.dim[TIME_FRAMES],
                                                 self.dim[ATOMS],
                                                 self.dim[COORDINATES]))


class TrajectorySubset(DataTrajectory):
    def __init__(self, time_window_size=None, quantity=1, **kwargs):
        super().__init__(**kwargs)

    def make_subsets(self):

        return None

    def data_input(self, model_parameters: dict = None) -> np.ndarray:
        """
        TODO
        @param model_parameters:
        @return:
        """


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
