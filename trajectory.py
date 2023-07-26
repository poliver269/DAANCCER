import random
import warnings
from contextlib import contextmanager
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
from mdtraj import Trajectory
from sklearn.decomposition import FastICA, PCA

from utils.algorithms.interfaces import DeeptimeTICAInterface, PyemmaTICAInterface, PyemmaPCAInterface
from utils.algorithms.tensor_dim_reductions.dropp import DROPP
from utils.algorithms.tsne import MyTSNE, MyTimeLaggedTSNE
from utils.errors import InvalidSubsetTrajectory
from utils.math import basis_transform, explained_variance
from utils.matrix_tools import reconstruct_matrix
from utils.param_keys import *
from utils.param_keys.model import ALGORITHM_NAME, NDIM, LAG_TIME
from utils.param_keys.model_result import MODEL, PROJECTION, TITLE_PREFIX, EXPLAINED_VAR, INPUT_PARAMS
from utils.param_keys.traj_dims import TIME_FRAMES, TIME_DIM, ATOMS, ATOM_DIM, COORDINATES, COORDINATE_DIM


class TrajectoryFile:
    def __init__(self, filename, folder_path, extra_filename=""):
        self.root_path: Path = Path(folder_path)
        self.filename: str = filename
        self.extra_filename = extra_filename

    @property
    def filepath(self) -> str:
        return str(self.root_path / self.filename)

    @property
    def extra_filepath(self) -> str:
        return str(self.root_path / self.extra_filename)


class DataTrajectory(TrajectoryFile):
    def __init__(self, filename, folder_path, extra_filename="", params=None, **kwargs):
        super().__init__(filename, folder_path, extra_filename)
        if params is None:
            params = {}
        self.params: dict = {
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            TRAJECTORY_NAME: params.get(TRAJECTORY_NAME, 'Not Found')
        }
        self.dim = {TIME_FRAMES: -1}

    def _check_init_params(self):
        """
        Checks if the initialized parameters are set correctly.
        """
        if self.params[N_COMPONENTS] is None:
            self.params[N_COMPONENTS] = self.max_components

    def _init_preprocessing(self):
        """
        Implement here the preprocessing steps for the DataTrajectory (if necessary)
        """
        pass

    @property
    def max_components(self) -> int:
        """
        Calculates the correlation dimension (Atom/Hours) of the data
        :return: int
            correlation dimension
        """
        raise NotImplementedError(f'In `{self.__class__}` the `max_components` variable is not implemented')

    def get_model_result(self, model_parameters: dict, log: bool = True) -> dict:
        """
        Returns a dict of all the important result values. Used for analysing the different models
        :param model_parameters: dict
            The input parameters for the model
        :param log: bool
            Enables the log output while running the program (default: True)
        :return: dict
            of the results with the keys: {MODEL, PROJECTION, EXPLAINED_VAR, INPUT_PARAMS}
        """
        model, projection = self.get_model_and_projection(model_parameters, log=log)
        try:
            # TODO@Oli&Prio4: Explained Variance not correctly calculated
            ex_var = explained_variance(model.explained_variance_, self.params[N_COMPONENTS])
        except AttributeError as e:
            warnings.warn(str(e))
            ex_var = 0
        return {MODEL: model, PROJECTION: projection, EXPLAINED_VAR: ex_var, INPUT_PARAMS: model_parameters}

    def data_input(self, model_parameters: dict = None) -> np.ndarray:
        """
        Determines the input data for the model on the basis of the model_parameters and the trajectory parameters.
        If the model_parameters are unknown, then the input for TENSOR_NDIM is used.
        @param model_parameters: dict
            The input parameters for the model.
        @return: np.ndarray
            The input data for the model
        """
        raise NotImplementedError(f"Data input could not be calculated, since the method `{self.data_input.__name__}` "
                                  f"is not implemented.")

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
                    pca = PCA(n_components=self.params[N_COMPONENTS])
                    return pca, pca.fit_transform(inp)
                elif model_parameters[ALGORITHM_NAME] == 'original_pyemma_pca':
                    pca = PyemmaPCAInterface(dim=self.params[N_COMPONENTS])
                    return pca, pca.fit_transform(inp)
                elif model_parameters[ALGORITHM_NAME] == 'original_tica':
                    tica = PyemmaTICAInterface(lag=model_parameters[LAG_TIME], dim=self.params[N_COMPONENTS])
                    return tica, tica.fit_transform(inp)
                elif model_parameters[ALGORITHM_NAME] == 'original_deeptime_tica':
                    tica = DeeptimeTICAInterface(dim=self.params[N_COMPONENTS], lagtime=model_parameters[LAG_TIME])
                    return tica, tica.fit_transform(inp)
                elif model_parameters[ALGORITHM_NAME] == 'original_tsne':
                    tsne = MyTSNE(n_components=self.params[N_COMPONENTS])
                    return tsne, tsne.fit_transform(inp)
                elif model_parameters[ALGORITHM_NAME] == 'original-tl-tsne':
                    tsne = MyTimeLaggedTSNE(lag_time=model_parameters[LAG_TIME], n_components=self.params[N_COMPONENTS])
                    return tsne, tsne.fit_transform(inp)
                elif model_parameters[ALGORITHM_NAME] == 'original_ica':
                    ica = FastICA(n_components=self.params[N_COMPONENTS])
                    return ica, ica.fit_transform(inp)
                else:
                    warnings.warn(f'No original algorithm was found with name: {model_parameters[ALGORITHM_NAME]}')
            except TypeError:
                raise TypeError(f'Input data of the function is not correct. '
                                f'Original algorithms take only 2-n-dimensional ndarray')
        else:
            model = DROPP(**model_parameters)
            return model, model.fit_transform(inp, n_components=self.params[N_COMPONENTS])


class WeatherTrajectory(DataTrajectory):
    def __init__(self, filename, folder_path='data/', params=None, **kwargs):
        super().__init__(filename, folder_path, params=params, **kwargs)
        try:
            print(f"Loading trajectory {self.filename}...")
            self.weather_df = pd.read_csv(self.filepath)
            self.weather_df = self.weather_df.loc[:, (round(self.weather_df) != 0).any()]

            self.feat_traj = np.array(list(map(np.stack, self.weather_df.to_numpy())))
        except IOError:
            raise FileNotFoundError(f"Cannot load {self.filepath}.")

        self.params.update({REDUCEE_FEATURE: params.get(REDUCEE_FEATURE, "temperature")})

        features_encoding = {
            'temperature': 0,
            'radiation_direct_horizontal': 1,
            'radiation_diffuse_horizontal': 2
        }
        if self.params[REDUCEE_FEATURE] in features_encoding:  # TODO string wird überschrieben mit Encoding-Index
            self.params[REDUCEE_FEATURE] = features_encoding[self.params[REDUCEE_FEATURE]]
        else:
            raise KeyError(f'WeatherTrajectory needs a specific parameter: `{REDUCEE_FEATURE}`. '
                           f'Set to one of {features_encoding.keys()}')

        self._check_init_params()
        self._init_preprocessing(self.params[REDUCEE_FEATURE])

    def _init_preprocessing(self, feature=2):
        def get_feature(list_as_text):
            result = eval(list_as_text)
            return result[feature]

        self.weather_df = self.weather_df.applymap(get_feature)
        # noinspection PyUnresolvedReferences
        self.weather_df = self.weather_df.loc[:, (round(self.weather_df) != 0).any()]

        self.feat_traj = np.array(list(map(np.stack, self.weather_df.to_numpy())))
        self.feat_traj = (self.feat_traj - np.mean(self.feat_traj, axis=0)[np.newaxis, :]) / np.std(self.feat_traj,
                                                                                                    axis=0)
        self.dim[TIME_FRAMES] = self.feat_traj.shape[DUMMY_ZERO]

    @property
    def max_components(self) -> int:
        return len(self.weather_df.columns)

    def data_input(self, model_parameters: dict = None) -> np.ndarray:
        ft_traj = self.feat_traj

        try:
            if model_parameters is None:
                n_dim = TENSOR_NDIM
            else:
                n_dim = model_parameters[NDIM]
        except KeyError as e:
            raise KeyError(f'Model-parameter-dict needs the key: {e}. Set to ´2´ or ´3´.')

        if n_dim == MATRIX_NDIM:
            # print("INFO: Flat feature trajectory has shape", ft_traj.shape)
            pass
        else:
            ft_traj = np.array([np.array([np.array([hour]) for hour in day]) for day in ft_traj])
            # print("INFO: Feature trajectory has shape", ft_traj.shape)
        return ft_traj


class ProteinTrajectory(DataTrajectory):
    def __init__(self, filename, topology_filename=None, folder_path='data/2f4k', params=None, atoms=None, **kwargs):
        super().__init__(filename, folder_path, extra_filename=topology_filename, params=params, **kwargs)
        try:
            print(f"Loading trajectory {self.filename}...")
            if str(self.filename).endswith('dcd'):
                self.traj: Trajectory = md.load_dcd(self.filepath, top=self.extra_filepath, atom_indices=atoms)
            else:
                self.traj: Trajectory = md.load(self.filepath, top=self.extra_filepath, atom_indices=atoms)
            self.reference_pdb = md.load_pdb(self.extra_filepath, atom_indices=atoms)
            self.dim: dict = {TIME_FRAMES: self.traj.xyz.shape[TIME_DIM],
                              ATOMS: self.traj.xyz.shape[ATOM_DIM],
                              COORDINATES: self.traj.xyz.shape[COORDINATE_DIM]}
            self.phi: np.ndarray = md.compute_phi(self.traj)
            self.psi: np.ndarray = md.compute_psi(self.traj)
        except IOError:
            raise FileNotFoundError(f"Cannot load {self.filepath} or {self.extra_filepath}.")
        else:
            print(f"Trajectory `{self.traj}` successfully loaded.")

        self.params.update({
            CARBON_ATOMS_ONLY: params.get(CARBON_ATOMS_ONLY, True),
            BASIS_TRANSFORMATION: params.get(BASIS_TRANSFORMATION, False),
            RANDOM_SEED: params.get(RANDOM_SEED, 42),
            USE_ANGLES: params.get(USE_ANGLES, False),
            SUPERPOSING_INDEX: params.get(SUPERPOSING_INDEX, -1)
        })

        self._check_init_params()
        self._init_preprocessing()
        self._init_coordinate_params()

    def _init_coordinate_params(self):
        self.x_coordinates = self._filter_coordinates_by_coordinates(0)
        self.y_coordinates = self._filter_coordinates_by_coordinates(1)
        self.z_coordinates = self._filter_coordinates_by_coordinates(2)
        self.coordinate_mins = {X: self.x_coordinates.min(), Y: self.y_coordinates.min(),
                                Z: self.z_coordinates.min()}
        self.coordinate_maxs = {X: self.x_coordinates.max(), Y: self.y_coordinates.max(),
                                Z: self.z_coordinates.max()}

    def _init_preprocessing(self):
        if self.params[SUPERPOSING_INDEX] is not None:
            if self.params[SUPERPOSING_INDEX] < 0:
                # random.seed(self.params[RANDOM_SEED])
                superposing_frame = random.randint(0, self.dim[TIME_FRAMES])
                print(f'Random frame: {superposing_frame}')
            else:
                superposing_frame = self.params[SUPERPOSING_INDEX]
            self.traj = self.traj.superpose(
                self.traj, frame=superposing_frame).center_coordinates(mass_weighted=True)
        else:
            self.traj = self.traj.superpose(self.reference_pdb).center_coordinates(mass_weighted=True)
        self.traj.xyz = (self.traj.xyz - np.mean(self.traj.xyz, axis=0)[np.newaxis, :, :]) / np.std(self.traj.xyz,
                                                                                                    axis=0)

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

    def data_input(self, model_parameters: dict = None) -> np.ndarray:
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
        if isinstance(model, DROPP):
            return model.reconstruct(projection[0])
        else:
            eigenvectors = model.components_
            input_data = self.data_input(model_parameters)
            reconstructed_matrix = reconstruct_matrix(projection[0], eigenvectors, self.params[N_COMPONENTS],
                                                      mean=np.mean(input_data, axis=0))
            return reconstructed_matrix.reshape((self.dim[TIME_FRAMES],
                                                 self.dim[ATOMS],
                                                 self.dim[COORDINATES]))


class SubTrajectoryDecorator(DataTrajectory):
    def __init__(self,
                 data_trajectory: DataTrajectory,
                 quantity: int = 1,
                 time_window_size: int = None,
                 part_count: int = None,
                 **kwargs):
        self.data_trajectory: DataTrajectory = data_trajectory
        super().__init__(**kwargs)
        self._tmp_part_count = part_count
        self.quantity = quantity
        self.time_window_size = time_window_size
        self.rest = 0
        self.part_count: [None, int] = part_count
        self.__random_part_count = False
        self._check_init_params()

    def __setattr__(self, name, value):
        # Update the parameter in the decorator class
        if name in ["dim"]:
            super().__setattr__(name, self.data_trajectory.__dict__[name])
        else:
            super().__setattr__(name, value)

            # Update the parameter in the data_trajectory object
            if (hasattr(self, 'data_trajectory') and
                    name in self.data_trajectory.__dict__ and
                    name != "params"):
                setattr(self.data_trajectory, name, value)

    @property
    def max_components(self) -> int:
        return self.data_trajectory.max_components

    def _check_init_params(self):
        super()._check_init_params()
        # self.update_model_params(dim=self.data_trajectory.dim)
        # self.dim = self.data_trajectory.dim
        if self.quantity > 1:
            if self.dim[TIME_FRAMES] < self.quantity:
                raise InvalidSubsetTrajectory(f'The number of quantities `{self.quantity}` is invalid '
                                              f'for the trajectory with time steps `{self.dim[TIME_FRAMES]}`')

            if self.time_window_size is not None:
                warnings.warn(f'Quantity for the trajectory subset has a higher priority. '
                              f'Time window size: `{self.time_window_size}` is overwritten.')

            self.rest = self.dim[TIME_FRAMES] % self.quantity
            self.time_window_size = self.dim[TIME_FRAMES] // self.quantity
        else:
            if self.time_window_size is None:
                self.time_window_size = self.dim[TIME_FRAMES]
                warnings.warn(f'The subset of the trajectory is the same as the actual trajectory.')

            if self.dim[TIME_FRAMES] < self.time_window_size:
                raise InvalidSubsetTrajectory(f'The time window size `{self.time_window_size}` is invalid '
                                              f'for the trajectory with time steps `{self.dim[TIME_FRAMES]}`')
            self.rest = self.dim[TIME_FRAMES] % self.time_window_size
            self.quantity = self.dim[TIME_FRAMES] // self.time_window_size
        if self.part_count is not None and self.part_count < 0:
            self.__random_part_count = True
            self.part_count = random.randint(0, self.quantity - 1)

    def update_model_params(self, **params):
        r"""Update given model parameter if they are set to specific values"""
        for key, value in params.items():
            if not hasattr(self, key):
                setattr(self, key, value)  # set parameter for the first time.
            elif getattr(self, key) is None:
                setattr(self, key, value)  # update because this parameter is still None.
            elif value is not None:
                setattr(self, key, value)  # only overwrite if set to a specific value (None does not overwrite).

    def get_sub_results(self, model_parameters: dict) -> list:
        results = []
        tmp_count = self.part_count
        for count in range(self.quantity):
            self.part_count = count
            results.append(self.get_model_result(model_parameters, log=False))
        self.part_count = tmp_count
        return results

    def data_input(self, model_parameters: dict = None) -> np.ndarray:
        """
        Extracts the correct subset of the trajectory out of the whole input data
        @param model_parameters:
        @return:  data input subset
        """
        whole_input_data = self.data_trajectory.data_input(model_parameters)
        if self.part_count is None:
            return whole_input_data
        elif self.part_count == self.quantity - 1:
            return whole_input_data[-(self.time_window_size + self.rest):]
        else:
            return whole_input_data[
                   (self.part_count * self.time_window_size):(self.part_count + 1) * self.time_window_size]

    @contextmanager
    def use_full_input(self):
        original_value = self.part_count
        self.part_count = None  # Set the new value
        yield  # Execute code within the context (with)
        self.part_count = original_value  # Restore the original value

    def change_time_window_size(self, new_value):
        if self.dim[TIME_FRAMES] < new_value:
            raise InvalidSubsetTrajectory(f'The new time window size `{new_value}` is invalid '
                                          f'for the trajectory with time steps `{self.dim[TIME_FRAMES]}`')

        else:
            self.time_window_size = new_value
            self.rest = self.dim[TIME_FRAMES] % self.time_window_size
            self.quantity = self.dim[TIME_FRAMES] // self.time_window_size

            if self.__random_part_count:
                self.part_count = random.randint(0, self.quantity - 1)

        return self


class ProteinTopologyConverter(TrajectoryFile):
    def __init__(self, filename, topology_filename, goal_filename, folder_path='data/2f4k'):
        super().__init__(filename, extra_filename=topology_filename, folder_path=folder_path)
        self.goal_filename = goal_filename

    @property
    def goal_filepath(self):
        return str(self.root_path / self.goal_filename)

    def convert(self):
        import MDAnalysis
        print(f'Convert Topology {self.extra_filename} to {self.goal_filename}...')
        # noinspection PyProtectedMember
        if self.extra_filename.split('.')[-1].upper() in MDAnalysis._PARSERS:
            universe = MDAnalysis.Universe(self.extra_filepath)
            with MDAnalysis.Writer(self.goal_filepath) as writer:
                writer.write(universe)
        else:
            raise NotImplementedError(f'{self.extra_filename} cannot converted into {self.filename}')
        print('Convert successful.')
