from pathlib import Path

import mdtraj as md
import numpy as np
from scipy.optimize import curve_fit

from plotter import TrajectoryPlotter
from utils.algorithms.pca import MyPCA, TruncatedPCA
from utils.algorithms.tica import MyTICA, TruncatedTICA
from utils.math import basis_transform, explained_variance, matrix_diagonals_calculation, diagonal_indices, \
    gaussian_2d, expand_diagonals_to_matrix, calculate_pearson_correlations
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
            print("Loading trajectory...")
            self.traj = md.load(self.filepath, top=self.topology_path)
            self.dim = {TIME_FRAMES: self.traj.xyz.shape[0],
                        ATOMS: self.traj.xyz.shape[1],
                        COORDINATES: self.traj.xyz.shape[2]}
            self.phi = md.compute_phi(self.traj)
            self.psi = md.compute_psi(self.traj)
        except IOError:
            raise FileNotFoundError("Cannot load {} or {}.".format(self.filepath, self.topology_path))
        else:
            print("{} successfully loaded.".format(self.traj))

        if params is None:
            params = {}
        self.params = {
            PLOT_TYPE: params.get(PLOT_TYPE, COLOR_MAP),  # 'color_map', 'heat_map'
            PLOT_TICS: params.get(PLOT_TICS, True),
            CARBON_ATOMS_ONLY: params.get(CARBON_ATOMS_ONLY, True),
            INTERACTIVE: params.get(INTERACTIVE, True),
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            LAG_TIME: params.get(LAG_TIME, 10),
            TRUNCATION_VALUE: params.get(TRUNCATION_VALUE, 0),
            BASIS_TRANSFORMATION: params.get(BASIS_TRANSFORMATION, False),
            RANDOM_SEED: params.get(RANDOM_SEED, 30),
            USE_ANGLES: params.get(USE_ANGLES, True)
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
        import pyemma.coordinates as coor  # Todo: import globally and delete msmbuilder
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
        elif model_name == 'tica':
            tica = coor.tica(data=inp, lag=self.params[LAG_TIME], dim=self.params[N_COMPONENTS])
            return tica, tica.get_output()
        elif model_name == 'mytica':
            tica = MyTICA(lag_time=self.params[LAG_TIME])
            return tica, [tica.fit_transform(inp, n_components=self.params[N_COMPONENTS])]
        elif model_name == 'trunc_tica':
            tica = TruncatedTICA(lag_time=self.params[LAG_TIME], trunc_value=self.params[TRUNCATION_VALUE])
            return tica, [tica.fit_transform(inp, n_components=self.params[N_COMPONENTS])]
        else:
            raise ValueError(f'Model with name \"{model_name}\" does not exists.')

    def compare_angles(self, model_names):
        model_results = []
        for model_name in model_names:
            model, projection = self.get_model_and_projection(model_name,  # Todo: refactor to 'use_angles'
                                                              np.concatenate([self.phi[1], self.psi[1]], axis=1))
            model_results = model_results + [{MODEL: model, PROJECTION: projection,
                                              TITLE_PREFIX: f'Angles\n'}]
        self.compare_with_plot(model_results)

    def compare_with_msmbuilder(self, model_name1, model_name2):
        # noinspection PyUnresolvedReferences
        from msmbuilder.decomposition import tICA, PCA  # works only on python 3.5 and smaller
        reshaped_traj = np.reshape(self.traj.xyz, (self.dim[COORDINATES], self.dim[TIME_FRAMES], self.dim[ATOMS]))
        models = {'tica': tICA(n_components=self.params[N_COMPONENTS]),
                  'pca': PCA(n_components=self.params[N_COMPONENTS])}

        model1 = models[model_name1]  # --> (n_components, time_frames)
        # reshaped_traj = self.converted  # Does not Work
        model1.fit(reshaped_traj)
        reduced_traj1 = model1.transform(reshaped_traj)

        model2 = models[model_name2]
        model2.fit(self.flattened_coordinates)
        reduced_traj2 = model2.transform(self.flattened_coordinates)

        self.compare_with_plot([{MODEL: model1, PROJECTION: reduced_traj1},
                                {MODEL: model2, PROJECTION: reduced_traj2}])

    def compare_with_pyemma(self, model_names):
        model_results = []
        for model_name in model_names:
            model, projection = self.get_model_and_projection(model_name)
            ex_var = explained_variance(model.eigenvalues, self.params[N_COMPONENTS])
            model_results.append({MODEL: model, PROJECTION: projection,
                                  TITLE_PREFIX: f'explained var: {ex_var}\n'})
        self.compare_with_plot(model_results)

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
        mode = 'mean_first'
        if mode == 'coordinates_mean_first':
            coordinates_mean = np.mean(self.atom_coordinates, axis=2)
            coefficient_mean = np.corrcoef(coordinates_mean.T)
        elif mode == 'mean_first':
            if self.params[USE_ANGLES]:
                mean_matrix = np.mean(np.array([self.phi[DIHEDRAL_ANGEL_VALUES], self.psi[DIHEDRAL_ANGEL_VALUES]]),
                                      axis=0)
            else:
                mean_matrix = np.mean(self.alpha_carbon_coordinates, axis=2)
            coefficient_mean = np.corrcoef(mean_matrix.T)
        elif mode == 'coefficient_first':
            if self.params[USE_ANGLES]:
                input_list = [self.phi[DIHEDRAL_ANGEL_VALUES], self.psi[DIHEDRAL_ANGEL_VALUES]]
            else:
                input_list = [self.filter_coordinates_by_coordinates(c, ac_only=self.params[CARBON_ATOMS_ONLY])
                              for c in range(len(self.dim))]
            coefficient_mean = calculate_pearson_correlations(input_list, np.mean)
        else:
            raise ValueError('Invalid mode string was given')

        ydata = matrix_diagonals_calculation(coefficient_mean, np.mean)
        xdata = diagonal_indices(coefficient_mean)
        parameters, cov = curve_fit(gaussian_2d, xdata, ydata)
        fit_y = gaussian_2d(xdata, parameters[0], parameters[1])
        d_matrix = expand_diagonals_to_matrix(coefficient_mean, fit_y)

        weighted_alpha_coeff_matrix = coefficient_mean - d_matrix

        mode2 = 'plot_2d_gauss'
        if mode2 == 'plot_2d_gauss':
            TrajectoryPlotter(self).plot_gauss2d(fit_y, xdata, ydata, title_prefix='Angles fitted after Mean 1st',
                                                 mean_data=np.full(xdata.shape, ydata.mean()))
        else:
            TrajectoryPlotter(self).matrix_plot(weighted_alpha_coeff_matrix,
                                                title_prefix='Angles. Pearson Coefficient. Coefficient 1st, Mean 2nd',
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
        universe = MDAnalysis.Universe(self.topology_path)
        with MDAnalysis.Writer(self.goal_filepath) as writer:
            writer.write(universe)
