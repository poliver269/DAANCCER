from pathlib import Path
import mdtraj as md
import numpy as np
from scipy.optimize import curve_fit
from skimage.measure import block_reduce

from utils.algorithms.pca import MyPCA, TruncatedPCA
from utils.algorithms.tica import MyTICA, TruncatedTICA
from utils.param_key import *
from utils.math import basis_transform, explained_variance, \
    matrix_diagonals_calculation, diagonal_indices, gaussian_2d, expand_diagonals_to_matrix, \
    calculate_pearson_correlations
from plotter import TrajectoryPlotter


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
            PLOT_TYPE: params.get(PLOT_TYPE, 'color_map'),  # 'color_map', 'heat_map'
            PLOT_TICS: params.get(PLOT_TICS, True),
            CARBON_ATOMS_ONLY: params.get(CARBON_ATOMS_ONLY, True),
            INTERACTIVE: params.get(INTERACTIVE, True),
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            LAG_TIME: params.get(LAG_TIME, 10),
            TRUNCATION_VALUE: params.get(TRUNCATION_VALUE, 0),
            BASIS_TRANSFORMATION: params.get(BASIS_TRANSFORMATION, False),
            RANDOM_SEED: params.get(RANDOM_SEED, 30)
        }

        self.x_coordinates = self.get_coordinates([0])
        self.y_coordinates = self.get_coordinates([1])
        self.z_coordinates = self.get_coordinates([2])
        self.coordinate_mins = {X: self.x_coordinates.min(), Y: self.y_coordinates.min(),
                                Z: self.z_coordinates.min()}
        self.coordinate_maxs = {X: self.x_coordinates.max(), Y: self.y_coordinates.max(),
                                Z: self.z_coordinates.max()}

    @property
    def coordinates(self):
        if self.params[BASIS_TRANSFORMATION]:
            return self.basis_transformed_coordinates
        else:
            return self.traj.xyz

    @property
    def flattened_coordinates(self):
        if self.params[CARBON_ATOMS_ONLY]:
            return self.alpha_coordinates.reshape(self.dim[TIME_FRAMES],
                                                  len(self.carbon_alpha_indexes) * self.dim[COORDINATES])
        else:
            return self.coordinates.reshape(self.dim[TIME_FRAMES], self.dim[ATOMS] * self.dim[COORDINATES])

    @property
    def carbon_alpha_indexes(self):
        return [a.index for a in self.traj.topology.atoms if a.name == 'CA']

    @property
    def alpha_coordinates(self):
        return self.get_atoms(self.carbon_alpha_indexes)

    @property
    def basis_transformed_coordinates(self):
        np.random.seed(self.params[RANDOM_SEED])
        return basis_transform(self.traj.xyz, self.dim[COORDINATES])

    def get_time_frames(self, element_list):
        return self.coordinates[element_list, :, :]

    def get_atoms(self, element_list):
        return self.coordinates[:, element_list, :]

    def get_coordinates(self, element_list):
        return self.coordinates[:, :, element_list]

    def get_model_and_projection(self, model_name, inp=None):
        import pyemma.coordinates as coor
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
            model, projection = self.get_model_and_projection(model_name,
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
        mode = 'gauss_distribution_on_data'
        if mode == 'flattened_coordinates':
            flattened_covariance = np.cov(self.flattened_coordinates)
            corr_coefficient = np.corrcoef(self.flattened_coordinates)
        elif mode == 'coordinates_mean_first':
            coordinates_mean = np.mean(self.coordinates, axis=2)
            coefficient_mean = np.corrcoef(coordinates_mean.T)
        elif mode == 'alpha_carbon_mean_first':
            alpha_coordinates_mean = np.mean(self.alpha_coordinates, axis=2)
            alpha_coefficient_mean = np.corrcoef(alpha_coordinates_mean.T)
        elif mode == 'alpha_carbon_coefficient_first':
            alpha_coefficient = np.corrcoef(self.flattened_coordinates.T)
            alpha_coefficient_mean = block_reduce(alpha_coefficient, (3, 3), np.diag)
        elif mode == 'alpha_carbon_optimized_gauss_kernel':
            pass  # params, cov = curve_fit()
        elif mode == 'gauss_distribution_on_data':
            alpha_coordinates_mean = np.mean(self.alpha_coordinates, axis=2)
            alpha_coefficient_mean = np.corrcoef(alpha_coordinates_mean.T)
            ydata = matrix_diagonals_calculation(alpha_coefficient_mean, np.mean)
            xdata = diagonal_indices(alpha_coefficient_mean)
            parameters, cov = curve_fit(gaussian_2d, xdata, ydata)
            fit_y = gaussian_2d(xdata, parameters[0], parameters[1])
            d_matrix = expand_diagonals_to_matrix(alpha_coefficient_mean, fit_y)
            TrajectoryPlotter(self).plot_gauss2d(fit_y, xdata, ydata, mean_data=np.full(xdata.shape, ydata.mean()))
        # d_matrix = diagonal_gauss_matrix_kernel(alpha_coefficient_mean.shape[0], sig=0.2)
        # d_matrix = exponentiated_quadratic(alpha_coefficient_mean, s=2)
        weighted_alpha_coeff_matrix = alpha_coefficient_mean - d_matrix
        # sympy_matrix_alpha_carbon = Matrix(alpha_coefficient_mean)
        # data_nullspace = sympy_matrix_alpha_carbon * sympy_matrix_alpha_carbon.nullspace()
        TrajectoryPlotter(self).matrix_plot(weighted_alpha_coeff_matrix,
                                            title_prefix='Alpha Carbon Atoms Pearson Coefficient calculation first, mean second',
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
