from pathlib import Path
import mdtraj as md
import numpy as np

from utils.algorithms.pca import MyPCA, TruncatedPCA
from utils.algorithms.tica import MyTICA, TruncatedTICA
from utils.math import basis_transform
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
            self.dim = {'time_frames': self.traj.xyz.shape[0],
                        'atoms': self.traj.xyz.shape[1],
                        'coordinates': self.traj.xyz.shape[2]}
            self.phi = md.compute_phi(self.traj)
            self.psi = md.compute_psi(self.traj)
        except IOError:
            raise FileNotFoundError("Cannot load {} or {}.".format(self.filepath, self.topology_path))
        else:
            print("{} successfully loaded.".format(self.traj))

        if params is None:
            params = {}
        self.params = {
            'plot_type': params.get('plot_type', 'color_map'),  # 'color_map', 'heat_map'
            'plot_tics': params.get('plot_tics', True),
            'carbon_atoms_only': params.get('carbon_atoms_only', True),
            'interactive': params.get('interactive', True),
            'n_components': params.get('n_components', 2),
            'lag_time': params.get('lag_time', 10),
            'truncation_value': params.get('truncation_value', 0),
            'basis_transformation': params.get('basis_transformation', False),
            'random_seed': params.get('random_seed', 30)
        }

        x_coordinates = self.get_coordinates([0])
        y_coordinates = self.get_coordinates([1])
        z_coordinates = self.get_coordinates([2])
        self.coordinate_mins = {'x': x_coordinates.min(), 'y': y_coordinates.min(), 'z': z_coordinates.min()}
        self.coordinate_maxs = {'x': x_coordinates.max(), 'y': y_coordinates.max(), 'z': z_coordinates.max()}

    @property
    def coordinates(self):
        if self.params['basis_transformation']:
            return self.basis_transformed_coordinates
        else:
            return self.traj.xyz

    @property
    def flattened_coordinates(self):
        if self.params['carbon_atoms_only']:
            return self.alpha_coordinates.reshape(self.dim['time_frames'],
                                                  len(self.carbon_alpha_indexes) * self.dim['coordinates'])
        else:
            return self.coordinates.reshape(self.dim['time_frames'], self.dim['atoms'] * self.dim['coordinates'])

    @property
    def carbon_alpha_indexes(self):
        return [a.index for a in self.traj.topology.atoms if a.name == 'CA']

    @property
    def alpha_coordinates(self):
        return self.get_atoms(self.carbon_alpha_indexes)

    @property
    def basis_transformed_coordinates(self):
        np.random.seed(self.params['random_seed'])
        return basis_transform(self.traj.xyz, self.dim['coordinates'])

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
            pca = coor.pca(data=inp, dim=self.params['n_components'])
            return pca, pca.get_output()
        elif model_name == 'mypca':
            pca = MyPCA()
            return pca, [pca.fit_transform(inp, n_components=self.params['n_components'])]
        elif model_name == 'trunc_pca':
            pca = TruncatedPCA(self.params['truncation_value'])
            return pca, [pca.fit_transform(inp, n_components=self.params['n_components'])]
        elif model_name == 'tica':
            tica = coor.tica(data=inp, lag=self.params['lag_time'], dim=self.params['n_components'])
            return tica, tica.get_output()
        elif model_name == 'mytica':
            tica = MyTICA(lag_time=self.params['lag_time'])
            return tica, [tica.fit_transform(inp, n_components=self.params['n_components'])]
        elif model_name == 'trunc_tica':
            tica = TruncatedTICA(lag_time=self.params['lag_time'], trunc_value=self.params['truncation_value'])
            return tica, [tica.fit_transform(inp, n_components=self.params['n_components'])]
        else:
            print(f'Model with name \"{model_name}\" does not exists.')
            return None, None

    def compare_angles(self, model_names):
        model_results = []
        for model_name in model_names:
            phi_model, phi_projection = self.get_model_and_projection(model_name, self.phi[1])
            psi_model, psi_projection = self.get_model_and_projection(model_name, self.psi[1])
            model_results = model_results + [{'model': phi_model, 'projection': phi_projection,
                                              'title_prefix': f'Phi\n{model_name}'},
                                             {'model': psi_model, 'projection': psi_projection,
                                              'title_prefix': f'Psi\n{model_name}'}]
        self.compare_with_plot(model_results)

    def compare_with_msmbuilder(self, model_name1, model_name2):
        # noinspection PyUnresolvedReferences
        from msmbuilder.decomposition import tICA, PCA  # works only on python 3.5 and smaller
        reshaped_traj = np.reshape(self.traj.xyz, (self.dim['coordinates'], self.dim['time_frames'], self.dim['atoms']))
        models = {'tica': tICA(n_components=self.params['n_components']),
                  'pca': PCA(n_components=self.params['n_components'])}

        model1 = models[model_name1]  # --> (n_components, time_frames)
        # reshaped_traj = self.converted  # Does not Work
        model1.fit(reshaped_traj)
        reduced_traj1 = model1.transform(reshaped_traj)

        model2 = models[model_name2]
        model2.fit(self.flattened_coordinates)
        reduced_traj2 = model2.transform(self.flattened_coordinates)

        self.compare_with_plot([{'model': model1, 'projection': reduced_traj1},
                                {'model': model2, 'projection': reduced_traj2}])

    def compare_with_pyemma(self, model_names):
        model_results = []
        for model_name in model_names:
            model, projection = self.get_model_and_projection(model_name)
            model_results.append({'model': model, 'projection': projection})
        self.compare_with_plot(model_results)

    def compare_with_carbon_alpha_and_all_atoms(self, model_name):
        model_results = []
        model1, projection1 = self.get_model_and_projection(model_name)
        model_results.append({'model': model1, 'projection': projection1,
                              'title_prefix': f'Only Carbon-Alpha Atoms: {self.params["carbon_atoms_only"]}\n'})
        self.params['carbon_atoms_only'] = not self.params['carbon_atoms_only']
        model2, projection2 = self.get_model_and_projection(model_name)
        model_results.append({'model': model2, 'projection': projection2,
                              'title_prefix': f'Only Carbon-Alpha Atoms: {self.params["carbon_atoms_only"]}\n'})
        self.params['carbon_atoms_only'] = not self.params['carbon_atoms_only']
        self.compare_with_plot(model_results)

    def compare_with_basis_transformation(self, model_names):
        model_results = []
        for model_name in model_names:
            model, projection = self.get_model_and_projection(model_name)
            model_results.append({'model': model, 'projection': projection,
                                  'title_prefix': f'Basis transformation {self.params["basis_transformation"]}\n'})
            self.params['basis_transformation'] = not self.params['basis_transformation']
            model, projection = self.get_model_and_projection(model_name)
            model_results.append({'model': model, 'projection': projection,
                                  'title_prefix': f'Basis transformation {self.params["basis_transformation"]}\n'})
            self.params['basis_transformation'] = not self.params['basis_transformation']
        self.compare_with_plot(model_results)

    def compare_with_plot(self, model_projection_list):
        TrajectoryPlotter(self).plot_models(
            model_projection_list,
            data_elements=[0],  # [0, 1, 2]
            plot_type=self.params['plot_type'],
            plot_tics=self.params['plot_tics'],
            components=self.params['n_components']
        )


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
