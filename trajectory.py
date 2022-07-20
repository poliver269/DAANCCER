from pathlib import Path
import mdtraj as md
import numpy as np

from plotter import TrajectoryPlotter


class DataTrajectory:
    def __init__(self, filename, topology_filename, folder_path='data/2f4k'):
        self.root_path = Path(folder_path)
        self.filename = filename
        self.topology_filename = topology_filename
        # print(self.root_path / self.filename)
        self.traj = md.load(str(self.root_path / self.filename), top=str(self.root_path / self.topology_filename))
        self.dim = {'time_steps': self.traj.xyz.shape[0], 'atoms': self.traj.xyz.shape[1],
                    'coordinates': self.traj.xyz.shape[2]}

        x_coordinates = self.traj.xyz[:][:, 0]
        y_coordinates = self.traj.xyz[:][:, 1]
        z_coordinates = self.traj.xyz[:][:, 2]
        self.coordinate_mins = {'x': x_coordinates.min(), 'y': y_coordinates.min(), 'z': z_coordinates.min()}
        self.coordinate_maxs = {'x': x_coordinates.max(), 'y': y_coordinates.max(), 'z': z_coordinates.max()}

    def compare_with_msmbuilder(self, model_name1, model_name2):
        from msmbuilder.decomposition import tICA, PCA
        components = 2
        reshaped_traj = np.reshape(self.traj.xyz, (self.dim['coordinates'], self.dim['time_steps'], self.dim['atoms']))
        # reshaped_traj = self.traj.xyz
        models = {'tica': tICA(n_components=components), 'pca': PCA(n_components=components)}
        model1 = models[model_name1]  # --> (n_components, time_steps)
        # model1.fit(np.reshape(self.traj.xyz, (self.dim['coordinates'], self.dim['atoms'], self.dim['time_steps'])))
        model1.fit(reshaped_traj)
        reduced_traj1 = model1.transform(reshaped_traj)

        model2 = models[model_name2]
        model2.fit(reshaped_traj)
        reduced_traj2 = model2.transform(reshaped_traj)

        print(model1, model2, sep='\n')
        TrajectoryPlotter(self).plot_models(model1, model2, reduced_traj1, reduced_traj2, [0], plot_type=True)

    def compare_with_pyemma(self, model_name1, model_name2):
        import pyemma.coordinates as coor
        components = 2
        feat = coor.featurizer(str(self.root_path / self.topology_filename))
        inp = coor.source(str(self.root_path / self.filename), features=feat)
        reshaped_traj = np.reshape(self.traj.xyz, (self.dim['coordinates'], self.dim['time_steps'], self.dim['atoms']))
        models = {'tica': coor.tica(inp, dim=components),
                  'pca': coor.pca(inp, dim=components)}
        model1 = models[model_name1]
        model2 = models[model_name2]
        print(model1, model2, sep='\n')
        TrajectoryPlotter(self).plot_models(model1, model2, model1.get_output(), model2.get_output(),
                                            data_elements=[0],  # [0, 1, 2]
                                            plot_type='color_map',  # 'heat_map', 'color_map'
                                            plot_tics=True)  # 'True'
