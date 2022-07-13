from pathlib import Path
import mdtraj as md
import numpy as np
from msmbuilder.decomposition import tICA, PCA

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

    def compare(self, model_name1, model_name2):
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
        TrajectoryPlotter(self).plot_models(model1, model2, reduced_traj1, reduced_traj2)

