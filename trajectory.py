from pathlib import Path
import mdtraj as md
import numpy as np
import tltsne
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
    def __init__(self, filename, topology_filename, folder_path='data/2f4k'):
        super().__init__(filename, topology_filename, folder_path)
        try:
            print("Loading trajectory...")
            self.traj = md.load(self.filepath, top=self.topology_path)
            self.dim = {'time_frames': self.traj.xyz.shape[0],
                        'atoms': self.traj.xyz.shape[1],
                        'coordinates': self.traj.xyz.shape[2]}
        except IOError:
            raise FileNotFoundError("Cannot load {} or {}.".format(self.filepath, self.topology_path))
        else:
            print("{} successfully loaded.".format(self.traj))

        x_coordinates = self.traj.xyz[:][:, 0]
        y_coordinates = self.traj.xyz[:][:, 1]
        z_coordinates = self.traj.xyz[:][:, 2]
        self.coordinate_mins = {'x': x_coordinates.min(), 'y': y_coordinates.min(), 'z': z_coordinates.min()}
        self.coordinate_maxs = {'x': x_coordinates.max(), 'y': y_coordinates.max(), 'z': z_coordinates.max()}

    @property
    def converted(self):
        return self.traj.xyz.reshape(self.dim['time_frames'], self.dim['atoms'] * self.dim['coordinates'])

    def compare_with_msmbuilder(self, model_name1, model_name2):
        from msmbuilder.decomposition import tICA, PCA
        components = 2
        reshaped_traj = np.reshape(self.traj.xyz, (self.dim['coordinates'], self.dim['time_frames'], self.dim['atoms']))
        models = {'tica': tICA(n_components=components), 'pca': PCA(n_components=components)}

        model1 = models[model_name1]  # --> (n_components, time_frames)
        # reshaped_traj = self.converted  # Does not Work
        model1.fit(reshaped_traj)
        reduced_traj1 = model1.transform(reshaped_traj)

        model2 = models[model_name2]
        model2.fit(self.converted)
        reduced_traj2 = model2.transform(self.converted)

        print(model1, model2, sep='\n')
        self.compare_with_plot(model1, model2, reduced_traj1, reduced_traj2)

    def compare_with_pyemma(self, model_name1, model_name2):
        import pyemma.coordinates as coor
        components = 2
        feat = coor.featurizer(self.topology_path)
        inp = coor.source(self.filepath, features=feat)
        models = {'tica': coor.tica(inp, dim=components),
                  'pca': coor.pca(inp, dim=components)}
        model1 = models[model_name1]
        model2 = models[model_name2]
        print(model1, model2, sep='\n')
        self.compare_with_plot(model1, model2, model1.get_output(), model2.get_output())

    def compare_with_plot(self, model1, model2, proj1, proj2):
        TrajectoryPlotter(self).plot_models(model1, model2, proj1, proj2,
                                            data_elements=[0],  # [0, 1, 2]
                                            plot_type='color_map',  # 'heat_map', 'color_map'
                                            plot_tics=False)  # True, False


class TopologyConverter(TrajectoryFile):
    def __init__(self, filename, topology_filename, goal_filename, folder_path='data/2f4k'):
        super().__init__(filename, topology_filename, folder_path)
        self.goal_filename = goal_filename

    @property
    def goal_filepath(self):
        return str(self.root_path / self.goal_filename)

    def convert(self):
        import MDAnalysis as mda
        universe = mda.Universe(self.topology_path)
        with mda.Writer(self.goal_filepath) as writer:
            writer.write(universe)
