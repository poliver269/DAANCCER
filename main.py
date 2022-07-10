import numpy as np
import mdtraj as md
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from msmbuilder.decomposition import tICA, PCA

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

mpl.use('TkAgg')


class DataTrajectory:
    def __init__(self, filename, topology_filename, folder_path='data/2f4k'):
        self.root_path = Path(folder_path)
        self.filename = filename
        self.topology_filename = topology_filename
        # print(self.root_path / self.filename)
        self.traj = md.load(str(self.root_path / self.filename), top=str(self.root_path / self.topology_filename))
        self.dim = {'time_steps': self.traj.xyz.shape[0], 'atoms': self.traj.xyz.shape[1],
                    'coordinates': self.traj.xyz.shape[2]}
        self.fig, self.ax = None, None

    def plot_original_data_with_timestep_slider(self, min_max=None):
        if min_max is None:  # min_max is a list of two elements
            min_max = [0, self.traj.xyz.shape[0]]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        # noinspection PyTypeChecker
        ax_freq = plt.axes([0.25, 0.1, 0.65, 0.03])
        freq_slider = Slider(
            ax=ax_freq,
            label='Time Steps',
            valmin=min_max[0],  # minimun value of range
            valmax=min_max[-1] - 1,  # maximum value of range
            valinit=0,
            valstep=1.0  # step between values
        )
        freq_slider.on_changed(self.update_plot)
        plt.show()

    def plot_original_data_at(self, timeframe):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.update_plot(timeframe)

    def update_plot(self, timeframe):
        if 0 <= timeframe <= self.traj.n_frames:
            timeframe = int(timeframe)
            x_coordinates = self.traj.xyz[timeframe][:, 0]
            y_coordinates = self.traj.xyz[timeframe][:, 1]
            z_coordinates = self.traj.xyz[timeframe][:, 2]
            self.ax.cla()
            self.ax.scatter(x_coordinates, y_coordinates, z_coordinates, c='r', marker='.')
            self.ax.set_xlabel('x-Axis')
            self.ax.set_ylabel('y-Axis')
            self.ax.set_zlabel('z-Axis')
            plt.show()
        else:
            raise IndexError('Timestep does not exist')

    def plot_models(self, model1, model2, reduced1, reduced2):
        self.fig, self.ax = plt.subplots(1, 2)
        # self.ax = self.fig.add_subplot(111)
        self.ax[0].cla()
        self.ax[0].set_title('Model 1')
        # print(reduced1)
        self.ax[0].scatter(reduced1[0][:, 0], reduced1[0][:, 1], c='r', marker='.')
        self.ax[0].scatter(reduced1[1][:, 0], reduced1[1][:, 1], c='b', marker='.')
        self.ax[0].scatter(reduced1[2][:, 0], reduced1[2][:, 1], c='g', marker='.')
        print(model1.eigenvectors_, model1.eigenvalues_, reduced1[0].shape)
        try:
            self.ax[0].arrow(0, 0, 100 * model1.eigenvectors_[0, 0], 100 * model1.eigenvectors_[1, 0],
                             color='orange')
        except AttributeError as e:
            print('%s: %s'.format(e, model1))
        # print(reduced2)
        self.ax[1].cla()
        self.ax[1].set_title('Model 2')
        self.ax[1].scatter(reduced2[0][:, 0], reduced2[0][:, 1], c='r', marker='.')
        self.ax[1].scatter(reduced2[1][:, 0], reduced2[1][:, 1], c='b', marker='.')
        self.ax[1].scatter(reduced2[2][:, 0], reduced2[2][:, 1], c='g', marker='.')
        try:
            self.ax[0].arrow(0, 0, 100 * model2.eigenvectors_[0, 0], 100 * model2.eigenvectors_[1, 0],
                             color='orange')
        except AttributeError as e:
            print('{}: {}'.format(e, model2))
        # self.ax.set_xlabel('x-Axis')
        # self.ax.set_ylabel('y-Axis')
        plt.show()

    def plot_one_model(self, model, reduced_data):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.cla()
        self.ax.scatter(reduced_data[0][:, 0], reduced_data[0][:, 1], c='r', marker='.')
        self.ax.scatter(reduced_data[1][:, 0], reduced_data[1][:, 1], c='b', marker='.')
        self.ax.scatter(reduced_data[2][:, 0], reduced_data[2][:, 1], c='g', marker='.')

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
        self.plot_models(model1, model2, reduced_traj1, reduced_traj2)


def main():
    print('Starting time: {}'.format(datetime.now()))
    # TODO: Argsparser for options
    plot_option = 'only_one'
    tr = DataTrajectory('tr3_unfolded.xtc', topology_filename='2f4k.gro', folder_path='data/2f4k')
    if plot_option == 'with_slider':
        tr.plot_original_data_with_timestep_slider(min_max=None)  # [0, 10]
    else:
        tr.compare('tica', 'pca')
    print('Finishing time: {}'.format(datetime.now()))


if __name__ == '__main__':
    main()
