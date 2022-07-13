from matplotlib.widgets import Slider
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D


class TrajectoryPlotter:
    def __init__(self, trajectory, interactive=True):
        self.data_trajectory = trajectory
        self.fig = None
        self.axes = None
        self.interactive = interactive
        if self.interactive:
            mpl.use('TkAgg')
        else:
            pass

    def original_data_with_timestep_slider(self, min_max=None):
        """
        Creates an interactive plot window, where the plotted data at a timestep can be chosen by a Slider.
        Used as in https://matplotlib.org/stable/gallery/widgets/slider_demo.html
        :param min_max: data range of the data
        """
        if not self.interactive:
            raise ValueError('Plotter has to be interactive to use this plot.')

        # if 'Axes3D' not in sys.modules:
        #    raise ModuleNotFoundError('Axes3D has to be imported from mpl_toolkits.mplot3d to use 3d plotting.')

        if min_max is None:  # min_max is a list of two elements
            min_max = [0, self.data_trajectory.traj.xyz.shape[0]]

        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111, projection='3d')

        plt.subplots_adjust(bottom=0.25)
        # noinspection PyTypeChecker
        ax_freq = plt.axes([0.25, 0.1, 0.65, 0.03])
        freq_slider = Slider(
            ax=ax_freq,
            label='Time Step',
            valmin=min_max[0],  # minimun value of range
            valmax=min_max[-1] - 1,  # maximum value of range
            valinit=0,
            valstep=1,  # step between values
            valfmt='%0.0f'
        )
        freq_slider.on_changed(self.update_plot)
        plt.show()

    def plot_original_data_at(self, timeframe):
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.update_plot(timeframe)

    def update_plot(self, timeframe):
        if 0 <= timeframe <= self.data_trajectory.traj.n_frames:
            timeframe = int(timeframe)
            x_coordinates = self.data_trajectory.traj.xyz[timeframe][:, 0]
            y_coordinates = self.data_trajectory.traj.xyz[timeframe][:, 1]
            z_coordinates = self.data_trajectory.traj.xyz[timeframe][:, 2]
            self.axes.cla()
            self.axes.scatter(x_coordinates, y_coordinates, z_coordinates, c='r', marker='.')
            max_d = self.data_trajectory.traj.xyz.max()
            self.axes.set_xlim([0, self.data_trajectory.coordinate_maxs['x']])
            self.axes.set_ylim([0, self.data_trajectory.coordinate_maxs['y']])
            self.axes.set_zlim([0, self.data_trajectory.coordinate_maxs['z']])
            self.axes.set_xlabel('x-Axis')
            self.axes.set_ylabel('y-Axis')
            self.axes.set_zlabel('z-Axis')
            plt.show()
        else:
            raise IndexError('Timestep does not exist')

    def plot_models(self, model1, model2, reduced1, reduced2):
        self.fig, self.axes = plt.subplots(3, 2)
        self.plot_on_axis(self.axes[0][0], xy_data=reduced1[0], title=str(model1),
                          x_label='1st component', y_label='2nd component')
        self.axes[0][0].cla()
        self.axes[0][0].set_title(str(model1))
        # print(reduced1)
        self.axes[0][0].scatter(reduced1[0][:, 0], reduced1[0][:, 1], c='r', marker='.')
        # self.ax[0][0].contourf(reduced1[0][:, 0], reduced1[0][:, 1], cmap=plt.cm.hot, c='r', marker='.')
        # self.ax[0][0].scatter(reduced1[1][:, 0], reduced1[1][:, 1], c='b', marker='.')
        # self.ax[0][0].scatter(reduced1[2][:, 0], reduced1[2][:, 1], c='g', marker='.')
        try:
            print(reduced1[0].shape)
            print(model1.eigenvectors_, model1.eigenvalues_)
            self.axes[0][0].arrow(0, 0, 100 * model1.eigenvectors_[0, 0], 100 * model1.eigenvectors_[1, 0],
                                  color='orange')
        except AttributeError as e:
            print('%s: %s'.format(e, model1))
        # print(reduced2)
        self.axes[1][0].cla()
        self.axes[1][0].plot(reduced1[0][:, 0], c='r')
        # self.ax[1][0].plot(reduced1[1][:, 0], c='b')
        # self.ax[1][0].plot(reduced1[2][:, 0], c='g')
        self.axes[2][0].cla()
        self.axes[2][0].plot(reduced1[0][:, 1], c='r')
        # self.ax[2][0].plot(reduced1[1][:, 1], c='b')
        # self.ax[2][0].plot(reduced1[2][:, 1], c='g')
        self.axes[0][1].cla()
        self.axes[0][1].set_title(str(model2))
        self.axes[0][1].scatter(reduced2[0][:, 0], reduced2[0][:, 1], c='r', marker='.')
        # self.ax[0][1].scatter(reduced2[1][:, 0], reduced2[1][:, 1], c='b', marker='.')
        # self.ax[0][1].scatter(reduced2[2][:, 0], reduced2[2][:, 1], c='g', marker='.')
        try:
            print(reduced2[0].shape)
            print(model2.eigenvectors_, model2.eigenvalues_)
            self.axes[0][1].arrow(0, 0, 100 * model2.eigenvectors_[0, 0], 100 * model2.eigenvectors_[1, 0],
                                  color='orange')
        except AttributeError as e:
            print('{}: {}'.format(e, model2))
        self.axes[1][1].cla()
        self.axes[1][1].plot(reduced2[0][:, 0], c='r')
        # self.ax[1][1].plot(reduced2[1][:, 0], c='b')
        # self.ax[1][1].plot(reduced2[2][:, 0], c='g')
        self.axes[2][1].cla()
        self.axes[2][1].plot(reduced2[0][:, 1], c='r')
        # self.ax[2][1].plot(reduced2[1][:, 1], c='b')
        # self.ax[2][1].plot(reduced2[2][:, 1], c='g')
        plt.show()

    def plot_one_model(self, model, reduced_data):
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)
        self.axes.cla()
        self.axes.scatter(reduced_data[0][:, 0], reduced_data[0][:, 1], c='r', marker='.')
        self.axes.scatter(reduced_data[1][:, 0], reduced_data[1][:, 1], c='b', marker='.')
        self.axes.scatter(reduced_data[2][:, 0], reduced_data[2][:, 1], c='g', marker='.')

    def plot_on_axis(self, param, xy_data, title, x_label, y_label):
        pass
