import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # This line is needed for a 3d plot


class TrajectoryPlotter:
    def __init__(self, trajectory, interactive=True):
        self.data_trajectory = trajectory
        self.fig = None
        self.axes = None
        self.interactive = interactive
        self.colors = mcolors.TABLEAU_COLORS
        if self.interactive:
            mpl.use('TkAgg')
        else:
            pass

    def original_data_with_timestep_slider(self, min_max=None):
        """
        Creates an interactive plot window, where the plotted data at a timestep can be chosen by a Slider.
        Used as in https://matplotlib.org/stable/gallery/widgets/slider_demo.html
        :param min_max: data range of the data with a min and a max value
        """
        if not self.interactive:
            raise ValueError('Plotter has to be interactive to use this plot.')

        if min_max is None:  # min_max is a list of two elements
            min_max = [0, self.data_trajectory.dim['time_frames']]

        self.fig = plt.figure()
        # self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes = Axes3D(self.fig)

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
        freq_slider.on_changed(self.update_on_slider_change)
        plt.show()

    def plot_original_data_at(self, timeframe):
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.update_on_slider_change(timeframe)

    def update_on_slider_change(self, timeframe):
        if 0 <= timeframe <= self.data_trajectory.traj.n_frames:
            timeframe = int(timeframe)
            x_coordinates = self.data_trajectory.traj.xyz[timeframe][:, 0]
            y_coordinates = self.data_trajectory.traj.xyz[timeframe][:, 1]
            z_coordinates = self.data_trajectory.traj.xyz[timeframe][:, 2]
            self.axes.cla()
            self.axes.scatter(x_coordinates, y_coordinates, z_coordinates, c='r', marker='.')
            self.axes.set_xlim([self.data_trajectory.coordinate_mins['x'], self.data_trajectory.coordinate_maxs['x']])
            self.axes.set_ylim([self.data_trajectory.coordinate_mins['y'], self.data_trajectory.coordinate_maxs['y']])
            self.axes.set_zlim([self.data_trajectory.coordinate_mins['z'], self.data_trajectory.coordinate_maxs['z']])
            self.axes.set_xlabel('x-Axis')
            self.axes.set_ylabel('y-Axis')
            self.axes.set_zlabel('z-Axis')
            plt.show()
        else:
            raise IndexError('Timestep does not exist')

    def plot_models(self, model1, model2, reduced1, reduced2, data_elements, title_prefix1='', title_prefix2='',
                    plot_type='', plot_tics=False):
        """

        :param model1:
        :param model2:
        :param reduced1:
        :param reduced2:
        :param data_elements:
        :param title_prefix1:
        :param title_prefix2:
        :param plot_type: 'heat_map', 'color_map'
        :param plot_tics: True, False
        :return:
        """
        # TODO: model, reduced, title_prefix same length and not only for 2 models, but for n one.
        if plot_tics:
            self.fig, self.axes = plt.subplots(3, 2)  # subplots(rows, columns) axes[row][column]
            main_axes = self.axes[0]
            for i, reduced in enumerate([reduced1, reduced2]):
                for component in [0, 1]:
                    self.plot_time_tics(self.axes[component+1][i], reduced, data_elements, component=component)
        else:
            self.fig, self.axes = plt.subplots(1, 2)
            main_axes = self.axes

        if plot_type == 'heat_map':
            for i, (reduced, model) in enumerate([(reduced1, model1), (reduced2, model2)]):
                self.plot_transformed_data_heat_map(main_axes[i], reduced, data_elements, model=model)
        else:
            for i, (reduced, model, title_prefix) in enumerate([(reduced1, model1, title_prefix1),
                                                                (reduced2, model2, title_prefix2)]):
                self.plot_transformed_data_on_axis(main_axes[i], reduced, data_elements, model=model,
                                                   color_map=plot_type, title_prefix=title_prefix)
        plt.show()

    def plot_one_model(self, model, reduced_data):
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)
        self.axes.cla()
        self.axes.scatter(reduced_data[0][:, 0], reduced_data[0][:, 1], c='r', marker='.')
        self.axes.scatter(reduced_data[1][:, 0], reduced_data[1][:, 1], c='b', marker='.')
        self.axes.scatter(reduced_data[2][:, 0], reduced_data[2][:, 1], c='g', marker='.')
        plt.show()

    def plot_transformed_data_on_axis(self, ax, data_list, data_elements, model, color_map, title_prefix=''):
        ax.cla()
        ax.set_title(title_prefix + str(model))
        ax.set_xlabel('1st component')
        ax.set_ylabel('2nd component')
        for index, element in enumerate(data_elements):
            if color_map == 'color_map':
                color_array = np.arange(data_list[element].shape[0])
                c_map = plt.cm.viridis
                im = ax.scatter(data_list[element][:, 0], data_list[element][:, 1], c=color_array,
                                cmap=c_map, marker='.')
                if index == 0:
                    self.fig.colorbar(im, ax=ax)
            else:
                color = list(self.colors.values())[element]
                ax.scatter(data_list[element][:, 0], data_list[element][:, 1], c=color, marker='.')

        self.print_model_properties(ax, data_list, model)

    def plot_time_tics(self, ax, data_list, data_elements, component):
        ax.cla()
        ax.set_xlabel('Time step')
        ax.set_ylabel('Component {}'.format(component+1))

        for i in data_elements:
            ax.plot(data_list[i][:, component], c=list(self.colors.values())[i])

    def plot_transformed_data_heat_map(self, ax, data_list, data_elements, model):
        ax.cla()
        ax.set_title(str(model))
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        for i in data_elements:
            xi = data_list[i][:, 0]
            yi = data_list[i][:, 1]
            bins = 50
            z, x, y = np.histogram2d(xi, yi, bins)
            np.seterr(divide='ignore')
            free_energies = -np.log(z)
            np.seterr(divide='warn')
            ax.contourf(free_energies.T, bins, cmap=plt.cm.hot, extent=[x[0], x[-1], y[0], y[-1]])
        self.print_model_properties(ax, data_list, model)

    @staticmethod
    def print_model_properties(ax, data_list, model):
        try:
            print(data_list[0].shape)
            print(model.eigenvectors, model.eigenvalues)
            # ax.arrow(0, 0, model.eigenvectors[0, 0], model.eigenvectors[1, 0], color='tab:cyan')  # pyemma
            # ax.arrow(0, 0, model.eigenvectors_[0, 0], model.eigenvectors_[1, 0], color='tab:cyan')  # msm builder
        except AttributeError as e:
            print('{}: {}'.format(e, model))

    def ramachandran_plot(self, phi, psi):
        pass