import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity

from utils import function_name
from utils.param_key import *


class MyPlotter:
    def __init__(self, interactive=True):
        self.fig = None
        self.axes = None
        self.interactive = interactive
        self.colors = mcolors.TABLEAU_COLORS

        if self.interactive:
            mpl.use('TkAgg')


class TrajectoryPlotter(MyPlotter):
    def __init__(self, trajectory, interactive=True):
        super().__init__(interactive)
        self.data_trajectory = trajectory

    def _set_figure_title(self):
        self.fig.suptitle(f'Trajectory: {self.data_trajectory.params[TRAJECTORY_NAME]}-'
                          f'{self.data_trajectory.filename}')

    def plot_trajectory_at(self, timestep: int):
        """
        This function gives the opportunity to plot the trajectory at a specific timeframe.
        :param timestep: int
            Time step value of the trajectory
        """
        self.fig = plt.figure()
        self._set_figure_title()
        self.axes = Axes3D(self.fig)
        self._update_on_slider_change(timestep)

    def original_data_with_timestep_slider(self, min_max=None):
        """
        Creates an interactive plot window, where the trajectory to plot can can be chosen by a Slider at a specific
        timestep. Used as in https://matplotlib.org/stable/gallery/widgets/slider_demo.html
        :param min_max: data range of the data with a min and a max value
        """
        if not self.interactive:
            raise ValueError('Plotter has to be interactive to use this plot.')

        if min_max is None:  # min_max is a list of two elements
            min_max = [0, self.data_trajectory.dim[TIME_FRAMES]]

        self.fig = plt.figure()
        self.axes = Axes3D(self.fig)
        self._set_figure_title()

        plt.subplots_adjust(bottom=0.25)
        # noinspection PyTypeChecker
        ax_freq = plt.axes([0.25, 0.1, 0.65, 0.03])
        freq_slider = Slider(
            ax=ax_freq,
            label='Time Step',
            valmin=min_max[0],  # minimum value of range
            valmax=min_max[-1] - 1,  # maximum value of range
            valinit=1,
            valstep=1,  # step between values
            valfmt='%0.0f'
        )
        freq_slider.on_changed(self._update_on_slider_change)
        self._update_on_slider_change(0)
        plt.show()

    def _update_on_slider_change(self, timeframe):
        """
        Callable function for the slider, which updates the figure.
        :param timeframe: Input value of the slider.
        :return:
        """
        if 0 <= timeframe <= self.data_trajectory.traj.n_frames:
            timeframe = int(timeframe)
            if self.data_trajectory.params[CARBON_ATOMS_ONLY]:
                data_tensor = self.data_trajectory.alpha_carbon_coordinates
            else:
                data_tensor = self.data_trajectory.traj.xyz

            if self.data_trajectory.params[STANDARDIZED_PLOT]:
                # numerator = data_tensor - np.mean(data_tensor, axis=0)[np.newaxis, :, :]  # PCA - center by atoms
                numerator = data_tensor - np.mean(data_tensor, axis=1)[:, np.newaxis, :]  # correct: center over time
                # numerator = data_tensor - np.mean(data_tensor, axis=2)[:, :, np.newaxis]  # Raumdiagonale gleich
                denominator = np.std(data_tensor, axis=0)
                data_tensor = numerator  # / denominator
                coordinate_mins = {X: data_tensor[:, :, 0].min(), Y: data_tensor[:, :, 1].min(),
                                   Z: data_tensor[:, :, 2].min()}
                coordinate_maxs = {X: data_tensor[:, :, 0].max(), Y: data_tensor[:, :, 1].max(),
                                   Z: data_tensor[:, :, 2].max()}
            else:
                coordinate_mins = self.data_trajectory.coordinate_mins
                coordinate_maxs = self.data_trajectory.coordinate_maxs

            x_coordinates = data_tensor[timeframe, :, 0]
            y_coordinates = data_tensor[timeframe, :, 1]
            z_coordinates = data_tensor[timeframe, :, 2]
            self.axes.cla()
            self.axes.plot(x_coordinates, y_coordinates, z_coordinates, c='r', marker='o')
            self.axes.set_xlim(coordinate_mins[X], coordinate_maxs[X])
            self.axes.set_ylim(coordinate_mins[Y], coordinate_maxs[Y])
            self.axes.set_zlim(coordinate_mins[Z], coordinate_maxs[Z])
            self.axes.set_xlabel('x-Axis')
            self.axes.set_ylabel('y-Axis')
            self.axes.set_zlabel('z-Axis')
            plt.show()
        else:
            raise IndexError('Timestep does not exist')

    def plot_models(self, model_results, data_elements, plot_type='', plot_tics=False, components=None):
        """
        Plots the model results in 2d-coordinate system next to each other.
        Alternatively with tics of the components can be plotted under the figures when `plot_tics` is True
        :param model_results: list of dictionary
            dict should contain the keys: 'model', 'projection', 'title_prefix' (optional)
        :param data_elements: List of elements
            The result of the models can contain a list of results,
            from which is possible to choose with this parameter
        :param plot_type: param_key.plot_type
        :param plot_tics: bool (default: False)
            Plots the component tics under the base figures if True
        :param components: int
            Number of components used for the reduced
        """
        if plot_tics:
            self.fig, self.axes = plt.subplots(components + 1, len(model_results))  # subplots(rows, columns)
            main_axes = self.axes[0]  # axes[row][column]
            if len(model_results) == 1:
                for component_nr in range(components + 1)[1:]:
                    self._plot_time_tics(self.axes[component_nr], model_results[0][PROJECTION], data_elements,
                                         component=component_nr)
            else:
                for i, result in enumerate(model_results):
                    for component_nr in range(components + 1)[1:]:
                        self._plot_time_tics(self.axes[component_nr][i], result[PROJECTION], data_elements,
                                             component=component_nr)
        else:
            self.fig, self.axes = plt.subplots(1, len(model_results))
            main_axes = self.axes
        self._set_figure_title()
        if plot_type == HEAT_MAP:
            if len(model_results) == 1:
                self._plot_transformed_data_heat_map(main_axes, model_results[0], data_elements)
            else:
                for i, result in enumerate(model_results):
                    self._plot_transformed_data_heat_map(main_axes[i], result, data_elements)
        else:
            if len(model_results) == 1:
                self._plot_transformed_trajectory(main_axes, model_results[0], data_elements, color_map=plot_type)
            else:
                for i, result in enumerate(model_results):
                    self._plot_transformed_trajectory(main_axes[i], result, data_elements, color_map=plot_type)
        plt.show()

    def _plot_transformed_trajectory(self, ax, projection, data_elements, color_map):
        """
        Plot the projection results of the transformed trajectory on an axis
        :param ax: Which axis the result should be plotted on
        :param projection: dictionary
            dict should contain the keys: 'model', 'projection',
            and optional keys: 'title_prefix', 'explained_variance'
        :param data_elements: List of elements
            The result of the models can contain a list of results,
            from which is possible to choose with this parameter
        :param color_map: str
            String value of the plot mapping type
        """
        ax.cla()
        ex_var = projection.get(EXPLAINED_VAR, None)
        ax.set_title(projection.get(TITLE_PREFIX, '') +
                     str(projection[MODEL]) +
                     (f'\nExplained var: {ex_var:.4f}' if ex_var is not None else ''),
                     fontsize=8, wrap=True)
        ax.set_xlabel('1st component')
        ax.set_ylabel('2nd component')
        data_list = projection[PROJECTION]
        for index, element in enumerate(data_elements):
            if color_map == COLOR_MAP:
                color_array = np.arange(data_list[element].shape[0])
                c_map = plt.cm.viridis
                im = ax.scatter(data_list[element][:, 0], data_list[element][:, 1], c=color_array,
                                cmap=c_map, marker='.')
                if index == 0:
                    self.fig.colorbar(im, ax=ax)
            else:
                color = list(self.colors.values())[element]
                ax.scatter(data_list[element][:, 0], data_list[element][:, 1], c=color, marker='.')

        self._print_model_properties(projection)

    def _plot_time_tics(self, ax, projections, data_elements, component):
        """
        Plot the time tics on a specific axis
        :param ax: axis
        :param projections:
        :param data_elements: List of elements
            The result of the models can contain a list of results,
            from which is possible to choose with this parameter.
        :param component:
        :return:
        """
        ax.cla()
        ax.set_xlabel('Time step')
        ax.set_ylabel('Component {}'.format(component))

        for i in data_elements:
            ax.plot(projections[i][:, component - 1], c=list(self.colors.values())[i])

    def _plot_transformed_data_heat_map(self, ax, projection_dict, data_elements):
        """

        :param ax:
        :param projection_dict:
        :param data_elements: List of elements
            The result of the models can contain a list of results,
            from which is possible to choose with this parameter.
        :return:
        """
        ax.cla()
        ax.set_title(str(projection_dict[MODEL]))
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        projections = projection_dict[PROJECTION]
        for i in data_elements:
            xi = projections[i][:, 0]
            yi = projections[i][:, 1]
            bins = 50
            z, x, y = np.histogram2d(xi, yi, bins)
            np.seterr(divide='ignore')
            free_energies = -np.log(z, dtype='float')
            np.seterr(divide='warn')
            ax.contourf(free_energies.T, bins, cmap=plt.cm.hot, extent=[x[0], x[-1], y[0], y[-1]])
        self._print_model_properties(projection_dict)

    @staticmethod
    def _print_model_properties(projection_dict):
        """
        Prints the model result properties: Eigenvector (EV) and Eigenvalue (EW)
        :param projection_dict: dictionary
            dict should contain the keys: 'model', 'projection'
        """
        projections = projection_dict[PROJECTION]
        model = projection_dict[MODEL]
        try:
            print(projections[0].shape)
            print('EV:', model.eigenvectors, 'EW:', model.eigenvalues)
        except AttributeError as e:
            print('{}: {}'.format(e, model))


class MultiTrajectoryPlotter(MyPlotter):
    def plot_principal_components(self, algorithm, principal_components, components):
        self.fig, self.axes = plt.subplots(1, components)
        self.fig.suptitle(algorithm)
        for component in range(0, components):
            x_range = np.asarray(list(range(0, principal_components.shape[1])))
            principal_component = principal_components[:, :, component]
            cos_sim = cosine_similarity(principal_component)
            print(f'Cosine similarities using {algorithm}-algorithm for component {component}: {cos_sim}')
            self.axes[component].plot(x_range, principal_component.T)
            self.axes[component].set_title(f'Component Nr {component + 1}')
        plt.show()


class ArrayPlotter(MyPlotter):
    def __init__(self, interactive=False, bottom_text=None):
        super().__init__(interactive)
        self.bottom_text = bottom_text

    def __show(self):
        if self.bottom_text is not None:
            self.fig.text(0.01, 0.01, self.bottom_text, fontsize=10)
            self.fig.tight_layout()
            self.fig.subplots_adjust(bottom=(self.bottom_text.count('\n') + 1) * 0.1)
        else:
            self.fig.tight_layout()
        plt.show()

    def matrix_plot(self, matrix, title_prefix='', xy_label='', as_surface='2d'):
        """
        Plots the values of a matrix on a 2d or a 3d axes
        :param matrix: ndarray (2-ndim)
            matrix, which should be plotted
        :param title_prefix: str
        :param xy_label: str
            The description on the x and y label
        :param as_surface: str
            Plot as a 3d-surface if value PLOT_3D_MAP else 2d-axes
        """
        c_map = plt.cm.viridis
        if as_surface == PLOT_3D_MAP:
            x_coordinates = np.arange(matrix.shape[0])
            y_coordinates = np.arange(matrix.shape[1])
            x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
            self.fig = plt.figure()
            self.axes = self.fig.gca(projection='3d')
            self.axes.set_zlabel('covariance values')
            im = self.axes.plot_surface(x_coordinates, y_coordinates, matrix, cmap=c_map)
        else:
            self.fig, self.axes = plt.subplots(1, 1)
            im = self.axes.matshow(matrix, cmap=c_map)
        self.fig.colorbar(im, ax=self.axes)
        self.axes.set_xlabel(xy_label)
        self.axes.set_ylabel(xy_label)
        self.axes.set_title(title_prefix + ' Matrix', fontsize=10)
        # print(f'{title_prefix}: {matrix}')
        self.__show()

    def plot_gauss2d(self, xdata: np.ndarray, ydata: np.ndarray, new_ydata: np.ndarray, gauss_fitted: np.ndarray,
                     fit_method: str, title_prefix: str = '', statistical_function: callable = np.median):
        """
        Plot the data (ydata) in a range (xdata), the (fitted) gauss curve and a line (mean, median)
        :param xdata: ndarray (1-ndim)
            range of plotting
        :param ydata: ndarray (1-ndim)
            original data
        :param new_ydata: ndarray (1-ndim)
            the changed new data
        :param gauss_fitted: ndarray (1-ndim)
            the fitted curve on the new data
        :param fit_method: str
            the name of the fitting method
        :param title_prefix: str
            title of the plot
        :param statistical_function: callable
            Some statistical numpy function
        :return:
        """
        self.fig, self.axes = plt.subplots(1, 1)
        self.axes.plot(xdata, gauss_fitted, '-', label=f'fit {fit_method}')
        self.axes.plot(xdata, ydata, '.', label='original data')
        statistical_value = np.full(xdata.shape, statistical_function(ydata))
        function_label = function_name(statistical_function)
        self.axes.plot(xdata, statistical_value, '-', label=function_label)
        self.axes.plot(xdata, new_ydata, '.', label='interpolated data')
        self.axes.set_xlabel('matrix diagonal indexes')
        self.axes.set_ylabel(f'{function_label}-ed correlation values')
        self.axes.legend()
        self.axes.set_title(title_prefix)
        # self.axes.set_ylim(-1, 1)  # normalize plot
        self.__show()

    def plot_2d(self, ndarray_data, title_prefix='', xlabel='', ylabel='', statistical_func=None):
        self.fig, self.axes = plt.subplots(1, 1)
        self.axes.plot(ndarray_data, '-')
        if statistical_func is not None:
            statistical_value = statistical_func(ndarray_data)
            statistical_value_line = np.full(ndarray_data.shape, statistical_value)
            self.axes.plot(statistical_value_line, '-', label=f'{function_name(statistical_func)}: {statistical_value}')
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title_prefix)
        self.fig.tight_layout()
        self.axes.legend()
        self.__show()
