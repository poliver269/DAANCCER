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
    def __init__(self, interactive=True, title_prefix='', for_paper=False):
        self.fig = None
        self.axes = None
        self.interactive = interactive
        self.title_prefix = title_prefix
        self.colors = mcolors.TABLEAU_COLORS
        self.for_paper = for_paper

        if self.interactive:
            mpl.use('TkAgg')

        if self.for_paper:
            self.fontsize = 18
        else:
            self.fontsize = 10

    def _set_figure_title(self):
        self.fig.suptitle(self.title_prefix)

    def _post_processing(self):
        if not self.for_paper:
            self._set_figure_title()
        plt.show()


class TrajectoryPlotter(MyPlotter):
    def __init__(self, trajectory, reconstruct_params=None, interactive=True, for_paper=False):
        super().__init__(interactive, for_paper=for_paper)
        self.data_trajectory = trajectory
        if reconstruct_params is not None:
            self.reconstructed = self.data_trajectory.get_reconstructed_traj(reconstruct_params)
        else:
            self.reconstructed = None

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

    def data_with_timestep_slider(self, min_max=None):
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
            if self.reconstructed is not None:
                data_tensor = self.reconstructed
            elif self.data_trajectory.params[CARBON_ATOMS_ONLY]:
                data_tensor = self.data_trajectory.alpha_carbon_coordinates
            else:
                data_tensor = self.data_trajectory.traj.xyz

            if self.data_trajectory.params[STANDARDIZED_PLOT]:
                # numerator = data_tensor - np.mean(data_tensor, axis=0)[np.newaxis, :, :]  # PCA - center by atoms
                numerator = data_tensor - np.mean(data_tensor, axis=1)[:, np.newaxis, :]  # center over time
                # numerator = data_tensor - np.mean(data_tensor, axis=2)[:, :, np.newaxis]  # Raumdiagonale gleich
                denominator = np.std(data_tensor, axis=0)
                data_tensor = numerator / denominator
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


class ModelResultPlotter(MyPlotter):
    def plot_models(self, model_results, plot_type='', plot_tics=False, components=None):
        """
        Plots the model results in 2d-coordinate system next to each other.
        Alternatively with tics of the components can be plotted under the figures when `plot_tics` is True
        :param model_results: list of dictionary
            dict should contain the keys: 'model', 'projection', 'title_prefix' (optional)
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
                    self._plot_time_tics(self.axes[component_nr], model_results[0][PROJECTION], component=component_nr)
            else:
                for i, result in enumerate(model_results):
                    for component_nr in range(components + 1)[1:]:
                        self._plot_time_tics(self.axes[component_nr][i], result[PROJECTION], component=component_nr)
        else:
            self.fig, self.axes = plt.subplots(1, len(model_results))
            main_axes = self.axes
        if plot_type == HEAT_MAP:
            if len(model_results) == 1:
                self._plot_transformed_data_heat_map(main_axes, model_results[0])
            else:
                for i, result in enumerate(model_results):
                    self._plot_transformed_data_heat_map(main_axes[i], result)
        else:
            if len(model_results) == 1:
                self._plot_transformed_trajectory(main_axes, model_results[0], color_map=plot_type,
                                                  show_model_properties=True)
            else:
                for i, result in enumerate(model_results):
                    self._plot_transformed_trajectory(main_axes[i], result, color_map=plot_type,
                                                      show_model_properties=True)
        plt.show()

    def _plot_transformed_trajectory(self, ax, result_dict, color_map, show_model_properties=False, center_plot=False):
        """
        Plot the projection results of the transformed trajectory on an axis
        :param ax: Which axis the result should be plotted on
        :param result_dict: dictionary
            dict should contain the keys: 'model', 'projection',
            and optional keys: 'title_prefix', 'explained_variance'
        :param color_map: str
            String value of the plot mapping type
        """
        ax.cla()
        if show_model_properties:
            ex_var = result_dict.get(EXPLAINED_VAR, None)
            ax.set_title(result_dict.get(TITLE_PREFIX, '') +
                         str(result_dict[MODEL]) +
                         (f'\nExplained var: {ex_var:.4f}' if ex_var is not None else ''),
                         fontsize=8, wrap=True)
            ax.set_xlabel('1st component')
            ax.set_ylabel('2nd component')
        if color_map == COLOR_MAP:
            color_array = np.arange(result_dict[PROJECTION].shape[0])
            c_map = plt.cm.viridis
            im = ax.scatter(result_dict[PROJECTION][:, 0], result_dict[PROJECTION][:, 1], c=color_array,
                            cmap=c_map, marker='.')
            if not self.for_paper:
                self.fig.colorbar(im, ax=ax)
        else:
            ax.scatter(result_dict[PROJECTION][:, 0], result_dict[PROJECTION][:, 1], marker='.')

        if show_model_properties:
            self._print_model_properties(result_dict)

        if center_plot:
            # Move left y-axis and bottom x-axis to centre, passing through (0,0)
            ax.spines['left'].set_position('zero')
            # ax.spines['bottom'].set_position('center')
            ax.spines['bottom'].set_position('zero')

            # Eliminate upper and right axes
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            # Show ticks in the left and lower axes only
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        else:
            ax.tick_params(left=False, right=False, labelleft=False,
                           labelbottom=False, bottom=False)

    @staticmethod
    def _plot_time_tics(ax, projection, component):
        """
        Plot the time tics on a specific axis
        :param ax: axis
        :param projection:
        :param component:
        :return:
        """
        ax.cla()
        ax.set_xlabel('Time step')
        ax.set_ylabel('Component {}'.format(component))

        ax.plot(projection[:, component - 1])

    def _plot_transformed_data_heat_map(self, ax, result_dict):
        """
        :param ax:
        :param result_dict:
        :return:
        """
        ax.cla()
        ax.set_title(str(result_dict[MODEL]))
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        projection = result_dict[PROJECTION]
        xi = projection[:, 0]
        yi = projection[:, 1]
        bins = 50
        z, x, y = np.histogram2d(xi, yi, bins)
        np.seterr(divide='ignore')
        free_energies = -np.log(z, dtype='float')
        np.seterr(divide='warn')
        ax.contourf(free_energies.T, bins, cmap=plt.cm.hot, extent=[x[0], x[-1], y[0], y[-1]])
        self._print_model_properties(result_dict)

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

    def plot_multi_projections(self, model_result_list_in_list, plot_type, center_plot=True):
        self.fig, self.axes = plt.subplots(len(model_result_list_in_list), len(model_result_list_in_list[0]))
        for row_index, model_results in enumerate(model_result_list_in_list):
            if len(model_results) == 1:
                self._plot_transformed_trajectory(self.axes[row_index], model_results[0], color_map=plot_type,
                                                  center_plot=center_plot)
            else:
                for column_index, result in enumerate(model_results):
                    self._plot_transformed_trajectory(self.axes[row_index][column_index], result, color_map=plot_type,
                                                      center_plot=center_plot)
        self._post_processing()


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
    def __init__(self, interactive=False, title_prefix='', x_label='', y_label='', bottom_text=None, y_range=None,
                 show_grid=False, xtick_start=0, for_paper=False):
        super().__init__(interactive, title_prefix, for_paper)
        self.x_label = x_label
        self.y_label = y_label
        self.bottom_text = bottom_text
        self.range_tuple = y_range
        self._activate_legend = False
        self.show_grid = show_grid
        self.xtick_start = xtick_start

    def _post_processing(self, legend_outside=False):
        # self.axes.set_title(self.title_prefix)
        self.axes.set_xlabel(self.x_label, fontsize=self.fontsize)
        self.axes.set_ylabel(self.y_label, fontsize=self.fontsize)
        # plt.xticks(fontsize=self.fontsize)
        # plt.yticks(fontsize=self.fontsize)

        if self.bottom_text is not None:
            self.fig.text(0.01, 0.01, self.bottom_text, fontsize=self.fontsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(bottom=(self.bottom_text.count('\n') + 1) * 0.1)
        else:
            self.fig.tight_layout()

        if legend_outside:
            self.axes.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', fontsize=8)
            plt.subplots_adjust(bottom=0.25)
        elif self._activate_legend:
            self.axes.legend(fontsize=self.fontsize)

        if self.range_tuple is not None:
            self.axes.set_ylim(self.range_tuple)

        if self.show_grid:
            plt.grid(True, which='both')
            plt.minorticks_on()
        super()._post_processing()

    def matrix_plot(self, matrix, as_surface='2d', show_values=False):
        """
        Plots the values of a matrix on a 2d or a 3d axes
        :param matrix: ndarray (2-ndim)
            matrix, which should be plotted
        :param as_surface: str
            Plot as a 3d-surface if value PLOT_3D_MAP else 2d-axes
        :param show_values: If true, then show the values in the matrix
        """
        c_map = plt.cm.viridis
        # c_map = plt.cm.seismic
        if as_surface == PLOT_3D_MAP:
            x_coordinates = np.arange(matrix.shape[0])
            y_coordinates = np.arange(matrix.shape[1])
            x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
            self.fig = plt.figure()
            self.axes = self.fig.gca(projection='3d')
            self.axes.set_zlabel('Covariance Values', fontsize=self.fontsize)
            im = self.axes.plot_surface(x_coordinates, y_coordinates, matrix, cmap=c_map)
        else:
            self.fig, self.axes = plt.subplots(1, 1, dpi=80)
            im = self.axes.matshow(matrix, cmap=c_map)
            if show_values:
                for (i, j), value in np.ndenumerate(matrix):
                    self.axes.text(j, i, '{:0.2f}'.format(value), ha='center', va='center', fontsize=8)
        if not self.for_paper:
            self.fig.colorbar(im, ax=self.axes)
            plt.xticks(np.arange(matrix.shape[1]), np.arange(self.xtick_start, matrix.shape[1] + self.xtick_start))
            # plt.xticks(np.arange(matrix.shape[1], step=5),
            #            np.arange(self.xtick_start, matrix.shape[1] + self.xtick_start, step=5))
        self._post_processing()

    def plot_gauss2d(self,
                     x_index: np.ndarray,
                     ydata: np.ndarray,
                     new_ydata: np.ndarray,
                     gauss_fitted: np.ndarray,
                     fit_method: str,
                     statistical_function: callable = np.median):
        """
        Plot the original data (ydata), the new data (new_ydata) where the x-axis-indices is given by (x_index),
        the (fitted) gauss curve and a line (mean, median)
        :param x_index: ndarray (1-ndim)
            range of plotting
        :param ydata: ndarray (1-ndim)
            original data
        :param new_ydata: ndarray (1-ndim)
            the changed new data
        :param gauss_fitted: ndarray (1-ndim)
            the fitted curve on the new data
        :param fit_method: str
            the name of the fitting method
        :param statistical_function: callable
            Some statistical numpy function
        :return:
        """
        self.fig, self.axes = plt.subplots(1, 1, dpi=80)
        self.axes.plot(x_index, gauss_fitted, '-', label=f'fit {fit_method}')
        # self.axes.plot(x_index, gauss_fitted, ' ')
        self.axes.plot(x_index, ydata, '.', label='original data')
        # self.axes.plot(x_index, ydata, ' ')
        statistical_value = np.full(x_index.shape, statistical_function(ydata))
        if self.for_paper:
            function_label = 'threshold'
        else:
            function_label = function_name(statistical_function)
        self.axes.plot(x_index, statistical_value, '-', label=function_label)
        # self.axes.plot(x_index, statistical_value, ' ')
        self.axes.plot(x_index, new_ydata, '.', label='re-scaled data')
        self._activate_legend = True
        self._post_processing()

    def plot_2d(self, ndarray_data, statistical_func=None):
        self.fig, self.axes = plt.subplots(1, 1)
        self.axes.plot(ndarray_data, '-')
        if statistical_func is not None:
            statistical_value = statistical_func(ndarray_data)
            statistical_value_line = np.full(ndarray_data.shape, statistical_value)
            self.axes.plot(statistical_value_line, '-',
                           label=f'{function_name(statistical_func)}: {statistical_value:.4f}')
        self._activate_legend = False
        self._post_processing()

    def plot_merged_2ds(self, ndarray_dict: dict, statistical_func=None):
        self.fig, self.axes = plt.subplots(1, 1, dpi=80)
        self.title_prefix += f'with {function_name(statistical_func)}' if statistical_func is not None else ''
        for key, ndarray_data in ndarray_dict.items():
            # noinspection PyProtectedMember
            color = next(self.axes._get_lines.prop_cycler)['color']
            if statistical_func is not None:
                if isinstance(ndarray_data, list):
                    ndarray_data = np.asarray(ndarray_data)
                self.axes.plot(ndarray_data, '-', color=color)
                statistical_value = statistical_func(ndarray_data)
                statistical_value_line = np.full(ndarray_data.shape, statistical_value)
                self.axes.plot(statistical_value_line, '--',
                               label=f'{key.strip()}: {statistical_value:.4f}', color=color)
            else:
                self.axes.plot(ndarray_data, '-', color=color, label=f'{key.strip()[:35]}')

        self._activate_legend = True
        self._post_processing()
