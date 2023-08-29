import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity

from utils import function_name, get_algorithm_name, ordinal, nr_in_human_format
from utils.param_keys import TRAJECTORY_NAME, CARBON_ATOMS_ONLY, X, Y, Z, DUMMY_ZERO, DUMMY_ONE
from utils.param_keys.analyses import HEAT_MAP, COLOR_MAP, PLOT_3D_MAP
from utils.param_keys.kernel_functions import MY_SINC
from utils.param_keys.model_result import MODEL, PROJECTION, TITLE_PREFIX, EXPLAINED_VAR, FITTED_ON
from utils.param_keys.traj_dims import TIME_FRAMES
from utils.timer import Timer


class MyPlotter:
    def __init__(self, interactive: bool = True, title_prefix: str = '', for_paper: bool = False):
        self.fig: Figure = Figure()
        self.axes: Axes = Axes(self.fig, [0, 0, 0, 0])
        self.interactive: bool = interactive
        self.title_prefix: str = title_prefix
        self.colors: dict = mcolors.TABLEAU_COLORS
        self.line_styles = iter(Line2D.lineStyles)
        self.for_paper: bool = for_paper

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
        else:
            plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()


class ProteinPlotter(MyPlotter):
    def __init__(self, trajectory, reconstruct_params=None, interactive=True, for_paper=False, standardize=False):
        super().__init__(interactive, for_paper=for_paper)
        self.protein_trajectory = trajectory
        self.standardize: bool = standardize
        if reconstruct_params is not None:
            self.reconstructed: np.ndarray = self.protein_trajectory.get_reconstructed_traj(reconstruct_params)
        else:
            self.reconstructed = None

    def _set_figure_title(self):
        self.fig.suptitle(f'Trajectory: {self.protein_trajectory.params[TRAJECTORY_NAME]}-'
                          f'{self.protein_trajectory.filename}')

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
        Creates an interactive plot window, where the trajectory is plotted and the time-step can be chosen by a Slider.
        Used as in https://matplotlib.org/stable/gallery/widgets/slider_demo.html
        :param min_max: data range of the data with a min and a max value
        """
        if not self.interactive:
            raise ValueError('Plotter has to be interactive to use this plot.')

        if min_max is None:  # min_max is a list of two elements
            min_max = [0, self.protein_trajectory.dim[TIME_FRAMES]]

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
        Callable function for the slider, which updates the figure
        :param timeframe: Input value of the slider.
        :return:
        """
        if 0 <= timeframe <= self.protein_trajectory.traj.n_frames:
            timeframe = int(timeframe)
            if self.reconstructed is not None:
                data_tensor = self.reconstructed
            elif self.protein_trajectory.params[CARBON_ATOMS_ONLY]:
                data_tensor = self.protein_trajectory.alpha_carbon_coordinates
            else:
                data_tensor = self.protein_trajectory.traj.xyz

            if self.standardize:
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
                coordinate_mins = self.protein_trajectory.coordinate_mins
                coordinate_maxs = self.protein_trajectory.coordinate_maxs

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
            self.fig, self.axes = plt.subplots(components + 1, len(model_results),
                                               constrained_layout=True)  # subplots(rows, columns)
            main_axes = self.axes[0]  # axes[row][column]
            if len(model_results) == 1:
                for component_nr in range(components + 1)[1:]:
                    self._plot_time_tics(self.axes[component_nr], model_results[DUMMY_ZERO][PROJECTION],
                                         component=component_nr)
            else:
                for i, result in enumerate(model_results):
                    for component_nr in range(components + 1)[1:]:
                        self._plot_time_tics(self.axes[component_nr][i], result[PROJECTION], component=component_nr)
        else:
            self.fig, self.axes = plt.subplots(1, len(model_results), constrained_layout=True)
            main_axes = self.axes

        if plot_type == HEAT_MAP:
            if len(model_results) == 1:
                self._plot_transformed_data_heat_map(main_axes, model_results[DUMMY_ZERO])
            else:
                for i, result in enumerate(model_results):
                    self._plot_transformed_data_heat_map(main_axes[i], result)
        else:
            if len(model_results) == 1:
                self._plot_transformed_trajectory(main_axes, model_results[DUMMY_ZERO], color_map=plot_type,
                                                  show_model_properties=True)
            else:
                for i, result in enumerate(model_results):
                    self._plot_transformed_trajectory(main_axes[i], result, color_map=plot_type,
                                                      show_model_properties=True)
        plt.show()
        print(Timer().timers)

    def _plot_transformed_trajectory(self, ax, result_dict: dict, color_map: str, show_model_properties: bool = False,
                                     center_plot: bool = False, sub_part: [int, None] = None):
        """
        Plot the projection results of the transformed trajectory in a 2d-axis
        :param ax: Which axis the result should be plotted on
        :param result_dict: dictionary
            dict should contain the keys: 'model', 'projection',
            and optional keys: 'title_prefix', 'explained_variance'
        :param color_map: str
            String value of the plot mapping type
        :param show_model_properties: bool
            Show the model name and explained variance as title
        :param center_plot: bool
            Parameter for centering the projection
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
            # self._print_model_properties(result_dict)
        elif sub_part is None:
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(result_dict.get(FITTED_ON, 'Not Found').replace('-0-protein', ''),
                              fontsize=self.fontsize)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel(result_dict.get(TITLE_PREFIX, 'Not Found').replace('-0-protein', ''),
                              fontsize=self.fontsize)

        if color_map == COLOR_MAP:
            if sub_part == 0 or sub_part is None:
                color_array = np.arange(result_dict[PROJECTION].shape[0])
            else:
                shape = result_dict[PROJECTION].shape[0]
                color_array = np.arange((sub_part - 1) * shape, sub_part * shape)

            if not show_model_properties and sub_part is not None:  # else part of show_model_properties
                if ax.get_subplotspec().is_first_col():
                    ax.set_ylabel(get_algorithm_name(result_dict[MODEL]), size='large')
                if ax.get_subplotspec().is_first_row():
                    ax.set_title(f'Steps {color_array[0]}-{color_array[-1]}')

            result_dict[PROJECTION] = self.hard_corded_projection_flip(ax, result_dict, superposing='PDB')
            c_map = plt.cm.viridis
            im = ax.scatter(result_dict[PROJECTION][:, 0], result_dict[PROJECTION][:, 1], c=color_array,
                            cmap=c_map, marker='.', alpha=0.1)
            if not self.for_paper and ((ax.get_subplotspec().is_last_col() and sub_part is None) or
                                       (ax.get_subplotspec().is_first_col() and sub_part is not None)):
                self.fig.colorbar(im, ax=ax, alpha=10.0)
            elif self.for_paper:
                if get_algorithm_name(result_dict[MODEL]) == 'PCA':
                    ax.set_xlim((-15, 15))
                    ax.set_ylim((-15, 15))
                elif get_algorithm_name(result_dict[MODEL]) == 'DROPP':
                    ax.set_xlim(-5, 5)
                    ax.set_ylim(-5, 5)
                ax.set_xticks([])
                ax.set_yticks([])

            return im
        else:
            ax.scatter(result_dict[PROJECTION][:, 0], result_dict[PROJECTION][:, 1], marker='.')

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
            pass  # ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    @staticmethod
    def hard_corded_projection_flip(ax, result_dict, superposing=None):
        row_index = ax.get_subplotspec().rowspan.start
        col_index = ax.get_subplotspec().colspan.start
        model_description = get_algorithm_name(result_dict[MODEL])

        projection = result_dict[PROJECTION].copy()
        if superposing is not None:
            # TODO: do not use hardcoded flipping
            if superposing == 'PDB':
                flip_x_axis_pca = {(2, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)}
                flip_x_axis_dropp = {(2, 0), (3, 0), (4, 0), (3, 4), (1, 1),
                                     (1, 2), (1, 3), (2, 4), (4, 4)}
                flip_y_axis_pca = {(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)}
                flip_y_axis_dropp = {(1, 2), (4, 2), (2, 4), (3, 4)}
            else:  # Random-Superposing
                flip_x_axis_pca = {(0, 2), (0, 3), (2, 3), (2, 0), (3, 0), (4, 0), (1, 4), (2, 4)}
                flip_x_axis_dropp = {(0, 1), (0, 3), (0, 4), (1, 0), (2, 0),
                                     (2, 1), (2, 3), (3, 2), (3, 4), (4, 2), (4, 4)}
                flip_y_axis_pca = {(1, 4), (3, 4)}
                flip_y_axis_dropp = {(0, 4), (1, 1), (1, 2), (2, 0), (3, 0), (4, 0)}

            flip_components_pca = {}
            flip_components_dropp = {}

            flip_x_axis = [(x[0], x[1], 'PCA') for x in flip_x_axis_pca]
            flip_x_axis += [(x[0], x[1], 'DROPP') for x in flip_x_axis_dropp]

            flip_y_axis = [(x[0], x[1], 'PCA') for x in flip_y_axis_pca]
            flip_y_axis += [(x[0], x[1], 'DROPP') for x in flip_y_axis_dropp]

            flip_components = [(x[0], x[1], 'PCA') for x in flip_components_pca]
            flip_components += [(x[0], x[1], 'DROPP') for x in flip_components_dropp]

            if (row_index, col_index, model_description) in flip_x_axis:
                projection[:, 0] = -projection[:, 0]

            if (row_index, col_index, model_description) in flip_y_axis:
                projection[:, 1] = -projection[:, 1]

            if (row_index, col_index, model_description) in flip_components:
                projection = projection[:, ::-1]

        return projection

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
        ax.label_outer()

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
            print('EV:', model.components_, 'EW:', model.explained_variance_)
        except AttributeError as e:
            print('{}: {}'.format(e, model))

    def plot_multi_projections(self, model_result_list_in_list, plot_type, center_plot=True, sub_parts=False,
                               show_model_properties=True):
        self.fig, self.axes = plt.subplots(len(model_result_list_in_list), len(model_result_list_in_list[DUMMY_ZERO]),
                                           figsize=(1080 / 100, 1080 / 100), dpi=100, sharex='all', sharey='all')
        for row_index, model_results in enumerate(model_result_list_in_list):
            if len(model_results) == 1:
                self._plot_transformed_trajectory(self.axes[row_index], model_results[DUMMY_ZERO],
                                                  color_map=plot_type,
                                                  center_plot=center_plot)
            else:
                for column_index, result in enumerate(model_results):
                    sub_part = column_index if sub_parts else None
                    self._plot_transformed_trajectory(self.axes[row_index][column_index], result,
                                                      color_map=plot_type, center_plot=center_plot,
                                                      sub_part=sub_part,
                                                      show_model_properties=show_model_properties)
        if not show_model_properties and not sub_parts:
            model_name = get_algorithm_name(model_result_list_in_list[DUMMY_ZERO][DUMMY_ZERO][MODEL])
            model_name = 'DROPP' if model_name == 'DAANCCER' else model_name
            self.fig.supylabel(f'{model_name} fitted on', fontsize=self.fontsize)
            self.fig.supxlabel(f'{model_name} transformed on', fontsize=self.fontsize)

            color_max = len(np.arange(model_result_list_in_list[DUMMY_ZERO][DUMMY_ZERO][PROJECTION].shape[0]))
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=color_max))
            sm.set_array([])  # Required for the ScalarMappable to work

            color_bar = self.fig.colorbar(sm, cax=self.fig.add_axes([0.74, 0.015, 0.2, 0.02]),
                                          ticks=[], orientation='horizontal')
            color_bar.ax.text(-100, 0.375, 0, ha='right', va='center', fontsize=self.fontsize)
            color_bar.ax.text(color_max + 100, 0.375, nr_in_human_format(color_max),
                              ha='left', va='center', fontsize=self.fontsize)

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
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)

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
            self.axes.legend(fontsize=self.fontsize, ncol=4)

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
        :param show_values: bool
            If True (default: False), then show the values in the matrix
        """
        c_map = plt.cm.viridis
        # c_map = plt.cm.seismic
        # c_map = plt.cm.hot
        if as_surface == PLOT_3D_MAP:
            x_coordinates = np.arange(matrix.shape[0])
            y_coordinates = np.arange(matrix.shape[1])
            x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
            self.fig = plt.figure()
            self.axes = self.fig.gca(projection='3d')
            self.axes.set_zlabel('Covariance Values', fontsize=self.fontsize)
            im = self.axes.plot_surface(x_coordinates, y_coordinates, matrix, cmap=c_map)
        else:
            self.fig, self.axes = plt.subplots(1, 1, figsize=(1080 / 180, 1080 / 180), dpi=180)
            im = self.axes.matshow(matrix, cmap=c_map)

            # Set tick labels for countries
            countries = False
            if countries:
                tick_labels = ["GB", "IE", "LT", "LV"]  # TODO: this should be the country_list, if available
                tick_positions = [(i * matrix.shape[DUMMY_ZERO] / len(tick_labels)) - 0.5 for i in
                                  range(len(tick_labels) + 1)]
                minor_tick_positions = [(tick_positions[i] + tick_positions[i + 1]) / 2 for i in
                                        range(len(tick_positions) - 1)]

                self.axes.set_xticks(tick_positions)
                self.axes.set_yticks(tick_positions)
                self.axes.set_xticklabels('')
                self.axes.set_yticklabels('')

                # Set the tick positions and labels for both axes (x and y)
                self.axes.set_xticks(minor_tick_positions, minor=True)
                self.axes.set_yticks(minor_tick_positions, minor=True)
                self.axes.set_xticklabels(tick_labels, minor=True, fontsize=self.fontsize)
                self.axes.set_yticklabels(tick_labels, minor=True, fontsize=self.fontsize)
                self.axes.tick_params(which='minor', width=0)
                # color_bar = self.fig.colorbar(im, ax=self.axes)
                # color_bar.ax.tick_params(labelsize=self.fontsize)
            if show_values:
                for (i, j), value in np.ndenumerate(matrix):
                    self.axes.text(j, i, '{:0.2f}'.format(value), ha='center', va='center', fontsize=8)
        if not self.for_paper:
            self.fig.colorbar(im, ax=self.axes)
            # plt.xticks(np.arange(matrix.shape[1]), np.arange(self.xtick_start, matrix.shape[1] + self.xtick_start))
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
        self.fig, self.axes = plt.subplots(1, 1, figsize=(1350 / 180, 1080 / 180), dpi=180)
        self.axes.plot(x_index, gauss_fitted, '-', label=f'fit {fit_method}', linewidth=3.0)
        # self.axes.plot(x_index, gauss_fitted, ' ')
        self.axes.plot(x_index, ydata, '.', label='original data')
        # self.axes.plot(x_index, ydata, ' ')
        statistical_value = np.full(x_index.shape, statistical_function(ydata))
        if self.for_paper:
            function_label = 'threshold'
            # TODO: xy is hardcoded for 2f4k
            if fit_method == MY_SINC + '_center':
                fit_method = MY_SINC
            self.axes.annotate('threshold', xy=(-5, 0.675), color='tab:green', fontsize=self.fontsize)
            self.axes.annotate('mean of\ndiagonals', xy=(-14, 0.8), color='tab:orange', fontsize=self.fontsize)
            # self.axes.annotate('rescaled\ndiagonal\nvalues', xy=(10, 0.03), color='tab:red', fontsize=self.fontsize)
            self.axes.annotate(f'{fit_method.replace("my_", "")}\ncurve', xy=(4, 0.88), color='tab:blue',
                               fontsize=self.fontsize)
            # for flattened data
            # self.axes.annotate('threshold', xy=(-100, 0.03), color='tab:green', fontsize=self.fontsize)
            # self.axes.annotate('mean of\ndiagonals', xy=(1, -0.5), color='tab:orange', fontsize=self.fontsize)
            # self.axes.annotate(f'{fit_method.replace("my_", "")} curve', xy=(15, 0.6), color='tab:blue',
            #                      fontsize=self.fontsize)
        else:
            function_label = function_name(statistical_function)
            self._activate_legend = True
        self.axes.plot(x_index, statistical_value, '-', label=function_label)
        # self.axes.plot(x_index, statistical_value, ' ')
        self.axes.plot(x_index, new_ydata, '+', label='re-scaled data')
        # self.axes.plot(x_index, new_ydata, ' ')
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

    def plot_merged_2ds(self, ndarray_dict: dict, statistical_func=None, error_band: dict = None):
        self.fig, self.axes = plt.subplots(1, 1, figsize=(2160 / 200, 1080 / 200), dpi=200)
        self.title_prefix += f'with {function_name(statistical_func)}' if statistical_func is not None else ''
        for i, (key, ndarray_data) in enumerate(ndarray_dict.items()):
            # noinspection PyProtectedMember
            color = next(self.axes._get_lines.prop_cycler)['color']
            line_style = next(self.line_styles)
            if statistical_func is not None:
                if isinstance(ndarray_data, list):
                    ndarray_data = np.asarray(ndarray_data)
                self.axes.plot(ndarray_data, line_style, color=color)
                statistical_value = statistical_func(ndarray_data)
                statistical_value_line = np.full(ndarray_data.shape, statistical_value)
                self.axes.plot(statistical_value_line, '--',
                               label=f'{get_algorithm_name(key)}: {statistical_value:.4f}', color=color)
            else:
                self.axes.plot(ndarray_data, line_style, color=color, label=get_algorithm_name(key))

            if self.for_paper:
                self._activate_legend = False
                hard_coded_function = 'weather-2019'
                if (key.startswith('DROPP') and
                        hard_coded_function in ['FooToa-2f4k', 'EV-sim-2f4k', 'FooToa-weather-2019',
                                                'FooToa-FSP', 'EV-2019-rad-dir', 'EV-2019-rad-diff',
                                                'FooToa-weather-2019-rad-dir']):
                    if hard_coded_function == 'FooToa-2f4k':
                        xy = (55, 0.23)
                    elif hard_coded_function == 'EV-sim-2f4k':
                        xy = (60, 0.96)
                    elif hard_coded_function == 'FooToa-weather-2019':
                        xy = (5, 0.05)
                    elif hard_coded_function == 'FooToa-FSP':
                        xy = (10, 0.4)
                    elif hard_coded_function == 'EV-2019-rad-dir':
                        xy = (13, 0.96)
                    elif hard_coded_function == 'EV-2019-rad-diff':
                        xy = (13, 0.95)
                    elif hard_coded_function == 'FooToa-weather-2019-rad-dir':
                        xy = (1.6, 0.3)
                    else:
                        xy = (0, 0)
                elif (key.startswith('TICA') and
                      hard_coded_function in ['EV-2019-rad-dir', 'EV-2019-rad-diff']):
                    if hard_coded_function == 'EV-2019-rad-dir':
                        xy = (0, 0.68)
                    elif hard_coded_function == 'EV-2019-rad-diff':
                        xy = (0, 0.61)
                    else:
                        xy = (0, 0)
                else:
                    xy = self._find_optional_annotating_coordinates(ndarray_data, ndarray_dict)

                if key.startswith('DAANCCER'):
                    self.axes.annotate('DROPP', xy=xy, color=color, fontsize=self.fontsize)
                else:
                    self.axes.annotate(get_algorithm_name(key), xy=xy, color=color, fontsize=self.fontsize)

            else:
                self._activate_legend = True

            if error_band is not None:
                if not (error_band[key].shape[DUMMY_ONE] == ndarray_data.shape[DUMMY_ZERO]):
                    warnings.warn('Could not plot the error band, because the error band has the incorrect shape.')
                else:
                    self.axes.fill_between(range(error_band[key].shape[DUMMY_ONE]),
                                           error_band[key][DUMMY_ZERO], error_band[key][DUMMY_ONE], alpha=0.2)
        self._post_processing()

    @staticmethod
    def _find_optional_annotating_coordinates(current_line_data: np.ndarray, other_line_datas: dict[np.ndarray]):
        def euclidean_distance(point1, point2):
            return np.sqrt((point1 - point2) ** 2)

        max_distance = float('-inf')  # Initialize maximum distance to a small value
        max_index = 0
        for j, data_point in enumerate(current_line_data):
            min_distance_to_others = float('inf')  # Initialize minimum distance to other lines to a large value
            for other_key, other_ndarray_data in other_line_datas.items():
                if not np.array_equal(current_line_data, other_ndarray_data):
                    other_distances = euclidean_distance(data_point, other_ndarray_data)
                    min_distance_to_others = min(min_distance_to_others, np.min(other_distances))
            if min_distance_to_others > max_distance:
                max_distance = min_distance_to_others
                max_index = j

        y_coord = current_line_data[max_index] + .01
        x_coord = max_index + .01
        return x_coord, y_coord

    def plot_matrix_in_2d(self, model_median_scores: dict,
                          x_axis_values: np.ndarray,
                          line_values: np.ndarray,
                          error_band: dict = None):
        self.fig, self.axes = plt.subplots(1, 1, figsize=(2160 / 200, 1080 / 200), dpi=200)

        for model_name, model_scores in model_median_scores.items():
            # noinspection PyProtectedMember
            color = next(self.axes._get_lines.prop_cycler)['color']
            line_style = next(self.line_styles)

            for j, line_width in enumerate(np.log(line_values)):
                dont_plot = []
                if line_values[j] not in dont_plot:
                    self.axes.plot(x_axis_values, model_scores[:, j],
                                   color=color, linestyle=line_style, linewidth=line_width)
                    if j == 0 or line_values[0] in dont_plot:
                        self.axes.plot([], [], color=color, linewidth=10, label=model_name)

                    # self.axes.annotate(line_values[j], xy=(x_axis_values[-1], model_scores[-1, j] - 0.01),
                    #                    color=color, fontsize=self.fontsize)

                    if error_band is not None:
                        error_component_band = error_band[model_name][:, j, :].T
                        if not (error_component_band.shape[DUMMY_ONE] == model_scores[:, j].shape[DUMMY_ZERO]):
                            warnings.warn(
                                'Could not plot the error band, because the error band has the incorrect shape.')
                        else:
                            self.axes.fill_between(x_axis_values,
                                                   error_component_band[DUMMY_ZERO], error_component_band[DUMMY_ONE],
                                                   color=color, alpha=0.2)  # .1 * line_width)

        if self.for_paper:
            self.axes.set_xlim(0, x_axis_values[-1])
            default_xticks = self.axes.get_xticks()
            # Add 105 to the x-ticks
            specific_xticks = [105] + list(default_xticks[1:])

            # Set the x-ticks with both default and specific values
            self.axes.set_xticks(specific_xticks)

        self._activate_legend = True
        self._post_processing()


class MultiArrayPlotter:
    @staticmethod
    def plot_tensor_layers(tensor, combined, title_part='Covariance'):
        for i in range(tensor.shape[0]):  # for each combined dimension
            ArrayPlotter(
                interactive=True,
                title_prefix=f'{ordinal(i)} {title_part} Matrix'
            ).matrix_plot(tensor[i])
        ArrayPlotter(
            interactive=True,
            title_prefix=f'Combined {title_part} Matrix'
        ).matrix_plot(combined)  # and for the mean-ed
