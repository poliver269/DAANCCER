import warnings

import numpy as np
import scipy
from pyemma import coordinates as coor
from sklearn.metrics import mean_squared_error

from plotter import ArrayPlotter
from utils import statistical_zero, ordinal
from utils.algorithms.tensor_dim_reductions import TensorDR
from utils.errors import NonInvertibleEigenvectorException, InvalidComponentNumberException
from utils.math import is_matrix_orthogonal
from utils.matrix_tools import diagonal_block_expand, calculate_symmetrical_kernel_matrix, ensure_matrix_symmetry
from utils.param_key import *


class DAANCCER(TensorDR):
    def __init__(self,
                 cov_stat_func=np.mean,
                 kernel_stat_func=statistical_zero,
                 algorithm_name='pca',  # pca, tica, kica
                 ndim=TENSOR_NDIM,  # 3: tensor, 2: matrix
                 kernel=None,  # diff, multi, only, None
                 corr_kernel=False,  # only for tica: True, False
                 kernel_type=MY_GAUSSIAN,
                 ones_on_kernel_diag=False,
                 cov_function=np.cov,  # np.cov, np.corrcoef, co_mad
                 lag_time=0,
                 nth_eigenvector=1,
                 extra_dr_layer=False,
                 abs_eigenvalue_sorting=True,
                 analyse_plot_type=None,
                 use_std=True,
                 center_over_time=True
                 ):
        super().__init__(cov_stat_func, kernel_stat_func)

        self.algorithm_name = algorithm_name
        self.ndim = ndim

        self.kernel = kernel
        self.kernel_type = kernel_type
        self.corr_kernel = corr_kernel
        self.ones_on_kernel_diag = ones_on_kernel_diag

        self.cov_function = cov_function
        self.lag_time = lag_time

        self.nth_eigenvector = nth_eigenvector
        self.extra_dr_layer = extra_dr_layer

        self.abs_eigenvalue_sorting = abs_eigenvalue_sorting
        self.analyse_plot_type = analyse_plot_type
        self.use_std = use_std
        self.center_over_time = center_over_time
        self.__check_init_params__()

    def __check_init_params__(self):
        if self.nth_eigenvector < 1:
            self.nth_eigenvector = 1

        if self.extra_dr_layer and self.nth_eigenvector > 1:
            warnings.warn(f'`{NTH_EIGENVECTOR}` parameter is ignored '
                          f'since the parameter `{EXTRA_DR_LAYER}` has a higher priority.')

    def __str__(self):
        sb = self.describe()
        sb += f'\nPCs={self.n_components}'
        sb += f'lag-time={self.lag_time}, ' if self.lag_time > 0 else ''
        # sb += '\n'
        # sb += f'abs_ew_sorting={self.abs_eigenvalue_sorting}, '
        return sb
        # f'{function_name(self.cov_function)}'

    def describe(self):
        """
        Short version of __str__
        :return: Short description of the model
        """
        sb = f'{"Matrix" if self._is_matrix_model else "Tensor"}-{self.algorithm_name}'
        if self.kernel is not None:
            sb += (f', {self.kernel_type}-{self.kernel}' +
                   f'{f"-onCorr2" if self.corr_kernel else ""}')
            if self.use_evs:
                if self.extra_dr_layer:
                    sb += f'-2nd_layer_eevd'
                else:
                    sb += f'-{ordinal(self.nth_eigenvector)}_ev_eevd'
        return sb

    @property
    def _is_matrix_model(self) -> bool:
        return self.ndim == MATRIX_NDIM

    def _is_time_lagged_algorithm(self) -> bool:
        return self.algorithm_name in ['tica']

    def _use_kernel_as_correlations_matrix(self) -> bool:
        return self.algorithm_name in ['kica']

    def _use_correlations_matrix(self) -> bool:
        return self._use_kernel_as_correlations_matrix() or self._is_time_lagged_algorithm()

    @property
    def use_evs(self) -> bool:
        """
        EVS: EigenVector Selection
        :return:
        """
        return self.extra_dr_layer or self.nth_eigenvector > 1

    @property
    def _combine_dim(self) -> int:
        return self._standardized_data.shape[COORDINATE_DIM]

    @property
    def _atom_dim(self) -> int:
        return self._standardized_data.shape[ATOM_DIM]

    def fit_transform(self, data_tensor, **fit_params):
        return super().fit_transform(data_tensor, **fit_params)

    def fit(self, data_tensor, **fit_params):
        self.n_samples = data_tensor.shape[TIME_DIM]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self.eigenvectors = self.get_eigenvectors()
        if self.analyse_plot_type == EIGENVECTOR_MATRIX_ANALYSE:
            ArrayPlotter(
                interactive=False,
                title_prefix=EIGENVECTOR_MATRIX_ANALYSE,
                x_label='Eigenvector Number',
                y_label='Eigenvector Dimension',
                xtick_start=1,
                for_paper=True
            ).matrix_plot(self.eigenvectors[:12, :15], show_values=True)
        return self

    def _standardize_data(self, tensor):
        centered_data = self._center_data(tensor)

        if self.use_std:
            self._std = np.std(tensor, axis=0)
        else:
            self._std = 1
        return centered_data / self._std

    def _center_data(self, tensor):
        """
        Center the data by subtracting the mean vector
        :param tensor: Input data to center
        :return: centered data tensor or matrix
        """
        if self._is_matrix_model or not self.center_over_time:
            self.mean = np.mean(tensor, axis=0)
        else:
            # self.mean = np.mean(tensor, axis=1)[:, np.newaxis, :]
            self.mean = np.mean(tensor, axis=0)[np.newaxis, :, :]
        return tensor - self.mean

    def _standardize(self, tensor):
        tensor -= self.mean
        # tensor -= np.mean(tensor, axis=0)[np.newaxis, :, :]
        tensor /= self._std  # fitted std
        # tensor /= np.std(tensor, axis=0)  # original std
        return tensor

    def get_covariance_matrix(self):
        if self._is_matrix_model:
            cov = self._get_matrix_covariance()
            if self.kernel is not None and not self._use_kernel_as_correlations_matrix():
                cov = self._map_kernel(cov)
            return cov
        else:
            ccm = self.get_combined_covariance_matrix()
            if self.kernel is not None and not self._use_kernel_as_correlations_matrix():
                ccm = self._map_kernel(ccm)
            return diagonal_block_expand(ccm, self._combine_dim)

    def _get_matrix_covariance(self):
        if self._is_time_lagged_algorithm() and self.lag_time > 0:
            return np.cov(self._standardized_data[:-self.lag_time].T)
        else:
            return super().get_covariance_matrix()

    def get_combined_covariance_matrix(self):
        tensor_cov = self.get_tensor_covariance()
        return self.cov_stat_func(tensor_cov, axis=0)

    def get_tensor_covariance(self):
        if self._is_time_lagged_algorithm() and self.lag_time > 0:
            return np.asarray(list(
                map(lambda index: self.cov_function(
                    self._standardized_data[:-self.lag_time, :, index].T),
                    range(self._combine_dim))
            ))
        else:
            return np.asarray(list(
                map(lambda index: self.cov_function(self._standardized_data[:, :, index].T),
                    range(self._combine_dim))
            ))

    def _map_kernel(self, matrix):
        kernel_matrix = calculate_symmetrical_kernel_matrix(
            matrix, self.kernel_stat_func, self.kernel_type,
            analyse_mode=self.analyse_plot_type, flattened=self._is_matrix_model)
        if self.kernel == KERNEL_ONLY:
            matrix = kernel_matrix
        elif self.kernel == KERNEL_DIFFERENCE:
            matrix -= kernel_matrix
        elif self.kernel == KERNEL_MULTIPLICATION:
            matrix *= kernel_matrix

        if self.ones_on_kernel_diag:
            np.fill_diagonal(matrix, 1.0)

        return matrix

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        # assert is_matrix_symmetric(self._covariance_matrix), 'Covariance-Matrix should be symmetric.'
        if self.algorithm_name in ['tica', 'kica']:
            correlation_matrix = self._get_correlations_matrix()
            # assert is_matrix_symmetric(correlation_matrix), 'Correlation-Matrix should be symmetric.'
            # noinspection PyTupleAssignmentBalance
            eigenvalues, eigenvectors = scipy.linalg.eig(correlation_matrix, b=self._covariance_matrix)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        if self.abs_eigenvalue_sorting:
            eigenvalues = np.abs(eigenvalues)

        if np.any(np.iscomplex(eigenvalues)):
            eigenvalues = np.abs(eigenvalues)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = np.real_if_close(eigenvalues[sorted_eigenvalue_indexes])
        eigenvectors = np.real_if_close(eigenvectors[:, sorted_eigenvalue_indexes])

        if self.extra_dr_layer:
            return self._get_eigenvectors_with_dr_layer(eigenvectors)
        else:
            self.eigenvalues = self.eigenvalues[::self.nth_eigenvector]
            return eigenvectors[:, ::self.nth_eigenvector]

    def _get_eigenvectors_with_dr_layer(self, eigenvectors):
        eigenvalues2 = []
        eigenvectors2 = []
        for component in range(self._atom_dim):
            vector_from = component * self._combine_dim
            vector_to = (component + 1) * self._combine_dim
            model = coor.pca(data=eigenvectors[:, vector_from:vector_to], dim=1)

            # No idea if it makes sense TODO
            ew2 = model.eigenvalues[0]
            # eigenvalues2.append(np.mean(self.eigenvalues[vector_from:vector_to] * ew2))
            eigenvalues2.append(np.sum(self.eigenvalues[vector_from:vector_to] * ew2))

            ev2 = model.eigenvectors[0]
            ev = np.dot(eigenvectors[:, vector_from:vector_to], ev2)
            eigenvectors2.append(ev)

        self.eigenvalues = np.asarray(eigenvalues2).T
        return np.asarray(eigenvectors2).T

    def _get_correlations_matrix(self):
        if self._is_matrix_model:
            corr = self._get_matrix_correlation()

            if self.corr_kernel or self._use_kernel_as_correlations_matrix():
                corr = self._map_kernel(corr)

            return corr
        else:
            tensor_corr = self._get_tensor_correlation()
            corr = self.cov_stat_func(tensor_corr, axis=0)

            if self.analyse_plot_type == CORRELATION_MATRIX_PLOT:
                for i in range(tensor_corr.shape[0]):  # for each axis
                    ArrayPlotter(interactive=False).matrix_plot(tensor_corr[i])
                ArrayPlotter(interactive=False).matrix_plot(corr)  # and for the mean-ed

            if self.corr_kernel or self._use_kernel_as_correlations_matrix():
                corr = self._map_kernel(corr)

            return diagonal_block_expand(corr, tensor_corr.shape[0])

    def _get_matrix_correlation(self):
        if self.lag_time <= 0:
            return self._get_matrix_covariance()
        else:
            corr = np.dot(self._standardized_data[:-self.lag_time].T,
                          self._standardized_data[self.lag_time:]) / (self.n_samples - self.lag_time)
            return ensure_matrix_symmetry(corr)

    def _get_tensor_correlation(self):
        if self._use_kernel_as_correlations_matrix() or self.lag_time <= 0:
            return self.get_tensor_covariance()
        else:
            temp_list = []
            for index in range(self._standardized_data.shape[COORDINATE_DIM]):
                dot_i = np.dot(self._standardized_data[:-self.lag_time, :, index].T,
                               self._standardized_data[self.lag_time:, :, index]) / (
                                self.n_samples - self.lag_time)
                sym_i = ensure_matrix_symmetry(dot_i)
                temp_list.append(sym_i)
            return np.asarray(temp_list)

    def transform(self, data_tensor):
        data_tensor_standardized = self._standardize_data(data_tensor)
        data_matrix = self.convert_to_matrix(data_tensor_standardized)
        return np.dot(data_matrix, self.eigenvectors[:, :self.n_components])

    def convert_to_matrix(self, tensor):
        if self._is_matrix_model:
            return tensor
        else:
            return super().convert_to_matrix(tensor)

    def convert_to_tensor(self, matrix):
        if self._is_matrix_model:
            return matrix
        else:
            return super().convert_to_tensor(matrix)

    def inverse_transform(self, projection_data: np.ndarray, component_count: int):
        if is_matrix_orthogonal(self.eigenvectors):
            return np.dot(
                projection_data,
                self.eigenvectors[:, :component_count].T
            )  # Da orthogonal --> Transform = inverse
        else:
            if self.use_evs:
                raise NonInvertibleEigenvectorException('Eigenvectors are Non-Orthogonal and Non-Squared. ')
            else:
                return np.dot(
                    projection_data,
                    np.linalg.inv(self.eigenvectors)[:component_count]
                )

    def reconstruct(self, projection_matrix, component_count=None):
        if component_count is None:
            component_count = self._atom_dim * self._combine_dim
        elif component_count > self.eigenvectors.shape[1]:
            raise InvalidComponentNumberException(f'Model does not have {component_count} many components. '
                                                  f'Max: {self.eigenvectors.shape[1]}')

        inverse_matrix = self.inverse_transform(projection_matrix, component_count)

        reconstructed_tensor = self.convert_to_tensor(inverse_matrix)
        if self.use_std:
            reconstructed_tensor *= self._std
        reconstructed_tensor += self.mean

        return reconstructed_tensor

    def score(self, data_tensor, y=None):
        """
        Reconstruct data and calculate the root mean squared error (RMSE).
        See Notes: https://stats.stackexchange.com/q/229093

        Parameters
        ----------
        data_tensor : (X) array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """

        if y is not None:  # "use" variable, to not have a PyCharm warning
            data_projection = y
        else:
            data_projection = self.transform(data_tensor)

        reconstructed_tensor = self.reconstruct(data_projection, self.n_components)

        data_matrix = self.convert_to_matrix(data_tensor)
        reconstructed_matrix = self.convert_to_matrix(reconstructed_tensor)

        return mean_squared_error(data_matrix, reconstructed_matrix, squared=False)