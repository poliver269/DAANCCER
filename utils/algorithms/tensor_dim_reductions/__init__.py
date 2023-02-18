import numpy as np
import scipy
from sklearn.metrics import mean_squared_error

from plotter import ArrayPlotter
from utils.algorithms import MyModel
import pyemma.coordinates as coor

from utils.matrix_tools import diagonal_block_expand, calculate_symmetrical_kernel_matrix, ensure_matrix_symmetry
from utils.param_key import *


class TensorDR(MyModel):
    def __init__(self, cov_stat_func=np.mean, kernel_stat_func=np.median):
        super().__init__()

        self.cov_stat_func = cov_stat_func
        self.kernel_stat_func = kernel_stat_func

    def fit(self, data_tensor, **fit_params):
        self.n_samples = data_tensor.shape[TIME_DIM]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self._update_cov()
        self.eigenvectors = self.get_eigenvectors()
        return self

    def get_covariance_matrix(self):
        return super().get_covariance_matrix()

    def _update_cov(self):
        averaged_cov = self.cov_stat_func(self._covariance_matrix, axis=0)
        self._covariance_matrix = diagonal_block_expand(averaged_cov, self._covariance_matrix.shape[0])

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]

    def transform(self, data_tensor):
        data_matrix = self.convert_to_matrix(data_tensor)
        return super(TensorDR, self).transform(data_matrix)

    @staticmethod
    def convert_to_matrix(tensor):
        return tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])


class ParameterModel(TensorDR):
    def __init__(self,
                 cov_stat_func=np.mean,
                 kernel_stat_func=np.median,
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
                 analyze_plot_type=None,
                 use_std=False,
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
        self.analyze_plot_type = analyze_plot_type
        self.use_std = use_std
        self.center_over_time = center_over_time

    def __str__(self):
        sb = self.describe()
        sb += f' PCs={self.n_components}'
        sb += '\n'
        sb += f'lag-time={self.lag_time}, ' if self.lag_time > 0 else ''
        # sb += f'abs_ew_sorting={self.abs_eigenvalue_sorting}, '
        sb += f'2nd_layer={self.extra_dr_layer}' if self.extra_dr_layer else ''
        sb += f'\nn-th_ev={self.nth_eigenvector}' if self.nth_eigenvector > 1 else ''
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
        return self

    def _standardize_data(self, tensor):
        numerator = self._center_data(tensor)

        if self.use_std:
            denominator = np.std(tensor, axis=0)
            return numerator / denominator
        else:
            return numerator

    def _center_data(self, tensor):
        """
        Center the data by subtracting the mean vector
        :param tensor: Input data to center
        :return: centered data tensor or matrix
        """
        if self._is_matrix_model or not self.center_over_time:
            return tensor - np.mean(tensor, axis=0)
        else:
            return tensor - np.mean(tensor, axis=1)[:, np.newaxis, :]

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
            analyse_mode=self.analyze_plot_type, flattened=self._is_matrix_model)
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
            return eigenvectors

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

            if self.analyze_plot_type:  # plot the correlation matrix
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

        if self.nth_eigenvector < 1:
            self.nth_eigenvector = self._combine_dim

        if self.extra_dr_layer:
            return np.dot(data_matrix, self.eigenvectors[:, :self.n_components])
            # return self.transform_with_extra_layer(data_matrix)
        else:
            return np.dot(
                data_matrix,
                self.eigenvectors[:, :self.n_components * self.nth_eigenvector:self.nth_eigenvector]
            )

    def convert_to_matrix(self, tensor):
        if self._is_matrix_model:
            return tensor
        else:
            return super().convert_to_matrix(tensor)

    def inverse_transform(self, projection_data: np.ndarray):
        return self.inverse_transform_definite(projection_data, self.n_components)

    def inverse_transform_definite(self, projection_data: np.ndarray, inv_component: int):
        return np.dot(
                projection_data,
                self.eigenvectors[:, :inv_component * self.nth_eigenvector:self.nth_eigenvector].T
            )  # Da orthogonal --> Transform = inverse

    def score(self, data_tensor, y=None):
        """
        Reconstruct data and calculate the root mean squared error (RMSE).
        See Notes: https://stats.stackexchange.com/q/229093

        Parameters
        ----------
        data_tensor : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """

        if y is not None:  # "use" variable, to not have a warning
            data_projection = y
        else:
            data_projection = self.transform(data_tensor)

        reconstructed_matrix_data = self.inverse_transform(data_projection)

        data_matrix = self.convert_to_matrix(data_tensor)

        reconstructed_matrix_data += np.mean(data_matrix, axis=0)

        return mean_squared_error(data_matrix, reconstructed_matrix_data, squared=False)
        # R2 Score probably not good for tensor data... Negative values for the scoring implies bad results.
        # return r2_score(data_matrix, reconstructed_matrix_data)
