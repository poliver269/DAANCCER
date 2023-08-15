import warnings

import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from research_evaluations.plotter import ArrayPlotter, MultiArrayPlotter
from utils import statistical_zero, ordinal
from utils.algorithms import TensorDR
from utils.errors import NonInvertibleEigenvectorException, InvalidComponentNumberException
from utils.math import is_matrix_orthogonal
from utils.matrix_tools import diagonal_block_expand, calculate_symmetrical_kernel_matrix, ensure_matrix_symmetry
from utils.param_keys import N_COMPONENTS, MATRIX_NDIM, TENSOR_NDIM
from utils.param_keys.analyses import CORRELATION_MATRIX_PLOT, EIGENVECTOR_MATRIX_ANALYSE, COVARIANCE_MATRIX_PLOT
from utils.param_keys.kernel_functions import MY_GAUSSIAN, KERNEL_ONLY, KERNEL_DIFFERENCE, KERNEL_MULTIPLICATION
from utils.param_keys.model import *
from utils.param_keys.traj_dims import TIME_DIM, CORRELATION_DIM, COMBINED_DIM


class DROPP(TensorDR):
    def __init__(self,
                 cov_stat_func=np.mean,
                 algorithm_name='pca',  # pca, tica, kica
                 ndim=TENSOR_NDIM,  # 3: tensor, 2: matrix
                 kernel_kwargs=None,
                 # TODO Kernel kwargs on .json/README fix
                 cov_function=np.cov,  # np.cov, np.corrcoef, co_mad
                 lag_time=0,
                 nth_eigenvector=1,
                 extra_dr_layer=False,
                 abs_eigenvalue_sorting=True,
                 analyse_plot_type=None,
                 use_std=True,
                 center_over_time=True
                 ):
        super().__init__(cov_stat_func)

        self.algorithm_name = algorithm_name
        self.ndim = ndim

        self.kernel_kwargs = {
            KERNEL_MAP: kernel_kwargs.get(KERNEL_MAP, None),  # diff, multi, only, None
            KERNEL_FUNCTION: kernel_kwargs.get(KERNEL_FUNCTION, MY_GAUSSIAN),
            KERNEL_STAT_FUNC: kernel_kwargs.get(KERNEL_STAT_FUNC, statistical_zero),
            USE_ORIGINAL_DATA: kernel_kwargs.get(USE_ORIGINAL_DATA, False),
            CORR_KERNEL: kernel_kwargs.get(CORR_KERNEL, False),
            ONES_ON_KERNEL_DIAG: kernel_kwargs.get(ONES_ON_KERNEL_DIAG, False)
        }

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
            warnings.warn(f'`{NTH_EIGENVECTOR}` parameter (with value: {self.nth_eigenvector}) is ignored '
                          f'since the parameter `{EXTRA_DR_LAYER}` (={self.extra_dr_layer}) has a higher priority.')

        if self._is_time_lagged_algorithm() and self.lag_time == 0:
            warnings.warn(f'The `{ALGORITHM_NAME}` is set to a time-lagged approach: {self.algorithm_name}, '
                          f'but the `{LAG_TIME}` is not set is equal to: {self.lag_time}')

        if isinstance(self.kernel_kwargs[KERNEL_STAT_FUNC], str):
            self.kernel_kwargs[KERNEL_STAT_FUNC] = eval(self.kernel_kwargs[KERNEL_STAT_FUNC])

    def __str__(self):
        sb = f'DROPP('
        # sb = f'DROPP+{self.kernel_type.replace("my_", "")}('
        # sb = self.describe()
        sb += f'PCs={self.n_components}'
        sb += f',lag-time={self.lag_time}, ' if self.lag_time > 0 else ''
        # sb += '\n'
        # sb += f'abs_ew_sorting={self.abs_eigenvalue_sorting}, '
        return sb + ')'
        # f'{function_name(self.cov_function)}'

    def describe(self):
        """
        Short version of __str__
        :return: Short description of the model
        """
        sb = f'{"Matrix" if self._is_matrix_model else "Tensor"}-{self.algorithm_name}'
        if self.kernel_kwargs[KERNEL_MAP] is not None:
            sb += (f', {self.kernel_kwargs[KERNEL_FUNCTION]}-{self.kernel_kwargs[KERNEL_MAP]}' +
                   f'{f"-onCorr2" if self.kernel_kwargs[CORR_KERNEL] else ""}')
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
        """
        The _combine_dim is the size of the (3rd) dimension from the tensor
        which will be combined after calculating the covariance matrix separately.
        :return: int
            size of combined dimension
        """
        return self._standardized_data.shape[COMBINED_DIM]

    @property
    def _correlation_dim(self) -> int:
        """
        The _correlation_dim is the size of the (2nd) dimension from the tensor
        which is the size of the correlated features of the data
        :return: int
            size of the correlation dimension
        """
        return self._standardized_data.shape[CORRELATION_DIM]

    def fit_transform(self, data_tensor, **fit_params):
        return super().fit_transform(data_tensor, **fit_params)

    def fit(self, data_tensor, **fit_params):
        self.n_samples = data_tensor.shape[TIME_DIM]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        eigenvectors = self.get_eigenvectors()
        self.components_ = eigenvectors[:, :self.n_components].T
        if self.analyse_plot_type == EIGENVECTOR_MATRIX_ANALYSE:
            ArrayPlotter(
                interactive=False,
                title_prefix=EIGENVECTOR_MATRIX_ANALYSE,
                x_label='Eigenvector Number',
                y_label='Eigenvector Dimension',
                xtick_start=1,
                for_paper=True
            ).matrix_plot(eigenvectors[:12, :15], show_values=True)
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

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Returns the covariance matrix.
        The calculation depends on the ndim of the model.
            If the input is a matrix the covariance matrix is calculated as usual.
            Else the covariance matrix is calculated over the combined dimension span separately
            and forced the not correlated values not to correlate by block expanding it
            setting the values to zeros.
        The DROPP algorithm maps a gaussian curve onto the covariance matrix in default
        :return: np.ndarray
            Covariance matrix shape: _correlation_dim*_combined_dim x _correlation_dim*_combined_dim
            (or _correlation_dim x _correlation_dim, if _is_matrix_model)
        """
        if self._is_matrix_model:
            cov = self._get_matrix_covariance()
            if self.kernel_kwargs[KERNEL_MAP] is not None and not self._use_kernel_as_correlations_matrix():
                cov = self._map_kernel(cov)
            return cov
        else:
            ccm = self.get_combined_covariance_matrix()
            if self.kernel_kwargs[KERNEL_MAP] is not None and not self._use_kernel_as_correlations_matrix():
                ccm = self._map_kernel(ccm)
            return diagonal_block_expand(ccm, self._combine_dim)

    def _get_matrix_covariance(self) -> np.ndarray:
        """
        Get the covariance matrix for a _standardized_data in matrix shape.
        If the algorithm is time-lagged, then the lag_time is truncated from the matrix.
        Else the standard covariance matrix is used.
        :return: np.ndarray
        """
        if self._is_time_lagged_algorithm() and self.lag_time > 0:
            return np.cov(self._standardized_data[:-self.lag_time].T)
        else:
            return super().get_covariance_matrix()

    def get_combined_covariance_matrix(self):
        """

        :return:
        """
        tensor_cov = self.get_tensor_covariance()
        cov = self.cov_stat_func(tensor_cov, axis=0)
        if self.analyse_plot_type == COVARIANCE_MATRIX_PLOT:
            MultiArrayPlotter().plot_tensor_layers(tensor_cov, cov, 'Covariance')
        return cov

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
            matrix,
            flattened=self._is_matrix_model,
            analyse_mode=self.analyse_plot_type,
            **self.kernel_kwargs
        )
        if self.kernel_kwargs[KERNEL_MAP] == KERNEL_ONLY:
            matrix = kernel_matrix
        elif self.kernel_kwargs[KERNEL_MAP] == KERNEL_DIFFERENCE:
            matrix -= kernel_matrix
        elif self.kernel_kwargs[KERNEL_MAP] == KERNEL_MULTIPLICATION:
            matrix *= kernel_matrix

        if self.kernel_kwargs[ONES_ON_KERNEL_DIAG]:
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
        self.explained_variance_ = np.real_if_close(eigenvalues[sorted_eigenvalue_indexes])
        eigenvectors = np.real_if_close(eigenvectors[:, sorted_eigenvalue_indexes])

        if self.extra_dr_layer:
            return self._get_eigenvectors_with_dr_layer(eigenvectors)
        else:
            self.explained_variance_ = self.explained_variance_[::self.nth_eigenvector]
            return eigenvectors[:, ::self.nth_eigenvector]

    def _get_eigenvectors_with_dr_layer(self, eigenvectors):
        eigenvalues2 = []
        eigenvectors2 = []
        for component in range(self._correlation_dim):
            vector_from = component * self._combine_dim
            vector_to = (component + 1) * self._combine_dim
            model = PCA(n_components=1)
            model.fit_transform(eigenvectors[:, vector_from:vector_to])

            # No idea if it makes sense TODO
            ew2 = model.explained_variance_[0]
            # eigenvalues2.append(np.mean(self.eigenvalues[vector_from:vector_to] * ew2))
            eigenvalues2.append(np.sum(self.explained_variance_[vector_from:vector_to] * ew2))

            ev2 = model.components_[0]
            ev = np.dot(eigenvectors[:, vector_from:vector_to], ev2)
            eigenvectors2.append(ev)

        self.explained_variance_ = np.asarray(eigenvalues2).T
        return np.asarray(eigenvectors2).T

    def _get_correlations_matrix(self):
        if self._is_matrix_model:
            corr = self._get_matrix_correlation()

            if self.kernel_kwargs[CORR_KERNEL] or self._use_kernel_as_correlations_matrix():
                corr = self._map_kernel(corr)

            return corr
        else:
            tensor_corr = self._get_tensor_correlation()
            corr = self.cov_stat_func(tensor_corr, axis=0)

            if self.analyse_plot_type == CORRELATION_MATRIX_PLOT:
                MultiArrayPlotter().plot_tensor_layers(tensor_corr, corr, 'Correlation')

            if self.kernel_kwargs[CORR_KERNEL] or self._use_kernel_as_correlations_matrix():
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
            for index in range(self._combine_dim):
                dot_i = np.dot(self._standardized_data[:-self.lag_time, :, index].T,
                               self._standardized_data[self.lag_time:, :, index]) / (
                                self.n_samples - self.lag_time)
                sym_i = ensure_matrix_symmetry(dot_i)
                temp_list.append(sym_i)
            return np.asarray(temp_list)

    def transform(self, data_tensor):
        data_tensor_standardized = self._standardize_data(data_tensor)
        data_matrix = self.convert_to_matrix(data_tensor_standardized)
        return np.dot(data_matrix, self.components_.T)

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
        if is_matrix_orthogonal(self.components_.T):
            return np.dot(
                projection_data,
                self.components_[:component_count]
            )  # Da orthogonal --> Transform = inverse
        else:
            if self.use_evs:
                raise NonInvertibleEigenvectorException('Eigenvectors are Non-Orthogonal and Non-Squared. ')
            else:
                return np.dot(
                    projection_data,
                    np.linalg.inv(self.components_.T)[:component_count]
                )

    def reconstruct(self, projection_matrix, component_count=None):
        if component_count is None:
            component_count = self._correlation_dim * self._combine_dim
        elif component_count > self.components_.shape[0]:
            raise InvalidComponentNumberException(f'Model does not have {component_count} many components. '
                                                  f'Max: {self.components_.shape[0]}')

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
