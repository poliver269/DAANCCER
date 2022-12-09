import numpy as np
import scipy

from plotter import ArrayPlotter
from utils.algorithms import MyModel
from utils.math import is_matrix_symmetric
from utils.matrix_tools import diagonal_block_expand, calculate_symmetrical_kernel_from_matrix
from utils.param_key import *


class TensorDR(MyModel):
    def __init__(self, model_parameters=None):
        super().__init__()
        if model_parameters is None:
            model_parameters = {}

        self.params = {
            COV_STAT_FUNC: model_parameters.get(COV_STAT_FUNC, np.mean),
            KERNEL_STAT_FUNC: model_parameters.get(KERNEL_STAT_FUNC, np.median)
        }

    def fit(self, data_tensor):
        self.n_samples = data_tensor.shape[0]
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self._update_cov()
        self.eigenvectors = self.get_eigenvectors()
        return self

    def get_covariance_matrix(self):
        return super().get_covariance_matrix()

    def _update_cov(self):
        averaged_cov = self.params[COV_STAT_FUNC](self._covariance_matrix, axis=0)
        self._covariance_matrix = diagonal_block_expand(averaged_cov, self._covariance_matrix.shape[0])

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]

    def transform(self, data_tensor, n_components):
        data_matrix = self._update_data_tensor(data_tensor)
        return super(TensorDR, self).transform(data_matrix, n_components)

    @staticmethod
    def _update_data_tensor(data_tensor):
        return data_tensor.reshape(data_tensor.shape[0], data_tensor.shape[1] * data_tensor.shape[2])


class ParameterModel(TensorDR):
    def __init__(self, model_parameters):
        super().__init__(model_parameters)
        self.params.update({
            ALGORITHM_NAME: model_parameters.get(ALGORITHM_NAME, 'pca'),  # pca, tica
            NDIM: model_parameters.get(NDIM, TENSOR_NDIM),  # 3: tensor, 2: matrix

            KERNEL: model_parameters.get(KERNEL, None),  # diff, multi, only, None
            CORR_KERNEL: model_parameters.get(CORR_KERNEL, False),  # True, False
            KERNEL_TYPE: model_parameters.get(KERNEL_TYPE, MY_GAUSSIAN),
            ONES_ON_KERNEL_DIAG: model_parameters.get(ONES_ON_KERNEL_DIAG, False),

            COV_FUNCTION: model_parameters.get(COV_FUNCTION, np.cov),  # np.cov, np.corrcoef, co_mad
            LAG_TIME: model_parameters.get(LAG_TIME, 0),
            NTH_EIGENVECTOR: model_parameters.get(NTH_EIGENVECTOR, 1),

            PLOT_2D: model_parameters.get(PLOT_2D, False),
            USE_STD: model_parameters.get(USE_STD, False),
            CENTER_OVER_TIME: model_parameters.get(CENTER_OVER_TIME, True),
        })

    def __str__(self):
        sb = f'{"Matrix" if self.__is_matrix_model else "Tensor"}-{self.params[ALGORITHM_NAME]}'
        if self.params[KERNEL] is not None:
            sb += f', {self.params[KERNEL_TYPE]}-{self.params[KERNEL]}{f"-onCorr2" if self.params[CORR_KERNEL] else ""}'
        sb += f'{f", lag-time={self.params[LAG_TIME]}" if self.params[LAG_TIME] > 0 else ""}'
        return sb
        # f'{function_name(self.params[COV_FUNCTION])}'

    @property
    def __is_matrix_model(self) -> bool:
        return self.params[NDIM] == 2

    def fit_transform(self, data_tensor, n_components=2):
        return super().fit_transform(data_tensor, n_components)

    def fit(self, data_tensor):
        self.n_samples = data_tensor.shape[0]
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self.eigenvectors = self.get_eigenvectors()
        return self

    def _standardize_data(self, tensor):
        if self.__is_matrix_model or not self.params[CENTER_OVER_TIME]:
            numerator = tensor - np.mean(tensor, axis=0)
        else:
            numerator = tensor - np.mean(tensor, axis=1)[:, np.newaxis, :]

        if self.params[USE_STD]:
            denominator = np.std(tensor, axis=0)
            return numerator / denominator
        else:
            return numerator

    def get_covariance_matrix(self):
        if self.__is_matrix_model:
            cov = self._get_matrix_covariance()
            if self.params[KERNEL] is not None:
                cov = self._map_kernel(cov)
            return cov
        else:
            tensor_cov = self._get_tensor_covariance()
            cov = self.params[COV_STAT_FUNC](tensor_cov, axis=0)
            if self.params[KERNEL] is not None:
                cov = self._map_kernel(cov)
            return diagonal_block_expand(cov, tensor_cov.shape[0])

    def _get_matrix_covariance(self):
        if self.params[ALGORITHM_NAME] in ['tica'] and self.params[LAG_TIME] > 0:
            return np.cov(self._standardized_data[:-self.params[LAG_TIME]].T)
        else:
            return super().get_covariance_matrix()

    def _get_tensor_covariance(self):
        if self.params[ALGORITHM_NAME] in ['tica'] and self.params[LAG_TIME] > 0:
            return np.asarray(list(
                map(lambda index: self.params[COV_FUNCTION](
                    self._standardized_data[:-self.params[LAG_TIME], :, index].T),
                    range(self._standardized_data.shape[2]))
            ))
        else:
            return np.asarray(list(
                map(lambda index: self.params[COV_FUNCTION](self._standardized_data[:, :, index].T),
                    range(self._standardized_data.shape[2]))
            ))

    def _map_kernel(self, matrix):
        trajectory_name = 'Not Model Related' if self.params[PLOT_2D] else None
        kernel_matrix = calculate_symmetrical_kernel_from_matrix(
            matrix, self.params[KERNEL_STAT_FUNC], self.params[KERNEL_TYPE],
            trajectory_name, flattened=self.__is_matrix_model)
        if self.params[KERNEL] == KERNEL_ONLY:
            matrix = kernel_matrix
        elif self.params[KERNEL] == KERNEL_DIFFERENCE:
            matrix -= kernel_matrix
        elif self.params[KERNEL] == KERNEL_MULTIPLICATION:
            matrix *= kernel_matrix

        if self.params[ONES_ON_KERNEL_DIAG]:
            np.fill_diagonal(matrix, 1.0)

        return matrix

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        # assert is_matrix_symmetric(self._covariance_matrix), 'Covariance-Matrix should be symmetric.'
        if self.params[ALGORITHM_NAME] in ['tica']:
            correlation_matrix = self._get_correlations_matrix()
            # assert is_matrix_symmetric(correlation_matrix), 'Correlation-Matrix should be symmetric.'
            eigenvalues, eigenvectors = scipy.linalg.eig(correlation_matrix, b=self._covariance_matrix)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = np.real_if_close(eigenvalues[sorted_eigenvalue_indexes])
        if np.any(np.iscomplex(self.eigenvalues)):
            self.eigenvalues = np.abs(self.eigenvalues)
        return np.real_if_close(eigenvectors[:, sorted_eigenvalue_indexes])

    def _get_correlations_matrix(self):
        if self.__is_matrix_model:
            corr = self._get_matrix_correlation()

            if self.params[CORR_KERNEL]:
                corr = self._map_kernel(corr)

            return corr
        else:
            tensor_corr = self._get_tensor_correlation()
            corr = self.params[COV_STAT_FUNC](tensor_corr, axis=0)

            if self.params[PLOT_2D]:
                for i in range(tensor_corr.shape[0]):
                    ArrayPlotter(False).matrix_plot(tensor_corr[i])
                ArrayPlotter(False).matrix_plot(corr)

            if self.params[CORR_KERNEL]:
                corr = self._map_kernel(corr)

            return diagonal_block_expand(corr, tensor_corr.shape[0])

    def _get_matrix_correlation(self):
        if self.params[LAG_TIME] <= 0:
            return self._get_matrix_covariance()
        else:
            corr = np.dot(self._standardized_data[:-self.params[LAG_TIME]].T,
                          self._standardized_data[self.params[LAG_TIME]:]) / (self.n_samples - self.params[LAG_TIME])
            return 0.5 * (corr + corr.T)

    def _get_tensor_correlation(self):
        if self.params[LAG_TIME] <= 0:
            return self._get_tensor_covariance()
        else:
            temp_list = []
            for index in range(self._standardized_data.shape[COORDINATE_DIM]):
                dot_i = np.dot(self._standardized_data[:-self.params[LAG_TIME], :, index].T,
                               self._standardized_data[self.params[LAG_TIME]:, :, index]) / (
                                self.n_samples - self.params[LAG_TIME])
                sym_i = 0.5 * (dot_i + dot_i.T)
                temp_list.append(sym_i)
            return np.asarray(temp_list)

    def transform(self, data_tensor, n_components):
        if self.__is_matrix_model:
            data_matrix = data_tensor
        else:
            data_matrix = self._update_data_tensor(data_tensor)

        self.n_components = n_components
        if self.params[NTH_EIGENVECTOR] < 1:
            self.params[NTH_EIGENVECTOR] = data_tensor.shape[2]

        return np.dot(
            data_matrix,
            self.eigenvectors[:, :self.n_components * self.params[NTH_EIGENVECTOR]:self.params[NTH_EIGENVECTOR]]
        )
