import numpy as np
import scipy

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

    def get_covariance_matrix(self):
        return

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
            ALGORITHM_NAME: model_parameters.get(ALGORITHM_NAME, 'pca'),  # pc, tica
            NDIM: model_parameters.get(NDIM, TENSOR_NDIM),  # 3: tensor, 2: matrix  # TODO implement for 2-ndim
            KERNEL: model_parameters.get(KERNEL, None),  # diff, multi, only, None
            KERNEL_TYPE: model_parameters.get(KERNEL_TYPE, MY_GAUSSIAN),
            COV_FUNCTION: model_parameters.get(COV_FUNCTION, np.cov),  # np.cov, np.corrcoef, co_mad
            PLOT_2D_GAUSS: model_parameters.get(PLOT_2D_GAUSS, False),
            LAG_TIME: model_parameters.get(LAG_TIME, 0),
            USE_STD: model_parameters.get(USE_STD, False),
            CENTER_OVER_TIME: model_parameters.get(CENTER_OVER_TIME, True)
        })

    def __str__(self):
        return (f'{"Tensor" if self.params[NDIM] == 3 else "Matrix"}-{self.params[ALGORITHM_NAME]}'
                f'{f", {self.params[KERNEL_TYPE]}-{self.params[KERNEL]}" if self.params[KERNEL] is not None else ""}'
                f'{f", lag-time={self.params[LAG_TIME]}" if self.params[LAG_TIME] > 0 else ""}')
        # f'{function_name(self.params[COV_FUNCTION])}'

    def fit_transform(self, data_tensor, n_components=2):
        return super().fit_transform(data_tensor, n_components)

    def fit(self, data_tensor):
        self.n_samples = data_tensor.shape[0]
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self.eigenvectors = self.get_eigenvectors()

    def _standardize_data(self, tensor):
        if self.params[CENTER_OVER_TIME]:
            numerator = tensor - np.mean(tensor, axis=1)[:, np.newaxis, :]
        else:
            numerator = tensor - np.mean(tensor, axis=0)

        if self.params[USE_STD]:
            denominator = np.std(tensor, axis=0)
            return numerator / denominator
        else:
            return numerator

    def get_covariance_matrix(self):
        tensor_cov = self._get_tensor_covariance()
        cov = self.params[COV_STAT_FUNC](tensor_cov, axis=0)
        if self.params[KERNEL] is not None:
            cov = self._map_kernel(cov)
        return diagonal_block_expand(cov, tensor_cov.shape[0])

    def _get_tensor_covariance(self):
        if self.params[ALGORITHM_NAME] in ['pca'] or self.params[LAG_TIME] <= 0:
            return np.asarray(list(
                map(lambda index: self.params[COV_FUNCTION](self._standardized_data[:, :, index].T),
                    range(self._standardized_data.shape[2]))
            ))
        else:  # 'tica'
            return np.asarray(list(
                map(lambda index: self.params[COV_FUNCTION](
                    self._standardized_data[:-self.params[LAG_TIME], :, index].T),
                    range(self._standardized_data.shape[2]))
            ))

    def _map_kernel(self, cov):
        trajectory_name = 'trajectory_name' if self.params[PLOT_2D_GAUSS] else None
        kernel_matrix = calculate_symmetrical_kernel_from_matrix(
            cov, self.params[KERNEL_STAT_FUNC], self.params[KERNEL_TYPE], trajectory_name)
        if self.params[KERNEL] == KERNEL_ONLY:
            cov = kernel_matrix
        elif self.params[KERNEL] == KERNEL_DIFFERENCE:
            cov -= kernel_matrix
        elif self.params[KERNEL] == KERNEL_MULTIPLICATION:
            cov *= kernel_matrix
        return cov

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        # assert is_matrix_symmetric(self._covariance_matrix), 'Covariance-Matrix should be symmetric.'
        if self.params[ALGORITHM_NAME] in ['tica']:
            correlation_matrix = self._get_correlations_matrix()
            assert is_matrix_symmetric(correlation_matrix), 'Correlation-Matrix should be symmetric.'
            eigenvalues, eigenvectors = scipy.linalg.eigh(correlation_matrix, b=self._covariance_matrix)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]

    def _get_correlations_matrix(self):
        if self.params[LAG_TIME] <= 0:
            corr = self._get_tensor_covariance()
        else:
            temp_list = []
            for index in range(self._standardized_data.shape[2]):
                dot_i = np.dot(self._standardized_data[:-self.params[LAG_TIME], :, index].T,
                               self._standardized_data[self.params[LAG_TIME]:, :, index]) / (
                                self.n_samples - self.params[LAG_TIME])
                sym_i = 0.5 * (dot_i + dot_i.T)
                temp_list.append(sym_i)
            corr = np.asarray(temp_list)
        stat_corr = self.params[COV_STAT_FUNC](corr, axis=0)
        return diagonal_block_expand(stat_corr, corr.shape[0])

    def transform(self, data_tensor, n_components):
        return super().transform(data_tensor, n_components)
