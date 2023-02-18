import numpy as np
import scipy

from utils.algorithms.tensor_dim_reductions import TensorDR
from utils.algorithms.tensor_dim_reductions.pca import (TensorKernelOnCovPCA, TensorKernelOnPearsonCovPCA,
                                                        TensorKernelFromCovPCA)
from utils.matrix_tools import (co_mad, calculate_symmetrical_kernel_matrix, diagonal_block_expand,
                                is_matrix_symmetric)


class TensorTICA(TensorDR):
    def __init__(self, lag_time):
        super().__init__()
        self.lag_time = lag_time

    def __str__(self):
        return f'{super().__str__()}\ntime_lag={self.lag_time}'

    def get_covariance_matrix(self):
        if self.lag_time <= 0:
            return np.asarray(list(
                map(lambda index: np.cov(self._standardized_data[:, :, index].T),
                    range(self._standardized_data.shape[2]))
            ))
        else:
            return np.asarray(list(
                map(lambda index: np.cov(self._standardized_data[:-self.lag_time, :, index].T),
                    range(self._standardized_data.shape[2]))
            ))

    def get_correlation_matrix(self):
        if self.lag_time <= 0:
            return self.get_covariance_matrix()
        else:
            temp_list = []
            for index in range(self._standardized_data.shape[2]):
                dot_i = np.dot(self._standardized_data[:-self.lag_time, :, index].T,
                               self._standardized_data[self.lag_time:, :, index]) / (
                                self.n_samples - self.lag_time)
                sym_i = 0.5 * (dot_i + dot_i.T)
                temp_list.append(sym_i)
            return np.asarray(temp_list)

    def _update_corr(self, correlation_matrix):
        for i in range(self._standardized_data.shape[2]):
            assert is_matrix_symmetric(correlation_matrix[i, :, :]), f'Correlation-Matrix ({i}) should be symmetric'
        averaged_corr = self.params['cov_stat_func'](correlation_matrix, axis=0)
        return diagonal_block_expand(averaged_corr, correlation_matrix.shape[0])

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        tensor_correlation_matrix = self.get_correlation_matrix()
        correlation_matrix = self._update_corr(tensor_correlation_matrix)
        assert is_matrix_symmetric(correlation_matrix), 'Correlation-Matrix should be symmetric.'
        assert is_matrix_symmetric(self._covariance_matrix), 'Covariance-Matrix should be symmetric.'
        eigenvalues, eigenvectors = scipy.linalg.eigh(correlation_matrix, b=self._covariance_matrix)

        # sort eigenvalues descending and select columns based on n_components
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]


class TensorPearsonCovTICA(TensorTICA):
    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: np.corrcoef(self._standardized_data[:-self.lag_time, :, index].T),
                range(self._standardized_data.shape[2]))
        ))


class TensorKernelOnCovTICA(TensorTICA, TensorKernelOnCovPCA):
    def _update_cov(self):
        averaged_cov = self.params['cov_stat_func'](self._covariance_matrix, axis=0)
        d_matrix = calculate_symmetrical_kernel_matrix(averaged_cov, self.params['kernel_stat_func'],
                                                            'my_gaussian', '2f4k')
        if not is_matrix_symmetric(d_matrix):
            if is_matrix_symmetric(d_matrix, rtol=1.e-3, atol=1.e-6):
                d_matrix = 0.5 * (d_matrix + d_matrix.T)
            else:
                raise ValueError('Created Matrix is asymmetric.')
        weighted_cov_matrix = averaged_cov - d_matrix
        self._covariance_matrix = diagonal_block_expand(weighted_cov_matrix, self._covariance_matrix.shape[0])


class TensorKernelOnPearsonCovTICA(TensorPearsonCovTICA, TensorKernelOnPearsonCovPCA):
    pass


class TensorKernelFromCovTICA(TensorTICA, TensorKernelFromCovPCA):
    def _update_cov(self):
        averaged_cov = self.params['cov_stat_func'](self._covariance_matrix, axis=0)
        d_matrix = calculate_symmetrical_kernel_matrix(averaged_cov, self.params['kernel_stat_func'],
                                                            'my_gaussian', analyse_mode='2f4k')
        if not is_matrix_symmetric(d_matrix):
            if is_matrix_symmetric(d_matrix, rtol=1.e-3, atol=1.e-6):
                d_matrix = 0.5 * (d_matrix + d_matrix.T)
            else:
                raise ValueError('Created Matrix is asymmetric.')
        self._covariance_matrix = diagonal_block_expand(d_matrix, self._covariance_matrix.shape[0])


class TensorKernelFromCoMadTICA(TensorKernelFromCovTICA):
    def get_covariance_matrix(self):
        if self.lag_time <= 0:
            return np.asarray(list(
                map(lambda index: co_mad(self._standardized_data[:, :, index].T),
                    range(self._standardized_data.shape[2]))
            ))
        else:
            return np.asarray(list(
                map(lambda index: co_mad(self._standardized_data[:-self.lag_time, :, index].T),
                    range(self._standardized_data.shape[2]))
            ))


class TensorKernelOnCoMadTICA(TensorKernelOnCovTICA):
    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: co_mad(self._standardized_data[:-self.lag_time, :, index].T),
                range(self._standardized_data.shape[2]))
        ))
