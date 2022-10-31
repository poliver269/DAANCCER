import numpy as np
import scipy

from utils.algorithms.tensor_dim_reductions import TensorDR
from utils.algorithms.tensor_dim_reductions.pca import TensorKernelOnCovPCA, TensorKernelOnPearsonCovPCA, \
    TensorKernelFromCovPCA
from utils.math import co_mad, calculate_gauss_kernel_on_matrix, diagonal_block_expand


class TensorTICA(TensorDR):
    def __init__(self, lag_time):
        super().__init__()
        self.lag_time = lag_time

    def get_covariance_matrix(self):
        if self.lag_time <= 0:
            return np.asarray(list(
                map(lambda index: np.cov(self.standardized_data[:, :, index].T),
                    range(self.standardized_data.shape[2]))
            ))
        else:
            return np.asarray(list(
                map(lambda index: np.cov(self.standardized_data[:, :, index][:-self.lag_time].T),
                    range(self.standardized_data.shape[2]))
            ))

    def get_correlation_matrix(self):
        if self.lag_time <= 0:
            return self.get_covariance_matrix()
        else:
            return np.asarray(list(
                map(lambda index: np.dot(self.standardized_data[:, :, index][:-self.lag_time].T,
                                         self.standardized_data[:, :, index][self.lag_time:]) / (
                                              self.n_samples - self.lag_time),
                    range(self.standardized_data.shape[2]))
            ))

    def _update_corr(self, correlation_matrix):
        averaged_corr = self.cov_statistical_function(correlation_matrix, axis=0)
        return diagonal_block_expand(averaged_corr, correlation_matrix.shape[0])

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        correlation_matrix = self.get_correlation_matrix()
        correlation_matrix = self._update_corr(correlation_matrix)
        self.eigenvalues, eigenvectors = scipy.linalg.eig(correlation_matrix, b=self._covariance_matrix)

        # sort eigenvalues descending and select columns based on n_components
        n_cols = np.argsort(self.eigenvalues)[::-1]
        return eigenvectors[:, n_cols]


class TensorPearsonCovTICA(TensorTICA):
    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: np.corrcoef(self.standardized_data[:, :, index][:-self.lag_time].T),
                range(self.standardized_data.shape[2]))
        ))  # TODO: Think of the approach: calculate the corrcoef for the correlation matrix too?


class TensorKernelOnCovTICA(TensorTICA, TensorKernelOnCovPCA):
    pass


class TensorKernelOnPearsonCovTICA(TensorPearsonCovTICA, TensorKernelOnPearsonCovPCA):
    pass


class TensorKernelFromCovTICA(TensorTICA, TensorKernelFromCovPCA):
    def _update_cov(self):
        averaged_cov = self.cov_statistical_function(self._covariance_matrix, axis=0)
        d_matrix = calculate_gauss_kernel_on_matrix(averaged_cov, self.kernel_statistical_function)
        self._covariance_matrix = diagonal_block_expand(d_matrix, self._covariance_matrix.shape[0])


class TensorKernelFromComadTICA(TensorKernelFromCovTICA):
    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: co_mad(self.standardized_data[:, :, index][:-self.lag_time].T),
                range(self.standardized_data.shape[2]))
        ))
