import numpy as np

from utils.algorithms.tensor_dim_reductions import TensorDR
from utils.matrix_tools import diagonal_block_expand, calculate_symmetrical_kernel_from_matrix, co_mad


class TensorPCA(TensorDR):
    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: np.cov(self._standardized_data[:, :, index].T),
                range(self._standardized_data.shape[2]))
        ))


class TensorPearsonCovPCA(TensorDR):
    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: np.corrcoef(self._standardized_data[:, :, index].T),
                range(self._standardized_data.shape[2]))
        ))


class TensorKernelOnCovPCA(TensorPCA):
    def _update_cov(self):
        statistical_cov = self.cov_stat_func(self._covariance_matrix, axis=0)
        d_matrix = calculate_symmetrical_kernel_from_matrix(statistical_cov)
        weighted_cov_matrix = statistical_cov - d_matrix
        self._covariance_matrix = diagonal_block_expand(weighted_cov_matrix, self._covariance_matrix.shape[0])


class TensorKernelOnPearsonCovPCA(TensorPearsonCovPCA):
    def _update_cov(self):
        averaged_cov = self.cov_stat_func(self._covariance_matrix, axis=0)
        d_matrix = calculate_symmetrical_kernel_from_matrix(averaged_cov, self.kernel_stat_func)
        weighted_alpha_coeff_matrix = averaged_cov - d_matrix
        self._covariance_matrix = diagonal_block_expand(weighted_alpha_coeff_matrix, self._covariance_matrix.shape[0])


class TensorKernelFromCovPCA(TensorPCA):
    def _update_cov(self):
        averaged_cov = self.cov_stat_func(self._covariance_matrix, axis=0)
        d_matrix = calculate_symmetrical_kernel_from_matrix(averaged_cov, self.kernel_stat_func)
        self._covariance_matrix = diagonal_block_expand(d_matrix, self._covariance_matrix.shape[0])


class TensorKernelFromComadPCA(TensorKernelFromCovPCA):
    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: co_mad(self._standardized_data[:, :, index].T),
                range(self._standardized_data.shape[2]))
        ))
