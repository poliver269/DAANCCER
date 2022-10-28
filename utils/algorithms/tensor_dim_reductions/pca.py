import numpy as np

from utils.algorithms.tensor_dim_reductions import TensorDR
from utils.math import diagonal_block_expand, gauss_kernel_symmetrical_matrix, co_mad


class TensorPCA(TensorDR):
    def __str__(self):
        return f'TensorPCA:\ncomponents={self.n_components}'

    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: np.cov(self.standardized_data[:, :, index].T),
                range(self.standardized_data.shape[2]))
        ))


class TensorPearsonPCA(TensorDR):
    def __str__(self):
        return f'PearsonPCA:\ncomponents={self.n_components}'

    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: np.corrcoef(self.standardized_data[:, :, index].T),
                range(self.standardized_data.shape[2]))
        ))


class TensorKernelPCA(TensorPCA):
    def __str__(self):
        return f'CovKernelPCA:\ncomponents={self.n_components}'

    def _update_cov(self):
        averaged_cov = self.cov_statistical_function(self._covariance_matrix, axis=0)
        d_matrix = gauss_kernel_symmetrical_matrix(averaged_cov)
        weighted_alpha_coeff_matrix = averaged_cov - d_matrix

        self._covariance_matrix = diagonal_block_expand(weighted_alpha_coeff_matrix, self._covariance_matrix.shape[0])


class TensorPearsonKernelPCA(TensorPearsonPCA):
    def __str__(self):
        return f'PearsonKernelPCA:\ncomponents={self.n_components}'

    def _update_cov(self):
        averaged_cov = self.cov_statistical_function(self._covariance_matrix, axis=0)
        d_matrix = gauss_kernel_symmetrical_matrix(averaged_cov, self.kernel_statistical_function)
        weighted_alpha_coeff_matrix = averaged_cov - d_matrix
        self._covariance_matrix = diagonal_block_expand(weighted_alpha_coeff_matrix, self._covariance_matrix.shape[0])


class KernelOnlyPCA(TensorPCA):
    def __str__(self):
        return f'KernelOnly:\ncomponents={self.n_components}'

    def _update_cov(self):
        averaged_cov = self.cov_statistical_function(self._covariance_matrix, axis=0)
        d_matrix = gauss_kernel_symmetrical_matrix(averaged_cov, self.kernel_statistical_function)
        self._covariance_matrix = diagonal_block_expand(d_matrix, self._covariance_matrix.shape[0])


class KernelOnlyMadPCA(KernelOnlyPCA):
    def __str__(self):
        return f'KernelOnlyMadPCA:\ncomponents={self.n_components}'

    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: co_mad(self.standardized_data[:, :, index].T),
                range(self.standardized_data.shape[2]))
        ))
