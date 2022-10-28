import numpy as np

from utils.algorithms import MyModel
from utils.math import diagonal_block_expand, gauss_kernel_symmetrical_matrix, comedian4df, co_mad


class TensorDR(MyModel):
    def __init__(self, cov_stat_func=np.mean, kernel_stat_func=np.median):
        super().__init__()
        self.cov_statistical_function = cov_stat_func
        self.kernel_statistical_function = kernel_stat_func

    def fit_transform(self, data_tensor, n_components=2):
        self.n_samples = data_tensor.shape[0]
        self.n_components = n_components
        self.standardized_data = self.standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self._update_cov()
        self.eigenvectors = self.get_eigenvectors()
        self._update_data_tensor()
        return self.transform(self.standardized_data)

    def get_covariance_matrix(self):
        pass

    def _update_cov(self):
        averaged_cov = self.cov_statistical_function(self._covariance_matrix, axis=0)
        self._covariance_matrix = diagonal_block_expand(averaged_cov, self._covariance_matrix.shape[0])

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending and select columns based on n_components
        index_matrix = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[index_matrix]
        return eigenvectors[:, index_matrix]

    def _update_data_tensor(self):
        self.standardized_data = self.standardized_data.reshape(self.standardized_data.shape[0],
                                                                self.standardized_data.shape[1] *
                                                                self.standardized_data.shape[2])


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
        return f'KernelOnlyMadPCA:'

    def get_covariance_matrix(self):
        return np.asarray(list(
            map(lambda index: co_mad(self.standardized_data[:, :, index].T),
                range(self.standardized_data.shape[2]))
        ))

