from utils.algorithms import MyModel
from utils.math import diagonal_block_expand
import numpy as np


class TensorDR(MyModel):
    def __init__(self, cov_stat_func=np.mean, kernel_stat_func=np.median):
        super().__init__()
        self.cov_statistical_function = cov_stat_func
        self.kernel_statistical_function = kernel_stat_func

    def __str__(self):
        return f'{self.__class__.__name__}:\ncomponents={self.n_components}\n'

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
        return

    def _update_cov(self):
        averaged_cov = self.cov_statistical_function(self._covariance_matrix, axis=0)
        self._covariance_matrix = diagonal_block_expand(averaged_cov, self._covariance_matrix.shape[0])

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]

    def _update_data_tensor(self):
        self.standardized_data = self.standardized_data.reshape(self.standardized_data.shape[0],
                                                                self.standardized_data.shape[1] *
                                                                self.standardized_data.shape[2])
