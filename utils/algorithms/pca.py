import numpy as np

from utils.algorithms import MyModel
from utils.matrix_tools import calculate_symmetrical_kernel_matrix


class MyPCA(MyModel):
    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending and select columns based on n_components
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]


class TruncatedPCA(MyPCA):
    def __init__(self, trunc_value):
        super().__init__()
        self.truncation_value = trunc_value

    def __str__(self):
        return f'{super().__str__()}\ntrunc_value={self.truncation_value}'

    def get_covariance_matrix(self):
        """
        Calculates covariance matrix, with some kind of ´truncation-lag´ (2nd dimension).
        :return:
        """
        return np.dot(self._standardized_data[:, :-self.truncation_value].T,
                      self._standardized_data[:, self.truncation_value:]) / self.n_samples

    def fit_transform(self, data_ndarray, y=None, **fit_params):
        raise NotImplementedError('Truncated eigenvalue matrix, has an other shape as the data matrix '
                                  'which should be transformed.')


class KernelFromCovPCA(MyPCA):
    def get_covariance_matrix(self):
        """
        Additionally to the original PCA, the covariance matrix is a kernel calculated from the covariance matrix.
        :return: The kernel from the covariance matrix
        """
        super_cov = super().get_covariance_matrix()
        return calculate_symmetrical_kernel_matrix(super_cov, flattened=True, analyse_mode='print')
