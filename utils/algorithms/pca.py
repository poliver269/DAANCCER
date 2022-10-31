from utils.algorithms import MyModel
import numpy as np

from utils.math import calculate_pearson_correlations


class MyPCA(MyModel):
    def __str__(self):
        return f'{self.__class__.__name__}:\ncomponents={self.n_components}'

    def get_covariance_matrix(self):
        """
        Calculate covariance matrix with standardized matrix A
        :return: Covariance Matrix
        """
        return np.cov(self.standardized_data.T)

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending and select columns based on n_components
        index_matrix = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[index_matrix]
        return eigenvectors[:, index_matrix]


class TruncatedPCA(MyPCA):
    def __init__(self, trunc_value):
        super().__init__()
        self.truncation_value = trunc_value

    def __str__(self):
        return f'{super().__str__()}, trunc_value={self.truncation_value}'

    def get_covariance_matrix(self):
        return np.dot(self.standardized_data[:, :-self.truncation_value].T,
                      self.standardized_data[:, self.truncation_value:]) / self.n_samples

    def fit_transform(self, data_matrix, n_components=2):
        raise NotImplementedError(
            'Truncated eigenvalue matrix, has an other shape as the data matrix which should be transformed.')
