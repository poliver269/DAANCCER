from utils.algorithms import MyModel
import numpy as np

from utils.math import calculate_pearson_correlations


class MyPCA(MyModel):
    def __str__(self):
        return f'MyPCA:\ncomponents={self.n_components}'

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
        return f'TruncatedPCA:\ncomponents={self.n_components}, trunc_value={self.truncation_value}'

    def get_covariance_matrix(self):
        return np.dot(self.standardized_data[:-self.truncation_value].T,
                      self.standardized_data[self.truncation_value:])  # / self.n_samples


class PccPCA(MyPCA):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f'PccPCA:\ncomponents={self.n_components}, '

    def fit_transform(self, data_matrix_list, n_components=2):
        self.n_samples = data_matrix_list[0].shape[0]
        self.n_components = n_components
        self.standardized_data_matrix = data_matrix_list
        covariance_matrix = self.get_covariance_matrix()

    @staticmethod
    def standardize_data(matrix):
        return matrix

    def get_covariance_matrix(self):
        input_list = self.standardized_data_matrix
        return calculate_pearson_correlations(input_list, np.mean)

