from utils.algorithms import MyModel
import numpy as np


class MyPCA(MyModel):
    def __str__(self):
        return f'MyPCA:\ncomponents={self.n_components}'

    def get_covariance_matrix(self):
        """
        Calculate covariance matrix with standardized matrix A
        :return: Covariance Matrix
        """
        # return np.dot(self.standardized_data_matrix.T, self.standardized_data_matrix) / self.n_samples
        return np.cov(self.standardized_data_matrix.T)

    def get_eigenvectors(self, covariance_matrix):
        # calculate eigenvalues & eigenvectors of covariance matrix
        self.eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # sort eigenvalues descending and select columns based on n_components
        n_cols = np.argsort(self.eigenvalues)[::-1][:self.n_components]
        selected_vectors = eigenvectors[:, n_cols]
        return selected_vectors


class TruncatedPCA(MyPCA):
    def __init__(self, trunc_value):
        super().__init__()
        self.truncation_value = trunc_value

    def __str__(self):
        return f'TruncatedPCA:\ncomponents={self.n_components}, trunc_value={self.truncation_value}'

    def get_covariance_matrix(self):
        return np.dot(self.standardized_data_matrix[:-self.truncation_value].T,
                      self.standardized_data_matrix[self.truncation_value:])  # / self.n_samples
