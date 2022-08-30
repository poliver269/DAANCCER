from utils.algorithms import MyModel
import numpy as np


class MyPCA(MyModel):
    """
    https://towardsdatascience.com/implementing-pca-from-scratch-fb434f1acbaa
    """
    def __str__(self):
        return f'MyPCA: components={self.n_components}'

    def get_covariance_matrix(self, ddof=0):
        """
        Calculate covariance matrix with standardized matrix A
        :param ddof:
        :return: Covariance Matrix
        """
        # return np.dot(self.standardized_data_matrix.T, self.standardized_data_matrix) / (self.n_samples - ddof)
        return np.cov(self.standardized_data_matrix.T)

    def get_eigenvectors(self, covariance_matrix):
        # calculate eigenvalues & eigenvectors of covariance matrix
        self.eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # sort eigenvalues descending and select columns based on n_components
        n_cols = np.argsort(self.eigenvalues)[::-1][:self.n_components]
        selected_vectors = eigenvectors[:, n_cols]
        return selected_vectors


class TruncatedPCA(MyPCA):
    def __init__(self, trunc_value):
        super().__init__()
        self.truncation_value = trunc_value

    def __str__(self):
        return f'TruncatedPCA: components={self.n_components}, trunc_value={self.truncation_value}'

    def get_covariance_matrix(self, ddof=0):
        return np.dot(self.standardized_data_matrix[self.truncation_value:].T,
                      self.standardized_data_matrix[:-self.truncation_value]) / (self.n_samples - ddof)
