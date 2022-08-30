import numpy as np


class MyModel:
    def __init__(self):
        self.eigenvalues = None
        self.eigenvectors = None
        self.standardized_data_matrix = None
        self.n_components = None
        self.n_samples = None

    def fit_transform(self, data_matrix, n_components=2):
        self.n_samples = data_matrix.shape[0]
        self.n_components = n_components
        # standardize data
        self.standardized_data_matrix = self.standardize_data(data_matrix)
        # calculate covariance matrix
        covariance_matrix = self.get_covariance_matrix()
        # retrieve selected eigenvectors
        self.eigenvectors = self.get_eigenvectors(covariance_matrix)
        # project into lower dimension
        return self.project_matrix()

    @staticmethod
    def standardize_data(matrix):
        # subtract mean and divide by standard deviation column-wise
        numerator = matrix - np.mean(matrix, axis=0)
        denominator = np.std(matrix, axis=0)
        return numerator / denominator

    def get_covariance_matrix(self, ddof=0):
        pass

    def get_eigenvectors(self, covariance_matrix):
        pass

    def project_matrix(self):
        return np.dot(self.standardized_data_matrix, self.eigenvectors)
