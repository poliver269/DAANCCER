from utils.algorithms import MyModel
import numpy as np
import scipy


class MyTICA(MyModel):
    def __init__(self, lag_time):
        super().__init__()
        self.lag_time = lag_time

    def __str__(self):
        return f'MyTICA:\ncomponents={self.n_components}, time_lag={self.lag_time}'

    def get_covariance_matrix(self):
        if self.lag_time <= 0:
            return np.cov(self.standardized_data_matrix.T)
        else:
            return np.cov(self.standardized_data_matrix[:-self.lag_time].T)

    def get_correlation_matrix(self):
        if self.lag_time <= 0:
            return self.get_covariance_matrix()
        else:
            return np.dot(self.standardized_data_matrix[:-self.lag_time].T,
                          self.standardized_data_matrix[self.lag_time:]) / self.n_samples

    def get_eigenvectors(self, covariance_matrix):
        # calculate eigenvalues & eigenvectors of covariance matrix
        correlation_matrix = self.get_correlation_matrix()
        self.eigenvalues, eigenvectors = scipy.linalg.eig(correlation_matrix, b=covariance_matrix)

        # sort eigenvalues descending and select columns based on n_components
        n_cols = np.argsort(self.eigenvalues)[::-1][:self.n_components]
        selected_vectors = eigenvectors[:, n_cols]
        return selected_vectors


class TruncatedTICA(MyTICA):
    def __init__(self, lag_time, trunc_value):
        super().__init__(lag_time)
        self.truncation_value = trunc_value

    def __str__(self):
        return (f'TruncatedTICA:\ncomponents={self.n_components}, '
                f'lag_time={self.lag_time}, '
                f'truncation_value={self.truncation_value}')

    def get_covariance_matrix(self):
        if self.truncation_value <= 0:
            return np.cov(self.standardized_data_matrix.T)
        else:
            return np.dot(self.standardized_data_matrix[self.truncation_value:].T,
                          self.standardized_data_matrix[:-self.truncation_value]) / self.n_samples
