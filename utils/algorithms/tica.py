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
        """
        Covariance Matrix = X_(0...-lag_time) dot X_(0...-lag_time).T
        :return:
        """
        if self.lag_time <= 0:
            return np.cov(self.standardized_data.T)
        else:
            return np.cov(self.standardized_data[:-self.lag_time].T)

    def get_correlation_matrix(self):
        """
        Correlation Matrix = X_(0...-lag_time) dot X_(lag_time...shape).T / shape-time_lag
        :return:
        """
        if self.lag_time <= 0:
            return self.get_covariance_matrix()
        else:
            return np.dot(self.standardized_data[:-self.lag_time].T,
                          self.standardized_data[self.lag_time:]) / (self.n_samples - self.lag_time)

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance and correlation matrix
        correlation_matrix = self.get_correlation_matrix()
        self.eigenvalues, eigenvectors = scipy.linalg.eig(correlation_matrix, b=self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(self.eigenvalues)[::-1]
        return eigenvectors[:, sorted_eigenvalue_indexes]


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
            return super().get_covariance_matrix()
        else:
            if self.lag_time <= 0:
                return np.cov(self.standardized_data[:, :-self.truncation_value].T)
            else:
                return np.cov(self.standardized_data[:-self.lag_time, :-self.truncation_value].T)

    def get_correlation_matrix(self):
        if self.truncation_value <= 0:
            return super().get_covariance_matrix()
        else:
            if self.lag_time <= 0:
                return np.dot(self.standardized_data[:, :-self.truncation_value:].T,
                              self.standardized_data[:, self.truncation_value:]) / self.n_samples
            else:
                return np.dot(self.standardized_data[:-self.lag_time, :-self.truncation_value].T,
                              self.standardized_data[self.lag_time:, self.truncation_value:]) / (
                                   self.n_samples - self.lag_time)

    def fit_transform(self, data_matrix, n_components=2):
        raise NotImplementedError(
            'Truncated eigenvalue matrix, has an other shape as the data matrix which should be transformed.')
