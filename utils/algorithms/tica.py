import numpy as np
import scipy

from plotter import ArrayPlotter
from utils.algorithms import MyModel
from utils.matrix_tools import calculate_symmetrical_kernel_from_matrix


class MyTICA(MyModel):
    def __init__(self, lag_time):
        super().__init__()
        self.lag_time = lag_time

    def __str__(self):
        return f'{super().__str__()}\ntime_lag={self.lag_time}'

    def get_covariance_matrix(self):
        """
        Covariance Matrix = X_(0...-lag_time) dot X_(0...-lag_time).T
        :return:
        """
        if self.lag_time <= 0:
            return super().get_covariance_matrix()
        else:
            return np.cov(self._standardized_data[:-self.lag_time].T)

    def get_correlation_matrix(self):
        """
        Correlation Matrix = X_(0...-lag_time) dot X_(lag_time...shape).T / shape-time_lag
        :return:
        """
        if self.lag_time <= 0:
            return self.get_covariance_matrix()
        else:
            corr = np.dot(self._standardized_data[:-self.lag_time].T,
                          self._standardized_data[self.lag_time:]) / (self.n_samples - self.lag_time)
            return 0.5 * (corr + corr.T)

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance and correlation matrix
        correlation_matrix = self.get_correlation_matrix()
        eigenvalues, eigenvectors = scipy.linalg.eigh(correlation_matrix, b=self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]


class TruncatedTICA(MyTICA):
    def __init__(self, lag_time, trunc_value):
        super().__init__(lag_time)
        self.truncation_value = trunc_value

    def __str__(self):
        return f'{super().__str__()}\ntruncation_value={self.truncation_value}'

    def get_covariance_matrix(self):
        if self.truncation_value <= 0:
            return super().get_covariance_matrix()
        else:
            if self.lag_time <= 0:
                return np.cov(self._standardized_data[:, :-self.truncation_value].T)
            else:
                return np.cov(self._standardized_data[:-self.lag_time, :-self.truncation_value].T)

    def get_correlation_matrix(self):
        if self.truncation_value <= 0:
            return super().get_correlation_matrix()
        else:
            if self.lag_time <= 0:
                cov = np.dot(self._standardized_data[:, :-self.truncation_value].T,
                             self._standardized_data[:, self.truncation_value:]) / self.n_samples
            else:
                cov = np.dot(self._standardized_data[:-self.lag_time, :-self.truncation_value].T,
                             self._standardized_data[self.lag_time:, self.truncation_value:]) / (
                              self.n_samples - self.lag_time)
            return 0.5 * (cov + cov.T)

    def fit_transform(self, data_ndarray, y=None, **fit_params):
        raise NotImplementedError('Truncated eigenvalue matrix, has an other shape as the data matrix '
                                  'which should be transformed. '
                                  'Idea: truncate also the input by the truncation value.')


class KernelFromCovTICA(MyTICA):
    def get_covariance_matrix(self):
        super_cov = super().get_covariance_matrix()
        ArrayPlotter(False).matrix_plot(super_cov)
        stat_cov = calculate_symmetrical_kernel_from_matrix(super_cov, flattened=True, trajectory_name='kernel diff')
        ArrayPlotter(False).matrix_plot(super_cov - stat_cov)
        return stat_cov
