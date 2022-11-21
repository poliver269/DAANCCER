import scipy

from utils.algorithms import MyModel
from utils.math import is_matrix_symmetric
from utils.matrix_tools import diagonal_block_expand, co_mad, calculate_symmetrical_kernel_from_matrix
import numpy as np

from utils.param_key import PLOT_2D_GAUSS


class TensorDR(MyModel):
    def __init__(self, model_parameters=None):
        super().__init__()
        if model_parameters is None:
            model_parameters = {}

        self.params = {
            'cov_stat_func': model_parameters.get('cov_stat_func', np.mean),
            'kernel_stat_func': model_parameters.get('kernel_stat_func', np.median)
        }

    def fit(self, data_tensor):
        self.n_samples = data_tensor.shape[0]
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self._update_cov()
        self.eigenvectors = self.get_eigenvectors()

    def get_covariance_matrix(self):
        return

    def _update_cov(self):
        averaged_cov = self.params['cov_stat_func'](self._covariance_matrix, axis=0)
        self._covariance_matrix = diagonal_block_expand(averaged_cov, self._covariance_matrix.shape[0])

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]

    def transform(self, data_tensor, n_components):
        data_matrix = self._update_data_tensor(data_tensor)
        return super(TensorDR, self).transform(data_matrix, n_components)

    @staticmethod
    def _update_data_tensor(data_tensor):
        return data_tensor.reshape(data_tensor.shape[0], data_tensor.shape[1] * data_tensor.shape[2])


class ParameterModel(TensorDR):
    def __init__(self, model_parameters):
        super().__init__(model_parameters)
        self.params.update({
            'algorithm_name': model_parameters.get('algorithm_name', 'pca'),  # pc, tica
            'ndim': model_parameters.get('ndim', 3),  # 3: tensor, 2: matrix
            'kernel': model_parameters.get('kernel', None),  # diff, multi, only, None
            'kernel_type': model_parameters.get('kernel_type', 'my_gaussian'),
            # my_gaussian, my_exponential, my_epanechnikov
            'cov_function': model_parameters.get('covariance', 'cov_function'),
            # covariance, pearson_correlation, comMAD
            PLOT_2D_GAUSS: model_parameters.get(PLOT_2D_GAUSS, False),
            'lag_time': model_parameters.get('lag_time', 0)
        })

    def fit_transform(self, data_tensor, n_components=2):
        super().fit_transform(data_tensor, n_components)

    def fit(self, data_tensor):
        self.n_samples = data_tensor.shape[0]
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self.eigenvectors = self.get_eigenvectors()

    def get_covariance_matrix(self):
        tensor_cov = self._get_tensor_covariance()
        cov = self.params['cov_stat_func'](tensor_cov, axis=0)
        if self.params['kernel'] is not None:
            cov = self._map_kernel(cov)
        return diagonal_block_expand(cov, tensor_cov.shape[0])

    def _get_tensor_covariance(self):
        funcs = {'covariance': np.cov, 'pearson_correlation': np.corrcoef, 'coMAD': co_mad}
        if self.params['lag_time'] <= 0:
            return np.asarray(list(
                map(lambda index: funcs[self.params['covariance']](self._standardized_data[:, :, index].T),
                    range(self._standardized_data.shape[2]))
            ))
        else:
            return np.asarray(list(
                map(lambda index: funcs[self.params['covariance']](
                    self._standardized_data[:-self.params['lag_time'], :, index].T),
                    range(self._standardized_data.shape[2]))
            ))

    def _map_kernel(self, cov):
        trajectory_name = 'trajectory_name' if self.params[PLOT_2D_GAUSS] else None
        kernel_matrix = calculate_symmetrical_kernel_from_matrix(
            cov, self.params['kernel_stat_func'], self.params['kernel_type'], trajectory_name)
        if self.params['kernel'] == 'only':
            cov = kernel_matrix
        elif self.params['kernel'] == 'diff':
            cov -= kernel_matrix
        elif self.params['kernel'] == 'multi':
            cov *= kernel_matrix
        return cov

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        assert is_matrix_symmetric(self._covariance_matrix), 'Covariance-Matrix should be symmetric.'
        if self.params['algorithm_name'] in ['tica']:
            correlation_matrix = self._get_correlations_matrix()
            assert is_matrix_symmetric(correlation_matrix), 'Correlation-Matrix should be symmetric.'
            eigenvalues, eigenvectors = scipy.linalg.eigh(correlation_matrix, b=self._covariance_matrix)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]

    def _get_correlations_matrix(self):
        if self.params['lag_time'] <= 0:
            corr = self._get_tensor_covariance()
        else:
            temp_list = []
            for index in range(self._standardized_data.shape[2]):
                dot_i = np.dot(self._standardized_data[:-self.params['lag_time'], :, index].T,
                               self._standardized_data[self.params['lag_time']:, :, index]) / (
                                self.n_samples - self.params['lag_time'])
                sym_i = 0.5 * (dot_i + dot_i.T)
                temp_list.append(sym_i)
            corr = np.asarray(temp_list)
        stat_corr = self.params['cov_stat_func'](corr, axis=0)
        return diagonal_block_expand(stat_corr, corr.shape[0])

    def transform(self, data_tensor, n_components):
        return super().transform(data_tensor, n_components)
