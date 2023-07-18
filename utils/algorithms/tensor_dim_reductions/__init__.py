import numpy as np

from utils.algorithms import MyModel

from utils.matrix_tools import diagonal_block_expand
from utils.param_keys import N_COMPONENTS
from utils.param_keys.traj_dims import TIME_DIM, ATOM_DIM, COORDINATE_DIM


class TensorDR(MyModel):
    def __init__(self, cov_stat_func=np.mean):
        super().__init__()

        if isinstance(cov_stat_func, str):
            cov_stat_func = eval(cov_stat_func)

        self.cov_stat_func = cov_stat_func

    def fit(self, data_tensor, **fit_params):
        self.n_samples = data_tensor.shape[TIME_DIM]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self._update_cov()
        self.components_ = self.get_eigenvectors()[:, :self.n_components].T
        return self

    def get_covariance_matrix(self):
        return super().get_covariance_matrix()

    def _update_cov(self):
        averaged_cov = self.cov_stat_func(self._covariance_matrix, axis=0)
        self._covariance_matrix = diagonal_block_expand(averaged_cov, self._covariance_matrix.shape[0])

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.explained_variance_ = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]

    def transform(self, data_tensor):
        data_matrix = self.convert_to_matrix(data_tensor)
        return super(TensorDR, self).transform(data_matrix)

    def convert_to_matrix(self, tensor):
        return tensor.reshape(tensor.shape[TIME_DIM],
                              self._standardized_data.shape[ATOM_DIM] * self._standardized_data.shape[COORDINATE_DIM])

    def convert_to_tensor(self, matrix):
        return matrix.reshape(matrix.shape[TIME_DIM],
                              self._standardized_data.shape[ATOM_DIM],
                              self._standardized_data.shape[COORDINATE_DIM])
