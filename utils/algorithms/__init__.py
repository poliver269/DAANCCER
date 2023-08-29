import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from utils.errors import ModelNotFittedError
from utils.matrix_tools import diagonal_block_expand
from utils.param_keys import N_COMPONENTS
from utils.param_keys.traj_dims import TIME_DIM, COORDINATE_DIM, FEATURE_DIM, COMBINED_DIM


class MyModel(TransformerMixin, BaseEstimator):
    """
    This class and some child classes are partly copied from:
    https://towardsdatascience.com/implementing-pca-from-scratch-fb434f1acbaa
    and commented with the help of:
    https://www.askpython.com/python/examples/principal-component-analysis
    """
    def __init__(self):
        self.explained_variance_ = None
        self.components_ = None
        self._standardized_data_ = None
        self.n_components = None
        self.n_samples = None
        self._covariance_matrix = None

    def __str__(self):
        return f'{self.__class__.__name__}:\ncomponents={self.n_components}'

    @property
    def _standardized_data(self):
        if self._standardized_data_ is None:
            raise ModelNotFittedError(f"The model `{self}` is not yet fitted. "
                                      "Please fit the model before accessing this property.")
        return self._standardized_data_

    def fit_transform(self, data_ndarray, **fit_params):
        self.fit(data_ndarray, **fit_params)
        return self.transform(data_ndarray)

    def fit(self, data_matrix, **fit_params):
        self.n_samples = data_matrix.shape[0]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data_ = self._standardize_data(data_matrix)
        self._covariance_matrix = self.get_covariance_matrix()
        self.components_ = self._get_eigenvectors()[:, :self.n_components].T
        return self

    @staticmethod
    def _standardize_data(matrix):
        """
        Subtract mean and divide by standard deviation column-wise.
        Doing this proves to be very helpful when calculating the covariance matrix.
        https://towardsdatascience.com/understanding-the-covariance-matrix-92076554ea44
        Mean-Center the data
        :param matrix: Data as matrix
        :return: Standardized data matrix
        """
        numerator = matrix - np.mean(matrix, axis=0)
        denominator = np.std(matrix, axis=0)
        return numerator / denominator

    def get_covariance_matrix(self):
        """
        Calculate covariance matrix with standardized matrix A
        :return: Covariance Matrix
        """
        return np.cov(self._standardized_data.T)

    def _get_eigenvectors(self):
        pass

    def transform(self, data_matrix):
        """
        Project the data to the lower dimension with the help of the eigenvectors.
        :return: Data reduced to lower dimensions from higher dimensions
        """
        data_matrix_standardized = self._standardize_data(data_matrix)
        return np.dot(data_matrix_standardized, self.components_.T)


class TensorDR(MyModel):
    def __init__(self, cov_stat_func=np.mean):
        super().__init__()

        if isinstance(cov_stat_func, str):
            cov_stat_func = eval(cov_stat_func)

        self.cov_stat_func = cov_stat_func

    def fit(self, data_tensor, **fit_params):
        self.n_samples = data_tensor.shape[TIME_DIM]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data_ = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self._update_cov()
        self.components_ = self._get_eigenvectors()[:, :self.n_components].T
        return self

    def get_covariance_matrix(self):
        return super().get_covariance_matrix()

    def _update_cov(self):
        averaged_cov = self.cov_stat_func(self._covariance_matrix, axis=0)
        self._covariance_matrix = diagonal_block_expand(averaged_cov, self._covariance_matrix.shape[0])

    def _get_eigenvectors(self):
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
                              self._standardized_data.shape[FEATURE_DIM] *
                              self._standardized_data.shape[COMBINED_DIM])

    def convert_to_tensor(self, matrix):
        return matrix.reshape(matrix.shape[TIME_DIM],
                              self._standardized_data.shape[FEATURE_DIM],
                              self._standardized_data.shape[COORDINATE_DIM])
