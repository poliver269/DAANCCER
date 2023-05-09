import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from utils.param_keys import N_COMPONENTS


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
        self._standardized_data = None
        self.n_components = None
        self.n_samples = None
        self._covariance_matrix = None

    def __str__(self):
        return f'{self.__class__.__name__}:\ncomponents={self.n_components}'

    def fit_transform(self, data_ndarray, **fit_params):
        self.fit(data_ndarray, **fit_params)
        return self.transform(data_ndarray)

    def fit(self, data_matrix, **fit_params):
        self.n_samples = data_matrix.shape[0]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data = self._standardize_data(data_matrix)
        self._covariance_matrix = self.get_covariance_matrix()
        self.components_ = self.get_eigenvectors()
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

    def get_eigenvectors(self):
        pass

    def transform(self, data_matrix):
        """
        Project the data to the lower dimension with the help of the eigenvectors.
        :return: Data reduced to lower dimensions from higher dimensions
        """
        data_matrix_standardized = self._standardize_data(data_matrix)
        return np.dot(data_matrix_standardized, self.components_[:, :self.n_components])
