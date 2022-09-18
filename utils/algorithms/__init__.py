import numpy as np


class MyModel:
    """
    This class and some child classes are partly copied from:
    https://towardsdatascience.com/implementing-pca-from-scratch-fb434f1acbaa
    and commented with the help of:
    https://www.askpython.com/python/examples/principal-component-analysis
    """
    def __init__(self):
        self.standardize = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.standardized_data_matrix = None
        self.n_components = None
        self.n_samples = None

    def fit_transform(self, data_matrix, n_components=2):
        self.n_samples = data_matrix.shape[0]
        self.n_components = n_components
        self.standardized_data_matrix = self.standardize_data(data_matrix)
        covariance_matrix = self.get_covariance_matrix()
        self.eigenvectors = self.get_eigenvectors(covariance_matrix)
        return self.transform(self.standardized_data_matrix)

    def standardize_data(self, matrix):
        """
        Subtract mean and divide by standard deviation column-wise.
        Doing this proves to be very helpful when calculating the covariance matrix.
        :param matrix: Data as matrix
        :return: Standardized data matrix
        """
        numerator = matrix - np.mean(matrix, axis=0)
        denominator = np.std(matrix, axis=0)
        return numerator / denominator

    def get_covariance_matrix(self):
        pass

    def get_eigenvectors(self, covariance_matrix):
        pass

    def transform(self, data_matrix):
        """
        Project the data to the lower dimension with the help of the eigenvectors.
        :return: Data reduced to lower dimensions from higher dimensions
        """
        return np.dot(data_matrix, self.eigenvectors)
