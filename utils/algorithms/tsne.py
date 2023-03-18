from scipy import spatial

from utils.algorithms import MyModel
import sklearn.manifold as sk
import numpy as np

from utils.matrix_tools import ensure_matrix_symmetry

"""
Parts of this file were originally copied from the tltsne python module.  
https://github.com/spiwokv/tltsne/blob/master/tltsne/__init__.py
Since the results in the text file are complicated to reuse, this module was modified somewhat.
This way, the results of the models can be used and it's Object Oriented.  
"""


class MyTSNE(MyModel):
    def __init__(self, n_components, perplexity, early_exaggeration, learning_rate, n_iter, metric="euclidean"):
        super().__init__()
        self.model = sk.TSNE(
            n_components=n_components, perplexity=perplexity,
            early_exaggeration=early_exaggeration, learning_rate=learning_rate,
            n_iter=n_iter, metric=metric
        )

    def fit_transform(self, data_matrix, **fit_params):
        return self.model.fit_transform(data_matrix, **fit_params)


class MyTimeLaggedTSNE(MyTSNE):
    def __init__(self, lag_time, kwargs):
        super().__init__(metric="precomputed", **kwargs)
        self.lag_time = lag_time

    def fit_transform(self, data_matrix, **fit_params):
        data_zero_mean = data_matrix - np.mean(data_matrix, axis=0)
        cov = np.cov(data_zero_mean.T)
        eigenvalue, eigenvector = np.linalg.eig(cov)
        eigenvalue_order = np.argsort(eigenvalue)[::-1]
        eigenvector = eigenvector[:, eigenvalue_order]
        eigenvalue = eigenvalue[eigenvalue_order]
        projection = data_zero_mean.dot(eigenvector) / np.sqrt(eigenvalue)

        n_frames = fit_params.get('n_frames', 0)
        if self.lag_time <= 0:
            covariance_matrix = np.dot(
                projection[:, np.newaxis].T,
                projection[:, np.newaxis]
            ) / (n_frames - 1)
        else:
            covariance_matrix = np.dot(
                projection[:-self.lag_time, np.newaxis].T,
                projection[self.lag_time:, np.newaxis]
            ) / (n_frames - self.lag_time - 1)
        covariance_matrix = ensure_matrix_symmetry(covariance_matrix)

        eigenvalue2, eigenvector2 = np.linalg.eig(covariance_matrix)
        eigenvalue_order = np.argsort(eigenvalue2)[::-1]
        eigenvector2 = eigenvector2[:, eigenvalue_order]
        eigenvalue2 = eigenvalue2[eigenvalue_order]
        projection = np.dot(
            projection,
            eigenvector2[:, :self.n_components]
        ) * np.sqrt(np.real(eigenvalue2[:self.n_components]))
        data_distance = spatial.distance_matrix(projection, projection)

        return self.model.fit_transform(data_distance)
