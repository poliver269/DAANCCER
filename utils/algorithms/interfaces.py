import numbers
from math import sqrt

import numpy as np
from deeptime.decomposition import TICA as DTICA
from numpy.linalg import linalg
from pyemma.coordinates.transform.tica import TICA
from pyemma.coordinates.transform.pca import PCA as pPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition._pca import _infer_dimension
from sklearn.utils.extmath import stable_cumsum, svd_flip


class DeeptimeTICAInterface(DTICA):
    def __str__(self):
        return f'dtTICA(n_components={self.n_components}, lag_time={self.lagtime})'

    @property
    def explained_variance_(self):
        return self.model.singular_values

    @property
    def n_components(self):
        return self.dim

    @property
    def components_(self):
        return self.model.instantaneous_coefficients.T

    @property
    def mean_(self):
        return self.model.mean_t  # TODO probably not the correct mean.


class PyemmaTICAInterface(TICA):
    __serialize_version = 0

    def __str__(self):
        return f'TICA(n_components={self.n_components}, lag_time={self.lag})'

    def _get_traj_info(self, filename):
        pass

    @property
    def explained_variance_(self):
        return self.eigenvalues

    @property
    def components_(self):
        return self.eigenvectors.T[:self.dim]

    @property
    def n_components(self):
        return self.dim

    @property
    def mean_(self):
        return self.mean


class PyemmaPCAInterface(pPCA):
    __serialize_version = 0

    def __str__(self):
        return f'pPCA(n_components={self.n_components})'

    def _get_traj_info(self, filename):
        pass

    @property
    def explained_variance_(self):
        return self.eigenvalues

    @property
    def components_(self):
        return self.eigenvectors.T[:self.dim]

    @property
    def n_components(self):
        return self.dim

    @property
    def mean_(self):
        return self.mean


class SklearnPCA(PCA):
    def __str__(self):
        return f'skPCA(n_components={self.n_components})'

    def fit_transform(self, X, y=None):
        U, S, Vt = self._fit(X)
        U = U[:, :self.n_components_]

        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        if self.whiten:
            # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            U *= sqrt(X.shape[0] - 1)
        else:
            # X_new = X * V = U * S * Vt * V = U * S
            U *= S[: self.n_components_]

        return np.dot(X, self.components_.T)

    def _fit_full(self, X, n_components):
        """Fit the model by computing full SVD on X."""
        n_samples, n_features = X.shape

        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError(
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, "
                    "was of type=%r" % (n_components, type(n_components))
                )

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        U, S, Vt = linalg.svd(X, full_matrices=True)
        # flip eigenvectors' sign to enforce deterministic output
        # U, Vt = svd_flip(U, Vt)

        components_ = Vt.T

        # Get variance explained by singular values
        explained_variance_ = (S**2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.

        # Postprocess the number of components required
        if n_components == "mle":
            n_components = _infer_dimension(explained_variance_, n_samples)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than n_components float
            # passed. More discussion in issue: #15669
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components, side="right") + 1
        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.0

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return U, S, Vt


class ICA(FastICA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.explained_variance_ = np.zeros(self.n_components)

    def __str__(self):
        return f'ICA(n_components={self.n_components})'

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None):
        fit_transformed = super().fit_transform(X, y)
        pca = PCA()
        pca.fit_transform(fit_transformed)
        self.explained_variance_ = pca.explained_variance_
        return fit_transformed
