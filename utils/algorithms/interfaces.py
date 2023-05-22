from deeptime.decomposition import TICA as DTICA
from pyemma.coordinates.transform.tica import TICA
from pyemma.coordinates.transform.pca import PCA


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


class PyemmaPCAInterface(PCA):
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
