from deeptime.decomposition import TICA


class DeeptimeTICAInterface(TICA):
    def __str__(self):
        return f'TICA(n_components={self.n_components}, lag_time={self.lagtime})'

    @property
    def explained_variance_(self):
        return self.model.singular_values

    @property
    def n_components(self):
        return self.dim

    @property
    def components_(self):
        return self.model.feature_component_correlation.T
