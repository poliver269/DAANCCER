from deeptime.decomposition import TICA


class DeeptimeTICAInterface(TICA):
    @property
    def explained_variance_(self):
        return self.model.singular_values

    @property
    def n_components(self):
        return self.dim

    @property
    def components_(self):
        return self.model.feature_component_correlation
