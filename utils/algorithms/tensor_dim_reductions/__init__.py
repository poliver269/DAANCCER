import numpy as np
import scipy

from plotter import ArrayPlotter
from utils.algorithms import MyModel
import pyemma.coordinates as coor
from utils.matrix_tools import diagonal_block_expand, calculate_symmetrical_kernel_from_matrix, ensure_matrix_symmetry
from utils.param_key import *


class TensorDR(MyModel):
    def __init__(self, model_parameters=None):
        super().__init__()
        if model_parameters is None:
            model_parameters = {}

        self.params = {
            COV_STAT_FUNC: model_parameters.get(COV_STAT_FUNC, np.mean),
            KERNEL_STAT_FUNC: model_parameters.get(KERNEL_STAT_FUNC, np.median)
        }

    def fit(self, data_tensor, **fit_params):
        self.n_samples = data_tensor.shape[0]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self._update_cov()
        self.eigenvectors = self.get_eigenvectors()
        return self

    def get_covariance_matrix(self):
        return super().get_covariance_matrix()

    def _update_cov(self):
        averaged_cov = self.params[COV_STAT_FUNC](self._covariance_matrix, axis=0)
        self._covariance_matrix = diagonal_block_expand(averaged_cov, self._covariance_matrix.shape[0])

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_eigenvalue_indexes]
        return eigenvectors[:, sorted_eigenvalue_indexes]

    def transform(self, data_tensor):
        data_matrix = self._update_data_tensor(data_tensor)
        return super(TensorDR, self).transform(data_matrix)

    @staticmethod
    def _update_data_tensor(data_tensor):
        return data_tensor.reshape(data_tensor.shape[0], data_tensor.shape[1] * data_tensor.shape[2])


class ParameterModel(TensorDR):
    def __init__(self, model_parameters):
        super().__init__(model_parameters)
        self.params.update({
            ALGORITHM_NAME: model_parameters.get(ALGORITHM_NAME, 'pca'),  # pca, tica, kica
            NDIM: model_parameters.get(NDIM, TENSOR_NDIM),  # 3: tensor, 2: matrix

            KERNEL: model_parameters.get(KERNEL, None),  # diff, multi, only, None
            CORR_KERNEL: model_parameters.get(CORR_KERNEL, False),  # only for tica: True, False
            KERNEL_TYPE: model_parameters.get(KERNEL_TYPE, MY_GAUSSIAN),
            ONES_ON_KERNEL_DIAG: model_parameters.get(ONES_ON_KERNEL_DIAG, False),

            COV_FUNCTION: model_parameters.get(COV_FUNCTION, np.cov),  # np.cov, np.corrcoef, co_mad
            LAG_TIME: model_parameters.get(LAG_TIME, 0),
            NTH_EIGENVECTOR: model_parameters.get(NTH_EIGENVECTOR, 1),
            EXTRA_DR_LAYER: model_parameters.get(EXTRA_DR_LAYER, False),
            EXTRA_LAYER_ON_PROJECTION: model_parameters.get(EXTRA_LAYER_ON_PROJECTION, True),
            ABS_EVAL_SORT: model_parameters.get(ABS_EVAL_SORT, True),

            PLOT_2D: model_parameters.get(PLOT_2D, False),
            USE_STD: model_parameters.get(USE_STD, False),
            CENTER_OVER_TIME: model_parameters.get(CENTER_OVER_TIME, True),
        })

    def __str__(self):
        sb = f'{"Matrix" if self._is_matrix_model else "Tensor"}-{self.params[ALGORITHM_NAME]}'
        if self.params[KERNEL] is not None:
            sb += (f', {self.params[KERNEL_TYPE]}-{self.params[KERNEL]}' +
                   f'{f"-onCorr2" if self.params[CORR_KERNEL] else ""}')
        sb += '\n'
        sb += f'lag-time={self.params[LAG_TIME]}, ' if self.params[LAG_TIME] > 0 else ''
        sb += f'abs_ew_sorting={self.params[ABS_EVAL_SORT]}, '
        sb += f'2nd_layer={self.params[EXTRA_DR_LAYER]}' if self.params[EXTRA_DR_LAYER] else ''
        sb += f'\nn-th_ev={self.params[NTH_EIGENVECTOR]}' if self.params[NTH_EIGENVECTOR] > 1 else ''
        return sb
        # f'{function_name(self.params[COV_FUNCTION])}'

    @property
    def _is_matrix_model(self) -> bool:
        return self.params[NDIM] == MATRIX_NDIM

    def _is_time_lagged_algorithm(self) -> bool:
        return self.params[ALGORITHM_NAME] in ['tica']

    def _use_kernel_as_correlations_matrix(self) -> bool:
        return self.params[ALGORITHM_NAME] in ['kica']

    def _use_correlations_matrix(self) -> bool:
        return self._use_kernel_as_correlations_matrix() or self._is_time_lagged_algorithm()

    @property
    def _combine_dim(self) -> int:
        return self._standardized_data.shape[COORDINATE_DIM]

    @property
    def _atom_dim(self) -> int:
        return self._standardized_data.shape[ATOM_DIM]

    def fit_transform(self, data_tensor, **fit_params):
        return super().fit_transform(data_tensor, fit_params)

    def fit(self, data_tensor, **fit_params):
        self.n_samples = data_tensor.shape[TIME_DIM]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        self.eigenvectors = self.get_eigenvectors()
        return self

    def _standardize_data(self, tensor):
        if self._is_matrix_model or not self.params[CENTER_OVER_TIME]:
            numerator = tensor - np.mean(tensor, axis=0)
        else:
            numerator = tensor - np.mean(tensor, axis=1)[:, np.newaxis, :]

        if self.params[USE_STD]:
            denominator = np.std(tensor, axis=0)
            return numerator / denominator
        else:
            return numerator

    def get_covariance_matrix(self):
        if self._is_matrix_model:
            cov = self._get_matrix_covariance()
            if self.params[KERNEL] is not None and not self._use_kernel_as_correlations_matrix():
                cov = self._map_kernel(cov)
            return cov
        else:
            tensor_cov = self._get_tensor_covariance()
            cov = self.params[COV_STAT_FUNC](tensor_cov, axis=0)
            if self.params[KERNEL] is not None and not self._use_kernel_as_correlations_matrix():
                cov = self._map_kernel(cov)
            return diagonal_block_expand(cov, tensor_cov.shape[0])

    def _get_matrix_covariance(self):
        if self._is_time_lagged_algorithm() and self.params[LAG_TIME] > 0:
            return np.cov(self._standardized_data[:-self.params[LAG_TIME]].T)
        else:
            return super().get_covariance_matrix()

    def _get_tensor_covariance(self):
        if self._is_time_lagged_algorithm() and self.params[LAG_TIME] > 0:
            return np.asarray(list(
                map(lambda index: self.params[COV_FUNCTION](
                    self._standardized_data[:-self.params[LAG_TIME], :, index].T),
                    range(self._combine_dim))
            ))
        else:
            return np.asarray(list(
                map(lambda index: self.params[COV_FUNCTION](self._standardized_data[:, :, index].T),
                    range(self._combine_dim))
            ))

    def _map_kernel(self, matrix):
        trajectory_name = 'Not Model Related' if self.params[PLOT_2D] else None
        kernel_matrix = calculate_symmetrical_kernel_from_matrix(
            matrix, self.params[KERNEL_STAT_FUNC], self.params[KERNEL_TYPE],
            trajectory_name, flattened=self._is_matrix_model)
        if self.params[KERNEL] == KERNEL_ONLY:
            matrix = kernel_matrix
        elif self.params[KERNEL] == KERNEL_DIFFERENCE:
            matrix -= kernel_matrix
        elif self.params[KERNEL] == KERNEL_MULTIPLICATION:
            matrix *= kernel_matrix

        if self.params[ONES_ON_KERNEL_DIAG]:
            np.fill_diagonal(matrix, 1.0)

        return matrix

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        # assert is_matrix_symmetric(self._covariance_matrix), 'Covariance-Matrix should be symmetric.'
        if self.params[ALGORITHM_NAME] in ['tica', 'kica']:
            correlation_matrix = self._get_correlations_matrix()
            # assert is_matrix_symmetric(correlation_matrix), 'Correlation-Matrix should be symmetric.'
            eigenvalues, eigenvectors = scipy.linalg.eig(correlation_matrix, b=self._covariance_matrix)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        if self.params[ABS_EVAL_SORT]:
            eigenvalues = np.abs(eigenvalues)

        if np.any(np.iscomplex(eigenvalues)):
            eigenvalues = np.abs(eigenvalues)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = np.real_if_close(eigenvalues[sorted_eigenvalue_indexes])
        return np.real_if_close(eigenvectors[:, sorted_eigenvalue_indexes])

    def _get_correlations_matrix(self):
        if self._is_matrix_model:
            corr = self._get_matrix_correlation()

            if self.params[CORR_KERNEL] or self._use_kernel_as_correlations_matrix():
                corr = self._map_kernel(corr)

            return corr
        else:
            tensor_corr = self._get_tensor_correlation()
            corr = self.params[COV_STAT_FUNC](tensor_corr, axis=0)

            if self.params[PLOT_2D]:
                for i in range(tensor_corr.shape[0]):
                    ArrayPlotter(False).matrix_plot(tensor_corr[i])
                ArrayPlotter(False).matrix_plot(corr)

            if self.params[CORR_KERNEL] or self._use_kernel_as_correlations_matrix():
                corr = self._map_kernel(corr)

            return diagonal_block_expand(corr, tensor_corr.shape[0])

    def _get_matrix_correlation(self):
        if self.params[LAG_TIME] <= 0:
            return self._get_matrix_covariance()
        else:
            corr = np.dot(self._standardized_data[:-self.params[LAG_TIME]].T,
                          self._standardized_data[self.params[LAG_TIME]:]) / (self.n_samples - self.params[LAG_TIME])
            return ensure_matrix_symmetry(corr)

    def _get_tensor_correlation(self):
        if self._use_kernel_as_correlations_matrix() or self.params[LAG_TIME] <= 0:
            return self._get_tensor_covariance()
        else:
            temp_list = []
            for index in range(self._standardized_data.shape[COORDINATE_DIM]):
                dot_i = np.dot(self._standardized_data[:-self.params[LAG_TIME], :, index].T,
                               self._standardized_data[self.params[LAG_TIME]:, :, index]) / (
                                self.n_samples - self.params[LAG_TIME])
                sym_i = ensure_matrix_symmetry(dot_i)
                temp_list.append(sym_i)
            return np.asarray(temp_list)

    def transform(self, data_tensor):
        if self._is_matrix_model:
            data_matrix = data_tensor
        else:
            data_matrix = self._update_data_tensor(data_tensor)

        if self.params[NTH_EIGENVECTOR] < 1:
            self.params[NTH_EIGENVECTOR] = self._combine_dim

        if self.params[EXTRA_DR_LAYER]:
            return self.transform_with_extra_layer(data_matrix)
        else:
            return np.dot(
                data_matrix,
                self.eigenvectors[:, :self.n_components * self.params[NTH_EIGENVECTOR]:self.params[NTH_EIGENVECTOR]]
            )

    def transform_with_extra_layer(self, data_matrix):
        if self.params[EXTRA_LAYER_ON_PROJECTION]:
            proj1 = np.dot(data_matrix, self.eigenvectors)
        else:
            proj1 = self.eigenvectors
        proj2 = []
        eigenvalues2 = []
        eigenvectors2 = []
        for component in range(self._atom_dim):
            vector_from = component * self._combine_dim
            vector_to = (component + 1) * self._combine_dim
            model = coor.pca(data=proj1[:, vector_from:vector_to], dim=1)

            proj2.append(np.squeeze(model.get_output()[0]))

            ew2 = model.eigenvalues[0]
            eigenvalues2.append(np.mean(self.eigenvalues[vector_from:vector_to] * ew2))

            ev2 = model.eigenvectors[0]
            ev = np.dot(self.eigenvectors[:, vector_from:vector_to], ev2)
            eigenvectors2.append(ev)

        self.eigenvalues = np.asarray(eigenvalues2).T
        self.eigenvectors = np.asarray(eigenvectors2).T

        if self.params[EXTRA_LAYER_ON_PROJECTION]:
            return np.asarray(proj2[:self.n_components]).T
        else:
            return np.dot(data_matrix, self.eigenvectors[:, :self.n_components])
