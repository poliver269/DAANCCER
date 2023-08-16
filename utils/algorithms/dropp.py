import warnings

import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from research_evaluations.plotter import ArrayPlotter, MultiArrayPlotter
from utils import statistical_zero, ordinal
from utils.algorithms import TensorDR
from utils.errors import NonInvertibleEigenvectorException, InvalidComponentNumberException
from utils.math import is_matrix_orthogonal
from utils.matrix_tools import diagonal_block_expand, calculate_symmetrical_kernel_matrix, ensure_matrix_symmetry
from utils.param_keys import N_COMPONENTS, MATRIX_NDIM, TENSOR_NDIM
from utils.param_keys.analyses import CORRELATION_MATRIX_PLOT, EIGENVECTOR_MATRIX_ANALYSE, COVARIANCE_MATRIX_PLOT
from utils.param_keys.kernel_functions import MY_GAUSSIAN, KERNEL_ONLY, KERNEL_DIFFERENCE, KERNEL_MULTIPLICATION
from utils.param_keys.model import *
from utils.param_keys.traj_dims import TIME_DIM, FEATURE_DIM, COMBINED_DIM


class DROPP(TensorDR):
    def __init__(self,
                 cov_stat_func: [callable, str] = np.mean,
                 algorithm_name: str = 'pca',  # pca, tica, kica
                 ndim: int = TENSOR_NDIM,  # 3: tensor, 2: matrix
                 kernel_kwargs: [dict, None] = None,
                 cov_function: [callable, str] = np.cov,  # np.cov, np.corrcoef, co_mad
                 lag_time: int = 0,
                 nth_eigenvector: int = 1,
                 extra_dr_layer: bool = False,
                 abs_eigenvalue_sorting: bool = True,
                 analyse_plot_type: str = '',
                 use_std: bool = True,
                 center_over_time: bool = True
                 ):
        """
        Initialize the DROPP (Dimensionality Reduction for Ordered Points with PCA) model.

        Parameters
        ----------
        cov_stat_func : callable or str, optional
            Statistical function to combine the values of the covariance
            matrices in case of a tensor input. Default is np.mean.
        algorithm_name : str, optional
            Dimensionality reduction algorithm name.
            Choose from 'pca', 'tica', or 'kica'. Default is 'pca'.
        ndim : int, optional
            Number of dimensions for dimensionality reduction.
            Use 3 for tensor or 2 for matrix. Default is TENSOR_NDIM.
        kernel_kwargs : dict or None, optional
            Additional arguments for the kernel function used in the algorithm.
            Default is None.
        cov_function : callable or str, optional
            Function to compute the covariance or correlation matrix.
            Choose from np.cov, np.corrcoef, or 'co_mad'. Default is np.cov.
        lag_time : int, optional
            Lag time for time lagged DROPP analysis. Default is 0.
        nth_eigenvector : int, optional
            Index of the eigenvector to be analyzed. Default is 1.
        extra_dr_layer : bool, optional
            Add an extra dimensionality reduction layer
            instead of combining the covariance matrices of a tensor input.
            Default is False.
        abs_eigenvalue_sorting : bool, optional
            Sort eigenvalues by absolute value. Default is True.
        analyse_plot_type : str, optional
            Type of analysis plot to generate. Default is '' (empty string).
        use_std : bool, optional
            Use standard deviation normalization. Default is True.
            (Preprocessing is still recommended)
        center_over_time : bool, optional
            Center data over time before dimensionality reduction. Default is True.
            (Preprocessing is still recommended)

        Notes
        -----
        - When providing string as cov_function or cov_stat_func,
          make sure to install the necessary dependencies.

        Examples
        --------
        >>> data = np.random.random((1000, 35, 3))
        >>> dr_instance = DROPP()
        >>> dr_instance = dr_instance.fit(data)
        >>> transformed_data = dr_instance.transform(data)

        >>> custom_params = {
        ...     'algorithm_name': 'pca',
        ...     'ndim': 3,
        ...     'kernel_kwargs': {
        ...         'kernel_map': 'only',
        ...         'kernel_function': 'my_gaussian',
        ...         'use_original_data': True
        ...     }
        ... }
        >>> custom_dropp = DROPP(**custom_params)
        >>> custom_dropp = custom_dropp.fit(data)
        >>> transformed_data = custom_dropp.transform(data)
        """
        super().__init__(cov_stat_func)

        self.algorithm_name = algorithm_name
        self.ndim = ndim

        if kernel_kwargs is None:
            kernel_kwargs = {}

        self.kernel_kwargs = {
            KERNEL_MAP: kernel_kwargs.get(KERNEL_MAP, KERNEL_ONLY),  # diff, multi, only, None
            KERNEL_FUNCTION: kernel_kwargs.get(KERNEL_FUNCTION, MY_GAUSSIAN),
            KERNEL_STAT_FUNC: kernel_kwargs.get(KERNEL_STAT_FUNC, statistical_zero),
            USE_ORIGINAL_DATA: kernel_kwargs.get(USE_ORIGINAL_DATA, False),
            CORR_KERNEL: kernel_kwargs.get(CORR_KERNEL, False),
            ONES_ON_KERNEL_DIAG: kernel_kwargs.get(ONES_ON_KERNEL_DIAG, False)
        }

        self.cov_function = cov_function
        self.lag_time = lag_time

        self.nth_eigenvector = nth_eigenvector
        self.extra_dr_layer = extra_dr_layer

        self.abs_eigenvalue_sorting = abs_eigenvalue_sorting
        self.analyse_plot_type = analyse_plot_type
        self.use_std = use_std
        self.center_over_time = center_over_time
        self.__check_init_params__()

    def __check_init_params__(self):
        """
        Check and validate the initialization parameters.

        This method performs checks on the initialization parameters to ensure they are set correctly
        and are consistent with the algorithm's requirements.

        Warnings
        --------
        UserWarning
            - If 'nth_eigenvector' is less than 1, it's adjusted to 1.
            - If 'extra_dr_layer' is True and 'nth_eigenvector' is greater than 1,
              'nth_eigenvector' is ignored as only the first eigenvector is used in this case.
            - If 'algorithm_name' indicates a time-lagged approach, but 'lag_time' is 0.
            - If 'kernel_kwargs[KERNEL_STAT_FUNC]' is a string, it's evaluated to a callable.
              Note: Using 'eval' on user input may be risky. Ensure security.

        Notes
        -----
        Call this method after initializing the instance to ensure the parameters are valid.

        """
        if self.nth_eigenvector < 1:
            self.nth_eigenvector = 1

        if self.extra_dr_layer and self.nth_eigenvector > 1:
            warnings.warn(f"The parameter '{NTH_EIGENVECTOR}' (with value: "
                          f"{self.nth_eigenvector}) is ignored because "
                          f"the parameter '{EXTRA_DR_LAYER}' is set to True. "
                          "When 'extra_dr_layer' is enabled, only the first "
                          "eigenvector will be used.",
                          UserWarning)

        if self._is_time_lagged_algorithm() and self.lag_time == 0:
            warnings.warn(f"The '{ALGORITHM_NAME}' is set to a time-lagged "
                          f"approach ('{self.algorithm_name}'), but the "
                          f"'{LAG_TIME}' parameter is not set. "
                          f"Consider setting a non-zero value for 'lag_time'.",
                          UserWarning)

        if isinstance(self.kernel_kwargs[KERNEL_STAT_FUNC], str):
            warnings.warn(f"The value of the '{KERNEL_STAT_FUNC}' parameter is a string. "
                          "Make sure it corresponds to a valid callable function. "
                          "Using 'eval' on user input may be risky. Please ensure security.",
                          UserWarning)
            self.kernel_kwargs[KERNEL_STAT_FUNC] = eval(self.kernel_kwargs[KERNEL_STAT_FUNC])

    def __str__(self):
        """
        Return a string representation of the DROPP instance.

        This method returns a human-readable string containing key information about the DROPP instance,
        including the number of principal components, lag time (if applicable), and more.

        Returns
        -------
        str
            A string representation of the DROPP instance.

        """
        sb = f"DROPP("
        sb += f"PCs={self.n_components}"
        sb += f", lag-time={self.lag_time}" if self.lag_time > 0 else ""
        sb += (f', kernel_map={self.kernel_kwargs[KERNEL_MAP]}'
               if self.kernel_kwargs[KERNEL_MAP] is not None else '')
        sb += (f', kernel_function={self.kernel_kwargs[KERNEL_FUNCTION]}'
               if self.kernel_kwargs[KERNEL_FUNCTION] != MY_GAUSSIAN else '')
        sb += ")"
        return sb

    @property
    def _is_matrix_model(self) -> bool:
        """
        Check if the dimensionality reduction model uses a matrix representation.

        Returns
        -------
        bool
            True if the dimensionality reduction model uses a matrix representation (ndim = 2), False otherwise.

        """
        return self.ndim == MATRIX_NDIM

    def _is_time_lagged_algorithm(self) -> bool:
        """
        Check if the dimensionality reduction algorithm is time-lagged.

        Returns
        -------
        bool
            True if the algorithm is time-lagged (e.g., 'tica'), False otherwise.

        """
        return self.algorithm_name in ['tica']

    def _use_kernel_as_correlation_matrix(self) -> bool:
        """
        Check if the dimensionality reduction model uses a kernel as the correlation matrix.

        Returns
        -------
        bool
            True if the model uses a kernel as the correlation matrix (e.g., 'kica'), False otherwise.

        """
        return self.algorithm_name in ['kica']

    def _use_correlation_matrix(self) -> bool:
        """
        Check if the dimensionality reduction model uses a correlation matrix.

        Returns
        -------
        bool
            True if the model uses a correlation matrix (e.g., 'kica') or
            is a time-lagged algorithm (e.g., 'tica'), False otherwise.

        """
        return self._use_kernel_as_correlation_matrix() or self._is_time_lagged_algorithm()

    @property
    def use_evs(self) -> bool:
        """
        Determine if EigenVector Selection (EVS) is enabled.

        Returns
        -------
        bool
            True if EigenVector Selection (EVS) is enabled (extra_dr_layer or nth_eigenvector > 1), False otherwise.

        """
        return self.extra_dr_layer or self.nth_eigenvector > 1

    @property
    def _combine_dim(self) -> int:
        """
        The _combine_dim is the size of the (3rd) dimension from the tensor
        which will be combined after calculating the covariance matrix separately.

        Get the size of the combined dimension before calculating the merged covariance matrix.

        Returns
        -------
        int
            Size of the combined dimension (3rd dimension) from the tensor.

        """
        return self._standardized_data.shape[COMBINED_DIM]

    @property
    def _feature_dim(self) -> int:
        """
        Get the size of the feature dimension (2nd dimension) of the data.

        Returns
        -------
        int
            Size of the feature dimension, which represents the size of correlated features in the data.

        """
        return self._standardized_data.shape[FEATURE_DIM]

    def fit_transform(self, data_tensor, **fit_params):
        return super().fit_transform(data_tensor, **fit_params)

    def fit(self, data_tensor, **fit_params):
        """
        Fit the DROPP model to the input data tensor.

        This method performs the fitting process of the DROPP model using the provided input data.
        It calculates the covariance matrix, extracts eigenvectors, and stores the components.

        Parameters
        ----------
        data_tensor : ndarray
            Input data tensor with shape (n_samples, correlation_dim, combine_dim) for tensor data,
            or (n_samples, feature_dim) for matrix data.
        **fit_params : dict
            Additional parameters for the fitting process. Available keys include:
            - 'n_components' (int, optional): Number of components to retain.
              Defaults to 2 if not provided.

        Raises
        ------
        ValueError
            If the input data tensor shape is incompatible with the model type (matrix or tensor).

        Returns
        -------
        self : DROPP
            Returns the instance of the DROPP model after fitting.

        Notes
        -----
        - The resulting eigenvectors are stored in the ´components_´ attribute.
        - Set 'analyse_plot_type' to 'EIGENVECTOR_MATRIX_ANALYSE' to visualize a subset of the eigenvectors.

        Examples
        --------
        >>> dropp_instance = DROPP()
        >>> data = np.random.rand(100, 10, 5)
        >>> dropp_instance.fit(data)
        >>> transformed_data = dropp_instance.transform(data)

        """
        if self._is_matrix_model and data_tensor.ndim != MATRIX_NDIM:
            raise ValueError("The input data tensor shape is incompatible with the model type. "
                             "For tensor data, use shape (n_samples, correlation_dim, combine_dim), "
                             "or for matrix data, use shape (n_samples, feature_dim).")

        self.n_samples = data_tensor.shape[TIME_DIM]
        self.n_components = fit_params.get(N_COMPONENTS, 2)
        self._standardized_data = self._standardize_data(data_tensor)
        self._covariance_matrix = self.get_covariance_matrix()
        eigenvectors = self.get_eigenvectors()
        self.components_ = eigenvectors[:, :self.n_components].T
        if self.analyse_plot_type == EIGENVECTOR_MATRIX_ANALYSE:
            ArrayPlotter(
                interactive=False,
                title_prefix=EIGENVECTOR_MATRIX_ANALYSE,
                x_label='Eigenvector Number',
                y_label='Eigenvector Dimension',
                xtick_start=1,
                for_paper=True
            ).matrix_plot(eigenvectors[:12, :15], show_values=True)
        return self

    def _standardize_data(self, tensor):
        """
        Standardize the input tensor data.

        This method standardizes the input tensor data by centering it
        and optionally scaling it by the standard deviation.

        Parameters
        ----------
        tensor : ndarray
            Input data tensor with shape (n_samples, feature_dim, combine_dim) for tensor data,
            or (n_samples, feature_dim) for matrix data.

        Returns
        -------
        standardized_data : ndarray
            Standardized tensor data with the same shape as the input tensor.

        Notes
        -----
        - If 'use_std' is True, the data is scaled by the standard deviation.
        - If 'use_std' is False, the data is only centered.

        Examples
        --------
        >>> dropp_instance = DROPP()
        >>> tensor_data = np.random.rand(100, 10, 5)  # Example tensor data
        >>> standardized_tensor = dropp_instance._standardize_data(tensor_data)

        """
        centered_data = self._center_data(tensor)

        if self.use_std:
            self._std = np.std(tensor, axis=0)
        else:
            self._std = 1
        return centered_data / self._std

    def _center_data(self, tensor):
        """
        Center the input data tensor by subtracting the mean vector.

        Parameters
        ----------
        tensor : ndarray
            Input data tensor with shape (n_samples, feature_dim, combine_dim) for tensor data,
            or (n_samples, feature_dim) for matrix data.

        Returns
        -------
        centered_data : ndarray
            Data tensor or matrix after centering.

        Notes
        -----
        - For matrix data or when 'center_over_time' is False, the mean vector is computed along axis 0.
        - For tensor data and when 'center_over_time' is True,
          the mean tensor is computed along axis 0 (time dimension).
          The mean tensor is broadcasted to match the shape of the input tensor for subtraction.

        """
        if self._is_matrix_model or not self.center_over_time:
            self.mean = np.mean(tensor, axis=0)
        else:
            # self.mean = np.mean(tensor, axis=1)[:, np.newaxis, :]
            self.mean = np.mean(tensor, axis=0)[np.newaxis, :, :]
        return tensor - self.mean

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Compute and return the covariance matrix.

        The calculation of the covariance matrix depends on the dimensionality (ndim) of the model:
        - For matrix data, the covariance matrix is calculated conventionally.
        - For tensor data, the covariance matrix is calculated over the combined dimension, and not correlated
          values are enforced to have zero correlations by block expanding and setting off-diagonal values to zeros.
          The DROPP algorithm maps a Gaussian curve onto the covariance matrix by default.

        Returns
        -------
        cov_matrix : np.ndarray
            Covariance matrix with shape (_feature_dim*_combined_dim x _feature_dim*_combined_dim)
            or (_feature_dim x _feature_dim) if the model is matrix-based.

        Notes
        -----
        - The 'kernel_kwargs[KERNEL_MAP]' parameter can modify the covariance matrix using a kernel function.
        - Set 'kernel_kwargs[KERNEL_MAP]' to None to disable kernel mapping.
        - The covariance matrix is used in the DROPP algorithm to capture correlations between features.
        """
        if self._is_matrix_model:
            cov = self._get_matrix_covariance()
            if self.kernel_kwargs[KERNEL_MAP] is not None and not self._use_kernel_as_correlation_matrix():
                cov = self._map_kernel(cov)
            return cov
        else:
            ccm = self.get_combined_covariance_matrix()
            if self.kernel_kwargs[KERNEL_MAP] is not None and not self._use_kernel_as_correlation_matrix():
                ccm = self._map_kernel(ccm)
            return diagonal_block_expand(ccm, self._combine_dim)

    def _get_matrix_covariance(self) -> np.ndarray:
        """
        Calculate the covariance matrix for standardized data in matrix shape.

        Depending on the algorithm and lag_time, this method computes either a standard covariance matrix
        or a time-lagged covariance matrix by truncating the lag_time from the matrix.

        Returns
        -------
        covariance_matrix : np.ndarray
            Covariance matrix for matrix-shaped standardized data.

        Notes
        -----
        - If the algorithm is time-lagged (e.g., 'tica') and lag_time is greater than 0,
          the time-lagged covariance matrix is calculated.
        - If not time-lagged, the standard covariance matrix is used.

        """
        if self._is_time_lagged_algorithm() and self.lag_time > 0:
            return np.cov(self._standardized_data[:-self.lag_time].T)
        else:
            return super().get_covariance_matrix()

    def get_combined_covariance_matrix(self):
        """
        Calculate the combined covariance matrix.

        This method computes the combined covariance matrix from the tensor covariance matrix,
        using the specified statistic function for the covariance matrix along axis 0.

        Returns
        -------
        combined_cov_matrix : np.ndarray
            Combined covariance matrix resulting from applying the covariance statistic function.

        Notes
        -----
        - The 'cov_stat_func' parameter determines the statistical function applied along axis 0.
        - If 'analyse_plot_type' is set to 'COVARIANCE_MATRIX_PLOT', a plot of tensor layers and the combined
          covariance matrix is generated.

        """
        tensor_cov = self.get_tensor_covariance()
        cov = self.cov_stat_func(tensor_cov, axis=0)
        if self.analyse_plot_type == COVARIANCE_MATRIX_PLOT:
            MultiArrayPlotter().plot_tensor_layers(tensor_cov, cov, 'Covariance')
        return cov

    def get_tensor_covariance(self):
        if self._is_time_lagged_algorithm() and self.lag_time > 0:
            return np.asarray(list(
                map(lambda index: self.cov_function(
                    self._standardized_data[:-self.lag_time, :, index].T),
                    range(self._combine_dim))
            ))
        else:
            return np.asarray(list(
                map(lambda index: self.cov_function(self._standardized_data[:, :, index].T),
                    range(self._combine_dim))
            ))

    def _map_kernel(self, matrix):
        kernel_matrix = calculate_symmetrical_kernel_matrix(
            matrix,
            flattened=self._is_matrix_model,
            analyse_mode=self.analyse_plot_type,
            **self.kernel_kwargs
        )
        if self.kernel_kwargs[KERNEL_MAP] == KERNEL_ONLY:
            matrix = kernel_matrix
        elif self.kernel_kwargs[KERNEL_MAP] == KERNEL_DIFFERENCE:
            matrix -= kernel_matrix
        elif self.kernel_kwargs[KERNEL_MAP] == KERNEL_MULTIPLICATION:
            matrix *= kernel_matrix

        if self.kernel_kwargs[ONES_ON_KERNEL_DIAG]:
            np.fill_diagonal(matrix, 1.0)

        return matrix

    def get_eigenvectors(self):
        # calculate eigenvalues & eigenvectors of covariance matrix
        # assert is_matrix_symmetric(self._covariance_matrix), 'Covariance-Matrix should be symmetric.'
        if self.algorithm_name in ['tica', 'kica']:
            correlation_matrix = self._get_correlations_matrix()
            # assert is_matrix_symmetric(correlation_matrix), 'Correlation-Matrix should be symmetric.'
            # noinspection PyTupleAssignmentBalance
            eigenvalues, eigenvectors = scipy.linalg.eig(correlation_matrix, b=self._covariance_matrix)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        if self.abs_eigenvalue_sorting:
            eigenvalues = np.abs(eigenvalues)

        if np.any(np.iscomplex(eigenvalues)):
            eigenvalues = np.abs(eigenvalues)

        # sort eigenvalues descending
        sorted_eigenvalue_indexes = np.argsort(eigenvalues)[::-1]
        self.explained_variance_ = np.real_if_close(eigenvalues[sorted_eigenvalue_indexes])
        eigenvectors = np.real_if_close(eigenvectors[:, sorted_eigenvalue_indexes])

        if self.extra_dr_layer:
            return self._get_eigenvectors_with_dr_layer(eigenvectors)
        else:
            self.explained_variance_ = self.explained_variance_[::self.nth_eigenvector]
            return eigenvectors[:, ::self.nth_eigenvector]

    def _get_eigenvectors_with_dr_layer(self, eigenvectors):
        eigenvalues2 = []
        eigenvectors2 = []
        for component in range(self._feature_dim):
            vector_from = component * self._combine_dim
            vector_to = (component + 1) * self._combine_dim
            model = PCA(n_components=1)
            model.fit_transform(eigenvectors[:, vector_from:vector_to])

            # No idea if it makes sense TODO
            ew2 = model.explained_variance_[0]
            # eigenvalues2.append(np.mean(self.eigenvalues[vector_from:vector_to] * ew2))
            eigenvalues2.append(np.sum(self.explained_variance_[vector_from:vector_to] * ew2))

            ev2 = model.components_[0]
            ev = np.dot(eigenvectors[:, vector_from:vector_to], ev2)
            eigenvectors2.append(ev)

        self.explained_variance_ = np.asarray(eigenvalues2).T
        return np.asarray(eigenvectors2).T

    def _get_correlations_matrix(self):
        if self._is_matrix_model:
            corr = self._get_matrix_correlation()

            if self.kernel_kwargs[CORR_KERNEL] or self._use_kernel_as_correlation_matrix():
                corr = self._map_kernel(corr)

            return corr
        else:
            tensor_corr = self._get_tensor_correlation()
            corr = self.cov_stat_func(tensor_corr, axis=0)

            if self.analyse_plot_type == CORRELATION_MATRIX_PLOT:
                MultiArrayPlotter().plot_tensor_layers(tensor_corr, corr, 'Correlation')

            if self.kernel_kwargs[CORR_KERNEL] or self._use_kernel_as_correlation_matrix():
                corr = self._map_kernel(corr)

            return diagonal_block_expand(corr, tensor_corr.shape[0])

    def _get_matrix_correlation(self):
        if self.lag_time <= 0:
            return self._get_matrix_covariance()
        else:
            corr = np.dot(self._standardized_data[:-self.lag_time].T,
                          self._standardized_data[self.lag_time:]) / (self.n_samples - self.lag_time)
            return ensure_matrix_symmetry(corr)

    def _get_tensor_correlation(self):
        if self._use_kernel_as_correlation_matrix() or self.lag_time <= 0:
            return self.get_tensor_covariance()
        else:
            temp_list = []
            for index in range(self._combine_dim):
                dot_i = np.dot(self._standardized_data[:-self.lag_time, :, index].T,
                               self._standardized_data[self.lag_time:, :, index]) / (
                                self.n_samples - self.lag_time)
                sym_i = ensure_matrix_symmetry(dot_i)
                temp_list.append(sym_i)
            return np.asarray(temp_list)

    def transform(self, data_tensor):
        data_tensor_standardized = self._standardize_data(data_tensor)
        data_matrix = self.convert_to_matrix(data_tensor_standardized)
        return np.dot(data_matrix, self.components_.T)

    def convert_to_matrix(self, tensor):
        if self._is_matrix_model:
            return tensor
        else:
            return super().convert_to_matrix(tensor)

    def convert_to_tensor(self, matrix):
        if self._is_matrix_model:
            return matrix
        else:
            return super().convert_to_tensor(matrix)

    def inverse_transform(self, projection_data: np.ndarray, component_count: int):
        if is_matrix_orthogonal(self.components_.T):
            return np.dot(
                projection_data,
                self.components_[:component_count]
            )  # Da orthogonal --> Transform = inverse
        else:
            if self.use_evs:
                raise NonInvertibleEigenvectorException('Eigenvectors are Non-Orthogonal and Non-Squared. ')
            else:
                return np.dot(
                    projection_data,
                    np.linalg.inv(self.components_.T)[:component_count]
                )

    def reconstruct(self, projection_matrix, component_count=None):
        if component_count is None:
            component_count = self._feature_dim * self._combine_dim
        elif component_count > self.components_.shape[0]:
            raise InvalidComponentNumberException(f'Model does not have {component_count} many components. '
                                                  f'Max: {self.components_.shape[0]}')

        inverse_matrix = self.inverse_transform(projection_matrix, component_count)

        reconstructed_tensor = self.convert_to_tensor(inverse_matrix)
        if self.use_std:
            reconstructed_tensor *= self._std
        reconstructed_tensor += self.mean

        return reconstructed_tensor

    def score(self, data_tensor, y=None):
        """
        Reconstruct data and calculate the root mean squared error (RMSE).
        See Notes: https://stats.stackexchange.com/q/229093

        Parameters
        ----------
        data_tensor : (X) array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """

        if y is not None:  # "use" variable, to not have a PyCharm warning
            data_projection = y
        else:
            data_projection = self.transform(data_tensor)

        reconstructed_tensor = self.reconstruct(data_projection, self.n_components)

        data_matrix = self.convert_to_matrix(data_tensor)
        reconstructed_matrix = self.convert_to_matrix(reconstructed_tensor)

        return mean_squared_error(data_matrix, reconstructed_matrix, squared=False)
