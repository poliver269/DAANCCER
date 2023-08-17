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

        if self._is_time_lagged_model and self.lag_time == 0:
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

    @property
    def _is_time_lagged_model(self) -> bool:
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
        return self._use_kernel_as_correlation_matrix() or self._is_time_lagged_model

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
                cov = self._map_kernel_on(cov)
            return cov
        else:
            ccm = self.get_combined_covariance_matrix()
            if self.kernel_kwargs[KERNEL_MAP] is not None and not self._use_kernel_as_correlation_matrix():
                ccm = self._map_kernel_on(ccm)
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
        if self._is_time_lagged_model and self.lag_time > 0:
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
        tensor_cov = self.get_covariance_tensor()
        cov = self.cov_stat_func(tensor_cov, axis=0)
        if self.analyse_plot_type == COVARIANCE_MATRIX_PLOT:
            MultiArrayPlotter().plot_tensor_layers(tensor_cov, cov, 'Covariance')
        return cov

    def get_covariance_tensor(self):
        """
        Calculate the tensor covariance tensor.

        This method computes the covariance tensor or a time-lagged covariance tensor for tensor data,
        where the tensor dimensions represent samples, features, and combined dimensions.
        The calculation depends on the model and lag_time.

        Returns
        -------
        tensor_cov_tensor : np.ndarray
            Covariance tensor with shape (_combined_dim, _feature_dim, _feature_dim)
            if lag_time is used and the model is time-lagged the samples of the tensor
            is truncated by the lag_time.

        Notes
        -----
        - The covariance tensor is calculated by applying the specified covariance function
          (e.g. np.cov) along axis 0 for each combined dimension.
        - If the algorithm is time-lagged (e.g., 'tica') and lag_time is greater than 0,
          the covariance tensor is calculated over the truncated data.
        - If not time-lagged, the covariance tensor is calculated using the full data.

        """
        if self._is_time_lagged_model and self.lag_time > 0:
            cov_indices = slice(None, -self.lag_time)
        else:
            cov_indices = slice(None)

        tensor_data = self._standardized_data[cov_indices]
        return np.asarray([
            self.cov_function(tensor_data[:, :, index].T) for index in range(self._combine_dim)
        ])

    def _map_kernel_on(self, covariance_matrix):
        """
        Apply kernel mapping to the input covariance matrix.

        This method applies kernel mapping to the input covariance matrix based on the specified kernel mapping mode.
        The kernel mapping can involve operations such as kernel matrix calculation, subtraction, multiplication,
        and diagonal modification.

        Parameters
        ----------
        covariance_matrix : np.ndarray
           Input covariance matrix to which kernel mapping is applied.

        Returns
        -------
        mapped_matrix : np.ndarray
           Covariance matrix after kernel mapping has been applied.

        Notes
        -----
        - Kernel mapping can modify the input covariance matrix according to the specified kernel mapping mode.
        - The kernel mapping is determined by the 'kernel_kwargs' attribute.

        """
        kernel_matrix = calculate_symmetrical_kernel_matrix(
            covariance_matrix,
            flattened=self._is_matrix_model,
            analyse_mode=self.analyse_plot_type,
            **self.kernel_kwargs
        )
        if self.kernel_kwargs[KERNEL_MAP] == KERNEL_ONLY:
            covariance_matrix = kernel_matrix
        elif self.kernel_kwargs[KERNEL_MAP] == KERNEL_DIFFERENCE:
            covariance_matrix -= kernel_matrix
        elif self.kernel_kwargs[KERNEL_MAP] == KERNEL_MULTIPLICATION:
            covariance_matrix *= kernel_matrix

        if self.kernel_kwargs[ONES_ON_KERNEL_DIAG]:
            np.fill_diagonal(covariance_matrix, 1.0)

        return covariance_matrix

    def get_eigenvectors(self):
        """
        Calculate the eigenvectors of the covariance matrix.

        This method computes the eigenvectors of the covariance matrix based on the specified algorithm.
        The eigenvalues and eigenvectors can be sorted and processed according to various settings.

        Returns
        -------
        eigenvectors : np.ndarray
            Eigenvectors of the covariance matrix, either with or without dimensionality reduction layer.

        Notes
        -----
        - The eigenvalues and eigenvectors of the covariance matrix are computed based on the chosen algorithm.
        - If the algorithm is 'tica' or 'kica', the correlation matrix is calculated and used to compute
          the eigenvalues and eigenvectors.
        - Eigenvalues are sorted in descending order, and the eigenvectors are reordered accordingly.
        - The 'abs_eigenvalue_sorting' option determines whether to sort eigenvalues by absolute values or
          if complex eigenvalues are encountered, their absolute values are used.
        - If 'extra_dr_layer' is enabled, an additional dimensionality reduction layer is applied to the eigenvectors.

        """

        if self.algorithm_name in ['tica', 'kica']:
            correlation_matrix = self._get_correlations_matrix()
            eigenvalues, eigenvectors = scipy.linalg.eig(correlation_matrix, b=self._covariance_matrix)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(self._covariance_matrix)

        if self.abs_eigenvalue_sorting or np.any(np.iscomplex(eigenvalues)):
            eigenvalues = np.abs(eigenvalues)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.explained_variance_ = np.real_if_close(eigenvalues[sorted_indices])
        eigenvectors = np.real_if_close(eigenvectors[:, sorted_indices])

        if self.extra_dr_layer:
            return self._get_eigenvectors_with_dr_layer(eigenvectors)
        else:
            self.explained_variance_ = self.explained_variance_[::self.nth_eigenvector]
            return eigenvectors[:, ::self.nth_eigenvector]

    def _get_eigenvectors_with_dr_layer(self, eigenvectors):
        """
        Apply an additional dimensionality reduction layer to the eigenvectors.

        This method applies a dimensionality reduction (DR) layer to the given eigenvectors to further reduce their
        dimensionality while capturing important information. The DR layer is applied per feature, and its effect
        on eigenvalues is reflected.

        Parameters
        ----------
        eigenvectors : np.ndarray
            Eigenvectors of the covariance matrix.

        Returns
        -------
        reduced_eigenvectors : np.ndarray
            Eigenvectors after the additional dimensionality reduction layer has been applied.

        Notes
        -----
        - The dimensionality reduction (DR) layer is applied to each feature's eigenvectors independently.
        - The DR layer uses PCA (Principal Component Analysis) with n_components=1 for each feature.
        - The DR layer modifies the eigenvalues and eigenvectors to capture essential information
          while reducing the dimensionality.

        """
        eigenvalues2 = []
        eigenvectors2 = []
        for component in range(self._feature_dim):
            vector_from = component * self._combine_dim
            vector_to = (component + 1) * self._combine_dim
            model = PCA(n_components=1)
            model.fit_transform(eigenvectors[:, vector_from:vector_to])

            ew2 = model.explained_variance_[0]
            # eigenvalues2.append(np.mean(self.eigenvalues[vector_from:vector_to] * ew2))
            eigenvalues2.append(np.sum(self.explained_variance_[vector_from:vector_to] * ew2))

            ev2 = model.components_[0]
            ev = np.dot(eigenvectors[:, vector_from:vector_to], ev2)
            eigenvectors2.append(ev)

        self.explained_variance_ = np.asarray(eigenvalues2).T
        return np.asarray(eigenvectors2).T

    def _get_correlations_matrix(self):
        """
        Calculate the correlations matrix based on the model's configuration.

        This method computes the correlations matrix based on the data's nature (matrix or tensor) and the chosen
        kernel mapping mode. The correlations matrix can involve operations such as correlation matrix calculation,
        kernel mapping, and diagonal expansion.

        Returns
        -------
        corr_matrix : np.ndarray
            Correlations matrix with shape (_feature_dim*_combined_dim - lag_time,
                                            _feature_dim*_combined_dim - lag_time)
            if using kernel mapping or diagonal expansion.
            Otherwise, the shape is (_feature_dim - lag_time, _feature_dim - lag_time).

        Notes
        -----
        - The correlations matrix is calculated differently based on whether the data is in matrix or tensor form.
        - If using matrix data, the correlations matrix is calculated directly or kernel-mapped and returned.
        - If using tensor data, the tensor correlation matrix is calculated, and statistical functions
          are applied along axis 0.
        - Kernel mapping can modify the correlations matrix according to the specified algorithm mode.

        """

        if self._is_matrix_model:
            corr = self._get_matrix_correlation()

            if self.kernel_kwargs[CORR_KERNEL] or self._use_kernel_as_correlation_matrix():
                corr = self._map_kernel_on(corr)

            return corr
        else:
            tensor_corr = self._get_tensor_correlation()
            corr = self.cov_stat_func(tensor_corr, axis=0)

            if self.analyse_plot_type == CORRELATION_MATRIX_PLOT:
                MultiArrayPlotter().plot_tensor_layers(tensor_corr, corr, 'Correlation')

            if self.kernel_kwargs[CORR_KERNEL] or self._use_kernel_as_correlation_matrix():
                corr = self._map_kernel_on(corr)

            return diagonal_block_expand(corr, tensor_corr.shape[0])

    def _get_matrix_correlation(self):
        """
        Calculate the matrix correlation based on the model's configuration.

        This method computes the matrix correlation based on the data's nature and lag time. The matrix correlation
        captures the linear relationship between features in the data matrix.

        Returns
        -------
        corr_matrix : np.ndarray
            Matrix correlation with shape (_feature_dim, _feature_dim).

        Notes
        -----
        - The matrix correlation is calculated by computing the dot product of standardized data slices
          with and without lag time and normalizing by the number of samples.
        - If lag time is not used (lag_time <= 0), the matrix covariance is returned instead.
        - The resulting matrix correlation is ensured to be symmetric.

        """

        if self.lag_time <= 0:
            return self._get_matrix_covariance()
        else:
            corr = np.dot(self._standardized_data[:-self.lag_time].T,
                          self._standardized_data[self.lag_time:]) / (self.n_samples - self.lag_time)
            return ensure_matrix_symmetry(corr)

    def _get_tensor_correlation(self):
        """
        Calculate the tensor correlation based on the model's configuration.

        This method computes the tensor correlation based on the data's nature, lag time, and kernel mapping mode.
        The tensor correlation captures the linear relationships between features across time steps.

        Returns
        -------
        tensor_corr : np.ndarray
            Tensor correlation with shape (_combined_dim, _feature_dim, _feature_dim)
            if lag_time is used and the model is time-lagged the samples of the tensor
            is truncated by the lag_time.

        Notes
        -----
        - The tensor correlation is calculated based on the standardized data slices with and without lag time.
        - If kernel mapping is used for the correlations matrix or lag time is not used (lag_time <= 0),
          the covariance tensor is returned instead.
        - The resulting tensor correlation is ensured to have symmetric matrices for each combined dimension.

        """

        if self._use_kernel_as_correlation_matrix() or self.lag_time <= 0:
            return self.get_covariance_tensor()
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
        """
        Transform input data tensor or matrix into a reduced-dimensional representation.

        This method performs the dimensionality reduction transformation on the input data tensor or matrix using
        the learned components of the DROPP model.

        Parameters
        ----------
        data_tensor : np.ndarray
            Input data tensor or matrix with shape (n_samples, _feature_dim, _combined_dim) for tensor data,
            or (n_samples, _feature_dim) for matrix data.

        Returns
        -------
        transformed_data : np.ndarray
            Reduced-dimensional representation of the input data tensor or matrix with shape
            (n_samples, n_components).

        """
        data_tensor_standardized = self._standardize_data(data_tensor)
        data_matrix = self.convert_to_matrix(data_tensor_standardized)
        return np.dot(data_matrix, self.components_.T)

    def convert_to_matrix(self, tensor):
        """
        Convert input data tensor to a matrix representation if required.

        This method converts the input data tensor to a matrix representation based on the model's configuration.

        Parameters
        ----------
        tensor : np.ndarray
            Input data tensor or matrix with shape (n_samples, _feature_dim, _combined_dim) for tensor data,
            or (n_samples, _feature_dim) for matrix data.

        Returns
        -------
        matrix : np.ndarray
            Matrix representation of the input data tensor.

        """
        if self._is_matrix_model:
            return tensor
        else:
            return super().convert_to_matrix(tensor)

    def convert_to_tensor(self, matrix):
        """
        Convert input data matrix to a tensor representation if required.

        This method converts the input data matrix to a tensor representation based on the model's configuration.

        Parameters
        ----------
        matrix : np.ndarray
            Input data matrix with shape (n_samples, _feature_dim*_combined_dim).

        Returns
        -------
        tensor : np.ndarray
            Tensor representation of the input data matrix with shape (n_samples, _feature_dim, _combined_dim).

        """
        if self._is_matrix_model:
            return matrix
        else:
            return super().convert_to_tensor(matrix)

    def inverse_transform(self, projection_data: np.ndarray, component_count: int):
        """
        Transform reduced-dimensional projection back to the original data space.

        This method performs the inverse transformation from the reduced-dimensional projection space to the original
        data space using the learned components of the DROPP model.

        Parameters
        ----------
        projection_data : np.ndarray
            Reduced-dimensional projection data with shape (n_samples, component_count).

        component_count : int
            Number of components to use for the inverse transformation.

        Returns
        -------
        original_data : np.ndarray
            Inverse-transformed data in the original data space with shape (n_samples, _feature_dim, _combined_dim).

        Raises
        ------
        NonInvertibleEigenvectorException
            If eigenvectors are non-orthogonal and non-squared, and `use_evs` flag is not set.

        """
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
        """
        Reconstruct original data tensor from a reduced-dimensional projection.

        This method reconstructs the original data tensor from a reduced-dimensional projection matrix using the
        learned components of the DROPP model.

        Parameters
        ----------
        projection_matrix : np.ndarray
            Reduced-dimensional projection matrix with shape (n_samples, component_count).

        component_count : int, optional
            Number of components to use for the reconstruction. If not provided, the maximum number of components
            available in the model will be used.

        Returns
        -------
        reconstructed_data : np.ndarray
            Reconstructed data tensor in the original data space with shape (n_samples, _feature_dim, _combined_dim).

        Raises
        ------
        InvalidComponentNumberException
            If the provided `component_count` exceeds the available number of components in the model.

        """
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
        Calculate the root mean squared error (RMSE) between original data and its reconstruction.

        This method calculates the root mean squared error (RMSE) between the original data tensor and its
        reconstruction obtained using the DROPP model. The RMSE provides a measure of the dissimilarity between the
        original and reconstructed data.

        Parameters
        ----------
        data_tensor : np.ndarray
            Original data tensor with shape (n_samples, _feature_dim, _combined_dim).

        y : None
            Ignored. This parameter exists only for compatibility with scikit-learns pipeline.

        Returns
        -------
        rmse : float
            Root mean squared error (RMSE) between original data and its reconstruction.

        See Also
        --------
        transform : Project data tensor into the reduced-dimensional space.
        reconstruct : Reconstruct original data tensor from a reduced-dimensional projection.

        Notes
        -----
        The RMSE is calculated based on the element-wise difference between the original data tensor and its
        reconstructed version. It provides an indication of the overall dissimilarity between the two data tensors.

        References
        ----------
        [1] StackExchange. "What does RMSE tell us?" URL: https://stats.stackexchange.com/q/229093
        """

        if y is not None:  # "use" variable, to not have a PyCharm warning
            data_projection = y
        else:
            data_projection = self.transform(data_tensor)

        reconstructed_tensor = self.reconstruct(data_projection, self.n_components)

        data_matrix = self.convert_to_matrix(data_tensor)
        reconstructed_matrix = self.convert_to_matrix(reconstructed_tensor)

        return mean_squared_error(data_matrix, reconstructed_matrix, squared=False)
