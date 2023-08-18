import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity

from research_evaluations.plotter import ArrayPlotter
from utils import function_name
from utils.array_tools import rescale_array, rescale_center
from utils.errors import InvalidKernelName
from utils.math import is_matrix_symmetric, exponential_2d, epanechnikov_2d, gaussian_2d, is_matrix_orthogonal, \
    my_sinc, my_sinc_sum, my_cos
from utils.param_keys.analyses import PLOT_3D_MAP, WEIGHTED_DIAGONAL, FITTED_KERNEL_CURVES, KERNEL_COMPARE, \
    PLOT_KERNEL_MATRIX_3D
from utils.param_keys.kernel_functions import MY_GAUSSIAN, MY_EPANECHNIKOV, MY_EXPONENTIAL, MY_LINEAR, \
    MY_LINEAR_INVERSE_P1, MY_LINEAR_NORM, MY_LINEAR_INVERSE_NORM, MY_SINC, MY_SINC_SUM, MY_COS, USE_DENSITY_KERNEL, MY

kernel_funcs = {
    MY_EXPONENTIAL: exponential_2d,
    MY_EPANECHNIKOV: epanechnikov_2d,
    MY_GAUSSIAN: gaussian_2d,
    MY_SINC: my_sinc,
    MY_SINC + '_center': my_sinc,
    MY_SINC_SUM: my_sinc_sum,
    MY_COS: my_cos
}


def diagonal_indices(matrix: np.ndarray):
    """
    Determines the (main and off) diagonal indices of a matrix
    :param matrix: array like
    :return:
    """
    lower_indices = -matrix.shape[0] + 1
    upper_indices = matrix.shape[1] - 1
    number_of_diagonals = sum(matrix.shape) - 1
    return np.linspace(lower_indices, upper_indices, number_of_diagonals, dtype=int)


def matrix_diagonals_calculation(matrix: np.ndarray, func: callable = np.sum, func_kwargs: dict = None):
    """
    Down sample matrix to an array by applying function `func` to diagonals
    :param matrix: ndarray
        N-dimensional input image
    :param func: callable
        Function object which is used to calculate the return value for each
        diagonal. Primary functions are ``numpy.sum``, ``numpy.min``, ``numpy.max``,
        ``numpy.mean`` and ``numpy.median``.  See also `func_kwargs`
    :param func_kwargs: dict
        Keyword arguments passed to `func`. Notably useful for passing dtype
        argument to ``np.mean``. Takes dictionary of inputs, e.g.:
        ``func_kwargs={'dtype': np.float16})``
    :return: ndarray
    """
    if func_kwargs is None:
        func_kwargs = {}

    calculated_diagonals = []
    for diagonal_index in diagonal_indices(matrix):
        m = np.diag(matrix, k=diagonal_index)
        calculated_diagonals.append(func(m, **func_kwargs))

    return np.asarray(calculated_diagonals)


def expand_diagonals_to_matrix(matrix: np.ndarray, array: np.ndarray):
    """
    Expand the values on the minor diagonals to its corresponding off-diagonal.
    Similar to: https://stackoverflow.com/questions/27875931/numpy-affect-diagonal-elements-of-matrix-prior-to-1-10
    :param matrix: ndarray
        2-array-dimensional input image. Specifies the size and the off-diagonal indexes of the return matrix
    :param array: ndarray
        Values on the minor diagonal which are going to expand by the function
    :return:
    """
    if not matrix.ndim == 2:
        raise ValueError(f'Input should be a matrix, but it\'s n-dimension is: {matrix.ndim}')
    if not array.ndim == 1:
        raise ValueError(f'Input should be an array, but it\'s n-dimension is: {array.ndim}')

    new_matrix = np.zeros_like(matrix)
    diag_indices = diagonal_indices(new_matrix)

    if len(diag_indices) != len(array):
        raise ValueError(f"Input matrix should have as many diagonals ({len(diag_indices)}) "
                         f"as the length of the input array ({len(array)}).")

    i, j = np.indices(new_matrix.shape)
    for array_index, diagonal_index in enumerate(diag_indices):
        new_matrix[i + diagonal_index == j] = array[array_index]

    return new_matrix


def calculate_pearson_correlations(matrix_list: list[np.ndarray], func: callable = np.sum, func_kwargs=None):
    """
    Calculates the Pearson product-moment correlation coefficients of all the matrix in the list and applies a function
    :param matrix_list: list of ndarray
        List of N-dimensional input image
    :param func: callable
        Function object which is used to calculate the return value for each
        element in the matrix. This function must implement an ``axis`` parameter.
        Primary functions are ``numpy.sum``, ``numpy.min``, ``numpy.max``,
        ``numpy.mean`` and ``numpy.median``.  See also `func_kwargs`
    :param func_kwargs: dict
        Keyword arguments passed to `func`. Notably useful for passing dtype
        argument to ``np.mean``. Takes dictionary of inputs, e.g.:
        ``func_kwargs={'dtype': np.float16})``
    """
    if func_kwargs is None:
        func_kwargs = {}

    pc_list = list(map(lambda m: np.corrcoef(m.T), matrix_list))

    return func(np.array(pc_list), axis=0, **func_kwargs)


def diagonal_block_expand(matrix, n_repeats):
    """
    Expands a Matrix with the values as diagonals on a block with the size n_repeats.
    https://stackoverflow.com/questions/74054138/fastest-way-to-resize-a-numpy-matrix-in-diagonal-blocks
    :param matrix: matrix ndarray which should be expanded
    :param n_repeats:
    :return:
    """
    return np.einsum('ij,kl->ikjl', matrix, np.eye(n_repeats)).reshape(len(matrix) * n_repeats, -1)


def calculate_symmetrical_kernel_matrix(
        matrix: np.ndarray,
        kernel_stat_func: callable = np.median,
        kernel_function: str = 'gaussian',
        analyse_mode: str = '',
        flattened: bool = False,
        use_original_data: bool = False,
        **kwargs) -> np.ndarray:
    """
    Create a symmetrical kernel matrix out of a symmetrical matrix.

    Parameters
    ----------
    matrix : ndarray
        Input symmetrical matrix.
    kernel_stat_func : callable, optional
        Numpy statistical function, e.g., np.median (default), np.mean, np.min, etc.
    kernel_function : str, optional
        Kernel function name: 'gaussian', 'exponential', 'epanechnikov', 'linear', etc.
    analyse_mode : str, optional
        Analysis mode: 'KERNEL_COMPARE', 'PLOT_3D_MAP', 'PLOT_KERNEL_MATRIX_3D', 'WEIGHTED_DIAGONAL',
        'FITTED_KERNEL_CURVES', or empty string for no analysis.
    flattened : bool, optional
        If True, permits discontinuous input values.
    use_original_data : bool, optional
        If True, uses only the original data without rescaling.

    Returns
    -------
    ndarray
        The kernel matrix.
    """
    if not is_matrix_symmetric(matrix):
        raise ValueError(f'Input matrix with shape ({matrix.shape}) has to be symmetric'
                         f'to calculate the {kernel_function}-kernel.')

    xdata = diagonal_indices(matrix)
    original_ydata = matrix_diagonals_calculation(matrix, np.mean)

    if use_original_data:
        rescaled_ydata = original_ydata
    else:
        if flattened:
            kernel_stat_func = np.min
            interp_range = [-1, 1] if kernel_function in [MY_COS] else None
            rescaled_ydata = rescale_array(original_ydata, kernel_stat_func, interp_range)
        else:
            rescaled_ydata = rescale_center(original_ydata, kernel_stat_func)

    fit_y = _get_fitted_y_curve(matrix, kernel_function, xdata, rescaled_ydata, **kwargs)

    if flattened:  # re-interpolate
        fit_y = rescale_array(
            fit_y,
            lower_bound=kernel_stat_func(fit_y),
            interp_range=[kernel_stat_func(original_ydata), np.max(original_ydata)]
        )

    kernel_matrix = expand_diagonals_to_matrix(matrix, fit_y)

    if analyse_mode != '':
        if analyse_mode == KERNEL_COMPARE:
            return mean_squared_error(rescaled_ydata, fit_y, squared=False)
        elif analyse_mode == PLOT_3D_MAP:
            ArrayPlotter(
                interactive=False,
                title_prefix='Combined Covariance Matrix',
                x_label='Hours',
                y_label='Hours',
                for_paper=True
            ).matrix_plot(matrix, as_surface='2d')
        elif analyse_mode == PLOT_KERNEL_MATRIX_3D:
            ArrayPlotter(
                interactive=True,
                title_prefix='Kernel Matrix',
                x_label='\nCarbon-Alpha Atom\nIndex',
                y_label='\nCarbon-Alpha Atom\nIndex',
                for_paper=True
            ).matrix_plot(kernel_matrix, as_surface=PLOT_3D_MAP)
        elif analyse_mode == WEIGHTED_DIAGONAL:
            ArrayPlotter(
                interactive=False,
                title_prefix=f'{WEIGHTED_DIAGONAL} of {function_name(kernel_stat_func)}',
                for_paper=True
            ).plot_gauss2d(xdata, original_ydata - fit_y, rescaled_ydata, fit_y, kernel_function, kernel_stat_func)
        elif analyse_mode == FITTED_KERNEL_CURVES:
            ArrayPlotter(
                interactive=False,
                title_prefix=f'Kernel Curves: {kernel_function}, use_original_data={use_original_data}',
                x_label='Off-Diagonal Index',
                y_label='Correlation Value',
                for_paper=True
            ).plot_gauss2d(xdata, original_ydata, rescaled_ydata, fit_y, kernel_function, kernel_stat_func)
    return kernel_matrix


def _get_fitted_y_curve(matrix: np.ndarray, kernel_name: str, xdata: np.ndarray, ydata: np.ndarray, **kwargs):
    """
    Generate a fitted y curve based on the specified kernel and input data.

    This function generates a fitted y curve based on the specified kernel and input data. The type of fitted y curve
    generation depends on the selected kernel and additional options provided through keyword arguments.

    Parameters
    ----------
    matrix : ndarray
        The input matrix for which the fitted y curve is calculated.
    kernel_name : str
        Name of the kernel to be used for curve fitting.
    xdata : ndarray
        x data for fitting.
    ydata : ndarray
        y data for fitting.
    **kwargs
        Additional keyword arguments to control the fitting process.

    Returns
    -------
    ndarray
        The fitted y curve generated using the specified kernel and input data.

    Raises
    ------
    InvalidKernelName
        If the specified kernel name is not valid.

    Notes
    -----
    - If the option `USE_DENSITY_KERNEL` is provided in kwargs and set to True, kernel density estimation (KDE) is
      used to generate the fitted y curve using the `_get_density_fitted_y` function.
    - Otherwise, the function determines the type of kernel and its fitting process based on the kernel_name parameter:
      - For specific kernel types (MY_EPANECHNIKOV, MY_COS, MY_SINC_CENTER), `_get_y_fitted_on_positive_values`
        is used to generate the fitted y curve.
      - For other kernel types, `_fit_y_curve` is used to generate the fitted y curve with specified fitting options.
      - For kernel names starting with MY_LINEAR, `_get_linear_fitted_y` is used to generate the fitted y curve.
    - The returned fitted y curve is based on the conditions and fitting process described above.

    """
    if USE_DENSITY_KERNEL in kwargs.keys() and kwargs[USE_DENSITY_KERNEL]:
        return _get_density_fitted_y(kernel_name, xdata, ydata)
    else:
        if not kernel_name.startswith(MY):
            kernel_name = MY + kernel_name

        if kernel_name in kernel_funcs.keys():
            if kernel_name in [MY_EPANECHNIKOV, MY_COS, MY_SINC + '_center']:
                return _get_y_fitted_on_positive_values(kernel_name, xdata, ydata)
            else:
                return _fit_y_curve(kernel_name, xdata, ydata, maxfev=5000)
        elif kernel_name.startswith(MY_LINEAR):
            return _get_linear_fitted_y(kernel_name, xdata, len(matrix))
        else:
            raise InvalidKernelName(f'Kernel name `{kernel_name.split(MY)[1]}` '
                                    f'does not exist. Please choose a valid kernel.')


def _get_y_fitted_on_positive_values(kernel_name, xdata, ydata):
    """
    Generate a fitted y curve for the given kernel based on positive values.

    This function calculates a fitted y curve using the specified kernel when the input data has positive values.
    The kernel curve fitting process varies depending on whether the kernel name is MY_COS and whether the
    rescaled y data has any positive values.

    Parameters
    ----------
    kernel_name : str
        Name of the kernel.
    xdata : ndarray
        x data for fitting.
    ydata : ndarray
        (Rescaled) y data.

    Returns
    -------
    ndarray
        The fitted y curve.

    Notes
    -----
    - If the first index above 0 in `ydata` is 0 (equals to not found) and the kernel is not MY_COS,
      or if there is only one positive value in `rescaled_ydata`, the fitted y curve is generated
      using the specified kernel function.
    - For MY_COS kernel, a magic number (6) is used to determine the region of non-zero values in `rescaled_ydata`.
      `_fit_y_on_positive_values_in_center` is called to generate the fitted y curve for the non-zero values region.
    - The returned fitted y curve is based on the conditions described above.

    """
    non_zero_i = np.argmax(ydata > 0)  # first index which is above 0
    fit_y = ydata.copy()
    if (non_zero_i == 0 and kernel_name not in [MY_COS]) or (np.sum(ydata > 0) == 1):
        return _fit_y_curve(kernel_name, xdata, ydata)
    else:
        if kernel_name in [MY_COS]:
            magic_number = 6
            non_zero_i = (len(xdata) // magic_number)
        fit_y[non_zero_i:-non_zero_i] = _fit_y_on_positive_values_in_the_middle(
            kernel_name, xdata, ydata, non_zero_i)
        return fit_y


def _fit_y_curve(kernel_name: str, xdata: np.ndarray, ydata: np.ndarray, **fit_kwargs):
    """
    Fit a curve for the specified kernel and generate the fitted y curve.

    This function fits a curve using the specified kernel function to the given x data and rescaled y data. The
    fitted y curve is generated based on the calculated fit parameters.

    Parameters
    ----------
    kernel_name : str
        Name of the kernel.
    xdata : ndarray
        x data for fitting.
    ydata : ndarray
        (Rescaled) y data.
    fit_kwargs
        Additional keyword arguments to be passed to the curve fitting function.

    Returns
    -------
    ndarray
        The fitted y curve generated using the calculated fit parameters.

    """
    fit_parameters, _ = curve_fit(kernel_funcs[kernel_name], xdata, ydata, **fit_kwargs)
    return kernel_funcs[kernel_name](xdata, *fit_parameters)


def _fit_y_on_positive_values_in_the_middle(kernel_name, xdata, ydata, non_zero_i):
    """
    Fit a curve for the specified kernel in the region of positive values within the middle range.

    This function fits a curve using the specified kernel function to the given x data and y data, considering
    only the region of positive values within the middle range. The fitted y curve is generated based on the
    calculated fit parameters.

    Parameters
    ----------
    kernel_name : str
        Name of the kernel.
    xdata : ndarray
        x data for fitting.
    ydata : ndarray
        y data for fitting.
    non_zero_i : int
        Index of the first non-zero value.

    Returns
    -------
    ndarray
        The fitted y curve generated using the calculated fit parameters.

    Notes
    -----
    - For MY_COS kernel, the magic number (6) is used to determine the region of non-zero values in `ydata`.
      The fitted y curve is generated for the non-zero values within the middle range.
    - The returned fitted y curve is based on the conditions described above.

    """
    p0 = (len(xdata) // 2) - non_zero_i if kernel_name in [MY_COS] else 1
    middle_fit_y = _fit_y_curve(kernel_name, xdata[non_zero_i:-non_zero_i],
                                ydata[non_zero_i:-non_zero_i],
                                p0=p0, maxfev=5000)
    if kernel_name not in [MY_COS]:
        middle_fit_y = np.where(middle_fit_y < 0, 0, middle_fit_y)
    fit_y = ydata
    fit_y[non_zero_i:-non_zero_i] = middle_fit_y
    return fit_y


def _get_linear_fitted_y(kernel_name, xdata, matrix_length):
    """
    Generate a fitted y curve for linear kernels or their inverses.

    This function generates a fitted y curve based on the specified linear kernel or its inverse for the given x data
    and matrix length. The fitted y curve is constructed by linearly interpolating values according to the selected
    kernel type.

    Parameters
    ----------
    kernel_name : str
        Name of the linear kernel or its inverse.
    xdata : ndarray
        x data for fitting.
    matrix_length : int
        Length of the matrix.

    Returns
    -------
    ndarray
        The fitted y curve generated using linear interpolation.

    Notes
    -----
    - For MY_LINEAR_NORM kernel, the fitted y curve is constructed by concatenating a linearly spaced sequence from
      0 to 1 with a reversed linearly spaced sequence from 1 to 0 (excluding the first element).
    - For MY_LINEAR_INVERSE_NORM kernel, the fitted y curve is constructed by concatenating a linearly spaced
      sequence from 1 to 0 with a reversed linearly spaced sequence from 0 to 1 (excluding the first element).
    - For MY_LINEAR kernel, the fitted y curve is constructed by concatenating a linearly spaced sequence from 0
      to the maximum value of xdata with a reversed linearly spaced sequence from the maximum value of xdata to 0
      (excluding the first element).
    - For other linear kernels or their inverses,
      `_get_inverse_linear_fitted_y` is called to generate the fitted y curve.

    """
    if kernel_name == MY_LINEAR_NORM:
        return np.concatenate((np.linspace(0, 1, matrix_length), np.linspace(1, 0, matrix_length)[1:]))
    elif kernel_name == MY_LINEAR_INVERSE_NORM:
        return np.concatenate((np.linspace(1, 0, matrix_length), np.linspace(0, 1, matrix_length)[1:]))
    elif kernel_name == MY_LINEAR:
        return np.concatenate((np.linspace(0, np.max(xdata), matrix_length),
                               np.linspace(np.max(xdata), 0, matrix_length)[1:]))
    else:
        return _get_inverse_linear_fitted_y(kernel_name, xdata)


def _get_inverse_linear_fitted_y(kernel_name, xdata):
    """
    Generate a fitted y curve for the inverse of a linear kernel.

    This function generates a fitted y curve based on the inverse of a linear kernel for the given x data. The fitted y
    curve is constructed by calculating the absolute values of x data and applying specific adjustments based on the
    selected kernel type.

    Parameters
    ----------
    kernel_name : str
        Name of the inverse linear kernel.
    xdata : ndarray
        x data for fitting.

    Returns
    -------
    ndarray
        The fitted y curve generated based on the inverse linear kernel.

    Notes
    -----
    - The fitted y curve is constructed by calculating the absolute values of x data.
    - For MY_LINEAR_INVERSE_P1 kernel, the fitted y curve is adjusted by subtracting 1 from the calculated absolute
      values and then applying a thresholding operation to ensure all values are non-negative.

    """
    fit_y = np.abs(xdata)
    if kernel_name == MY_LINEAR_INVERSE_P1:
        fit_y -= 1
        return np.where(fit_y < 0, 1, fit_y)


def _get_density_fitted_y(kernel_name, xdata, rescaled_ydata):
    """
    Generate a fitted y curve using kernel density estimation.

    This function generates a fitted y curve using kernel density estimation (KDE) for the given x data and rescaled
    y data. The KDE process involves finding the optimal bandwidth parameter and fitting the kernel to the data
    distribution. The fitted y curve is then generated based on the estimated kernel density.

    Parameters
    ----------
    kernel_name : str
        Name of the kernel for KDE.
    xdata : ndarray
        x data for fitting.
    rescaled_ydata : ndarray
        Rescaled y data.

    Returns
    -------
    ndarray
        The fitted y curve generated using kernel density estimation.

    Notes
    -----
    - The kernel density estimation process involves finding the optimal bandwidth parameter using a grid search
      approach based on a range of bandwidth values.
    - The best bandwidth parameter is used to fit the kernel to the rescaled y data distribution.
    - The fitted y curve is generated based on the estimated kernel density using the calculated bandwidth parameter.
    - The resulting fitted y curve is interpolated to ensure it ranges between 0 and 1.

    """
    xdata = xdata[:, np.newaxis]
    rescaled_ydata = rescaled_ydata[:, np.newaxis]
    rescaled_ydata = rescaled_ydata / rescaled_ydata.sum()
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel=kernel_name), {'bandwidth': bandwidths}, cv=LeaveOneOut())
    grid.fit(xdata)
    print(f'Best parameters for {kernel_name}: {grid.best_params_}')
    kde = KernelDensity(kernel=kernel_name, **grid.best_params_).fit(rescaled_ydata)
    # noinspection PyUnresolvedReferences
    fit_y = np.exp(kde.score_samples(xdata))
    return np.interp(fit_y, [0, fit_y.max()], [0, 1])


def co_mad(matrix):
    """
    Calculates the coMAD (Co Median Absolute Deviation) matrix of an input matrix.
    https://ceur-ws.org/Vol-2454/paper_74.pdf
    https://github.com/huenemoerder/CODEC/blob/master/CODEC.ipynb
    :param matrix: ndarray
    :return: Co-Median Absolute Deviation
    """
    matrix_sub = matrix - np.median(matrix, axis=1)[:, np.newaxis]
    return np.median(matrix_sub[np.newaxis, :, :] * matrix_sub[:, np.newaxis, :], axis=2)


def ensure_matrix_symmetry(matrix):
    return 0.5 * (matrix + matrix.T)


def reconstruct_matrix(projection, eigenvectors, dim, mean, std=1):
    """
    Reconstructs a matrix from a given projection and eigenvector
    :param projection:
    :param eigenvectors:
    :param dim:
    :param mean: of the original data
    :param std: of the original data
    :return:
    """
    # if is_matrix_orthogonal(eigenvectors):
    #   reconstructed_matrix = np.dot(projection[:, :dim], eigenvectors[:, :dim].T)
    if is_matrix_orthogonal(eigenvectors.T):
        reconstructed_matrix = np.dot(projection[:, :dim], eigenvectors[:dim])
    else:
        try:
            reconstructed_matrix = np.dot(projection[:, :dim], np.linalg.inv(eigenvectors.T)[:dim])
        except LinAlgError:
            reconstructed_matrix = np.dot(projection[:, :dim], np.linalg.pinv(eigenvectors.T)[:dim])
            # raise LinAlgError(f'Eigenvector shape: {eigenvectors.shape}\n'
            #                   f'Projection shape: {projection.shape}\n'
            #                   f'N-components: {dim}')
    reconstructed_matrix *= std
    reconstructed_matrix += mean
    return reconstructed_matrix


def expand_and_roll(matrix, expand_dim=3):
    new_matrix = matrix.reshape((-1, expand_dim, matrix.shape[1]))
    rotation_matrix = np.asarray(
        [Rotation.from_euler('x', 0, degrees=True).as_matrix(),  # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
         Rotation.from_euler('xz', [90, 90], degrees=True).as_matrix(),  # [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
         Rotation.from_euler('xy', [-90, -90], degrees=True).as_matrix()])  # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    return np.tensordot(new_matrix, rotation_matrix, [1, 0]).swapaxes(1, 2).reshape((-1, expand_dim * matrix.shape[1]))
