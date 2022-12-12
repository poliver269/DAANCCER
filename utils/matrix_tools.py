import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity

from plotter import ArrayPlotter
from utils.array_tools import interpolate_array, interpolate_center
from utils.math import is_matrix_symmetric, exponential_2d, epanechnikov_2d, gaussian_2d
from utils.param_key import MY_GAUSSIAN, MY_EPANECHNIKOV, MY_EXPONENTIAL, MY_LINEAR, MY_NORM_LINEAR


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
    # TODO: Other idea: map diagonals into indices difference (i-j)


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
        raise ValueError(f'Input should be a matrix, but it\'s number of array dimension is: {matrix.ndim}')
    if not array.ndim == 1:
        raise ValueError(f'Input should be an array, but it\'s number of array dimension is: {array.ndim}')

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


def calculate_symmetrical_kernel_from_matrix(
        matrix: np.ndarray,
        stat_func: callable = np.median,
        kernel_name: str = 'gaussian',
        trajectory_name: str = None,
        flattened: bool = False) -> np.ndarray:
    """
    Creates a symmetrical kernel matrix out of a symmetrical matrix.
    :param matrix: ndarray (symmetrical)
    :param stat_func: Numpy statistical function: np.median (default), np.mean, np.min, ... (See link below)
        https://www.tutorialspoint.com/numpy/numpy_statistical_functions.htm
    :param kernel_name: str
        (my_)gaussian, (my_)exponential, (my_)epanechnikov, (my_)linear
    :param trajectory_name: str
        If the name of the trajectory is given than a plot of the gauss curve will be plotted with the given
    :param flattened: bool
        If True: runs the calculation in a way, where discontinuous input values are permitted.
    :return: ndarray (ndim=2)
        The gaussian kernel matrix
    """
    if not is_matrix_symmetric(matrix):
        raise ValueError('Input matrix to calculate the gaussian kernel has to be symmetric.')

    xdata = diagonal_indices(matrix)
    original_ydata = matrix_diagonals_calculation(matrix, np.mean)
    if flattened:
        stat_func = np.min
        interpolated_ydata = interpolate_array(original_ydata, stat_func)
    else:
        interpolated_ydata = interpolate_center(original_ydata, stat_func)
    kernel_funcs = {MY_EXPONENTIAL: exponential_2d, MY_EPANECHNIKOV: epanechnikov_2d, MY_GAUSSIAN: gaussian_2d}
    if kernel_name in kernel_funcs.keys():
        if kernel_name == MY_EPANECHNIKOV:
            non_zero_i = np.argmax(interpolated_ydata > 0)
            if non_zero_i == 0:  # TODO: Correcting for flattened data
                fit_parameters, _ = curve_fit(epanechnikov_2d, xdata,
                                              interpolated_ydata)
            else:
                fit_parameters, _ = curve_fit(epanechnikov_2d, xdata[non_zero_i:-non_zero_i],
                                              interpolated_ydata[non_zero_i:-non_zero_i])
            center_fit_y = epanechnikov_2d(xdata[non_zero_i:-non_zero_i], *fit_parameters)
            center_fit_y = np.where(center_fit_y < 0, 0, center_fit_y)
            fit_y = interpolated_ydata.copy()
            fit_y[non_zero_i:-non_zero_i] = center_fit_y
        else:
            fit_parameters, _ = curve_fit(kernel_funcs[kernel_name], xdata, interpolated_ydata)
            fit_y = kernel_funcs[kernel_name](xdata, *fit_parameters)
    elif kernel_name in [MY_LINEAR, MY_NORM_LINEAR]:
        if kernel_name == MY_NORM_LINEAR:
            fit_y = np.concatenate((np.linspace(0, 1, len(matrix)), np.linspace(1, 0, len(matrix))[1:]))
        else:
            fit_y = np.abs(xdata)
    else:  # Try to use an implemented kernel from sklearn
        xdata = xdata[:, np.newaxis]
        interpolated_ydata = interpolated_ydata[:, np.newaxis]
        interpolated_ydata = interpolated_ydata / interpolated_ydata.sum()
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel=kernel_name), {'bandwidth': bandwidths}, cv=LeaveOneOut())
        grid.fit(xdata)
        print(f'Best parameters for {kernel_name}: {grid.best_params_}')
        kde = KernelDensity(kernel=kernel_name, **grid.best_params_).fit(interpolated_ydata)
        # noinspection PyUnresolvedReferences
        fit_y = np.exp(kde.score_samples(xdata))
        fit_y = np.interp(fit_y, [0, fit_y.max()], [0, 1])
    kernel_matrix = expand_diagonals_to_matrix(matrix, fit_y)
    if trajectory_name is not None:
        if trajectory_name == 'weighted':
            ArrayPlotter(interactive=False).plot_gauss2d(xdata, original_ydata - fit_y, interpolated_ydata, fit_y,
                                                         kernel_name,
                                                         title_prefix=f'{trajectory_name}'
                                                                      f' on diagonal of cov',
                                                         statistical_function=stat_func)
        else:
            ArrayPlotter(interactive=False).plot_gauss2d(xdata, original_ydata, interpolated_ydata, fit_y, kernel_name,
                                                         title_prefix=f'Trajectory: {trajectory_name}, '
                                                                      f' on diagonal of cov',
                                                         statistical_function=stat_func)
    return kernel_matrix


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
