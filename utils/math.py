import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import distance
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity

from plotter import ArrayPlotter


def generate_independent_matrix(row, dimension):
    """
    https://stackoverflow.com/questions/61905733/generating-linearly-independent-columns-for-a-matrix
    :param row:
    :param dimension:
    :return:
    """
    matrix = np.random.rand(row, 1)
    rank = 1
    while rank < dimension:
        t = np.random.rand(row, 1)
        if np.linalg.matrix_rank(np.hstack([matrix, t])) > rank:
            matrix = np.hstack([matrix, t])
            rank += 1
    return matrix


def basis_transform(matrix, dimension):
    independent_matrix = generate_independent_matrix(dimension, dimension)
    transformation_matrix = independent_matrix @ np.identity(dimension)
    return np.einsum('tac,cc->tac', matrix, transformation_matrix)


def explained_variance(eigenvalues, component):
    return np.sum(eigenvalues[:component]) / np.sum(eigenvalues)


def gaussian_kern_matrix(size, sig=1.):
    """
    Creates gaussian kernel distribution matrix with the given `size` and a sigma of `sig`.
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    # lin_array = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)  # linear array (from: -size/2, to: size/2)
    lin_array = np.linspace(0, size, size)  # linear array (from: 0, to: size)
    gauss = np.exp(-0.5 * np.square(lin_array) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def diagonal_gauss_matrix_kernel(matrix_size, sig=1.):
    """
    Gaussian kernel diagonally in the matrix with a specific size and σ (sigma).
    https://peterroelants.github.io/posts/gaussian-process-kernels/
    """
    lin_range = (-1, 1)
    lin_array = np.expand_dims(np.linspace(*lin_range, matrix_size), 1)  # expand (x,)-array to (x,1)-array
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * distance.cdist(lin_array, lin_array, 'sqeuclidean')
    return np.exp(sq_norm / np.square(sig))  # gaussian


def matrix_diagonals_calculation(matrix: np.ndarray, func: callable = np.sum, func_kwargs: dict = None):
    """
    Down sample matrix to array by applying function `func` to diagonals
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


def is_matrix_symmetric(matrix, rtol=1e-05, atol=1e-08):
    """
    Checks a matrix for its symmetric behavior.
    https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
    :param matrix: array_like
    :param rtol: float
        The relative tolerance parameter (see Notes).
    :param atol: float
        The absolute tolerance parameter (see Notes).
    :return: bool
        Returns True if the matrix is symmetric within a tolerance.
    """
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def diagonal_indices(matrix):
    """
    Determines the (main and off) diagonal indices of a matrix
    :param matrix: array like
    :return:
    """
    lower_indices = -matrix.shape[0] + 1
    upper_indices = matrix.shape[1] - 1
    number_of_diagonals = sum(matrix.shape) - 1
    return np.linspace(lower_indices, upper_indices, number_of_diagonals, dtype=int)


def exponential_2d(x, sigma):
    """
    Function from:
    https://medium.com/geekculture/kernel-methods-in-support-vector-machines-bb9409342c49
    :param x: int or ndarray
    :param sigma:
    :return:
    """
    return np.exp(-np.abs(x) / (2 * np.power(sigma, 2.)))


def epanechnikov_2d(x, sigma):
    return 1 - (np.power(x, 2.) / np.power(sigma, 2.))


def gaussian_2d(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


def expand_diagonals_to_matrix(matrix, array):
    """
    Expand the values on the minor diagonals to its corresponding off-diagonal.
    Similar to: https://stackoverflow.com/questions/27875931/numpy-affect-diagonal-elements-of-matrix-prior-to-1-10
    :param matrix: ndarray
        N-dimensional input image. Specifies the size and the off-diagonal indexes of the return matrix
    :param array: ndarray
        Values on the minor diagonal which are going to expand by the function
    :return:
    """
    new_matrix = np.zeros_like(matrix)
    diag_indices = diagonal_indices(new_matrix)

    if len(diag_indices) != len(array):
        raise ValueError("`matrix` should have as many diagonals as the `array` has.")

    i, j = np.indices(new_matrix.shape)
    for array_index, diagonal_index in enumerate(diag_indices):
        new_matrix[i + diagonal_index == j] = array[array_index]

    return new_matrix


def calculate_pearson_correlations(matrix_list: list, func: callable = np.sum, func_kwargs=None):
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


def calculate_symmetrical_kernel_from_matrix(matrix, stat_func=np.median, trajectory_name=None, flattened=False):
    """
    Creates a symmetrical kernel matrix out of a symmetrical matrix.
    :param matrix: ndarray (symmetrical)
    :param stat_func: Numpy statistical function: np.median (default), np.mean, np.min, ... (See link below)
        https://www.tutorialspoint.com/numpy/numpy_statistical_functions.htm
    :param trajectory_name: str
        If the name of the trajectory is given than a plot of the gauss curve will be plotted with the given
    :param flattened: bool
        If True: runs the calculation in a way, where discontinuous input values are permitted.
    :return: The gaussian kernel matrix
    """
    if not is_matrix_symmetric(matrix):
        raise ValueError('Input matrix to calculate the gaussian kernel has to be symmetric.')

    kernel_name = 'my_exponential'  # gaussian, exponential, epanechnikov
    xdata = diagonal_indices(matrix)
    original_ydata = matrix_diagonals_calculation(matrix, np.mean)
    if flattened:
        stat_func = np.min
        interpolated_ydata = interpolate_array(original_ydata, stat_func)
    else:
        interpolated_ydata = interpolate_center(original_ydata, stat_func)

    kernel_funcs = {'my_exponential': exponential_2d, 'my_epanechnikov': epanechnikov_2d, 'my_gaussian': gaussian_2d}
    if kernel_name in kernel_funcs.keys():
        if kernel_name == 'my_epanechnikov':
            non_zero_i = np.argmax(interpolated_ydata > 0)
            fit_parameters, _ = curve_fit(epanechnikov_2d, xdata[non_zero_i:-non_zero_i], interpolated_ydata[non_zero_i:-non_zero_i])
            center_fit_y = epanechnikov_2d(xdata[non_zero_i:-non_zero_i], *fit_parameters)
            center_fit_y = np.where(center_fit_y < 0, 0, center_fit_y)
            fit_y = interpolated_ydata.copy()
            fit_y[non_zero_i:-non_zero_i] = center_fit_y
        else:
            fit_parameters, _ = curve_fit(kernel_funcs[kernel_name], xdata, interpolated_ydata)
            fit_y = kernel_funcs[kernel_name](xdata, *fit_parameters)
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


def interpolate_center(symmetrical_array, stat_func: callable = np.median):
    """
    Interpolates the data in the array center, and zeroes all the other values
    :param symmetrical_array: (nd)array symmetrical
    :param stat_func: Some statistical function of an array: np.median (default), np.mean, ...
    :return:
    """
    symmetrical_array = interpolate_array(symmetrical_array, stat_func)
    return extinct_side_values(symmetrical_array)


def interpolate_array(array, stat_func: callable = np.median, interp_range=None):
    """
    Interpolate an array from a range, of its statistical value (mean, median, min, ...) to the maximum,
    into a new range `interp_range` (default: 0-1)
    :param array:
    :param stat_func: Some statistical function of an array: np.median (default), np.mean, ...
    :param interp_range: the new range for the interpolation
    :return: the new interpolated array
    """
    if interp_range is None:
        interp_range = [0, 1]
    statistical_value = stat_func(array)
    return np.interp(array, [statistical_value, array.max()], interp_range)


def extinct_side_values(symmetrical_array, smaller_than=0):
    """
    Takes an array and searches the first value from the center smaller than the given value.
    All the border values from that criteria are zeroed
    :param symmetrical_array: symmetrical (nd)array
    :param smaller_than: int
        Sets the criteria to find the index between the center and the border.
    :return:
    """
    right_i = np.argmax(split_list_in_half(symmetrical_array)[1] <= smaller_than)
    center_i = len(symmetrical_array) // 2
    new_y = np.zeros_like(symmetrical_array)
    new_y[center_i - right_i:center_i + right_i] = symmetrical_array[center_i - right_i:center_i + right_i]
    return new_y


def split_list_in_half(a_list):
    """
    https://stackoverflow.com/questions/752308/split-list-into-smaller-lists-split-in-half
    :param a_list: array (list, ndarray)
    :return: tuple with the first and the second half of the list
    """
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


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
