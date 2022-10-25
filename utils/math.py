import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import distance

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


def gauss_kernel_symmetrical_matrix(matrix, func=np.median, trajectory_name=None):
    xdata = diagonal_indices(matrix)
    diag_func = np.mean
    ydata = matrix_diagonals_calculation(matrix, diag_func)  # TODO: func or median
    ydata = interpolate_center(ydata, func)
    fit_parameters, _ = curve_fit(gaussian_2d, xdata, ydata)
    fit_y = gaussian_2d(xdata, fit_parameters[0], fit_parameters[1])
    kernel_matrix = expand_diagonals_to_matrix(matrix, fit_y)
    if trajectory_name is not None:
        ArrayPlotter(interactive=False).plot_gauss2d(fit_y, xdata, ydata,
                                                     title_prefix=f'{trajectory_name} and '
                                                                  f'{"mean" if "mean" in str(diag_func) else "median"}'
                                                                  f' on diagonal of cov',
                                                     statistical_function=func)
    return kernel_matrix


def interpolate_center(ydata, func: callable = np.median):
    ydata = interpolate_array(ydata, func)
    return extinct_side_values(ydata)


def interpolate_array(ydata, func: callable = np.median):
    statistical_value = func(ydata)
    return np.interp(ydata, [statistical_value, ydata.max()], [0, 1])


def extinct_side_values(ydata, smaller_than=0):
    right_i = np.argmax(split_list_in_half(ydata)[1] <= smaller_than)
    center_i = len(ydata) // 2
    new_y = np.zeros_like(ydata)
    new_y[center_i - right_i:center_i + right_i] = ydata[center_i - right_i:center_i + right_i]
    return new_y


def split_list_in_half(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]
