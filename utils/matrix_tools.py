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
from utils.math import is_matrix_symmetric, exponential_2d, epanechnikov_2d, gaussian_2d, is_matrix_orthogonal, my_sinc, \
    my_sinc_sum, my_cos
from utils.param_keys.analyses import PLOT_3D_MAP, WEIGHTED_DIAGONAL, FITTED_KERNEL_CURVES, KERNEL_COMPARE, \
    PLOT_KERNEL_MATRIX_3D
from utils.param_keys.kernel_functions import MY_GAUSSIAN, MY_EPANECHNIKOV, MY_EXPONENTIAL, MY_LINEAR, \
    MY_LINEAR_INVERSE_P1, MY_LINEAR_NORM, MY_LINEAR_INVERSE_NORM, MY_SINC, MY_SINC_SUM, MY_COS


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
        analyse_mode: str = None,
        flattened: bool = False,
        use_original_data: bool = False,
        **kwargs) -> np.ndarray:
    """
    Creates a symmetrical kernel matrix out of a symmetrical matrix.
    :param matrix: ndarray (symmetrical)
    :param kernel_stat_func: Numpy statistical function: np.median (default), np.mean, np.min, ... (See link below)
        https://www.tutorialspoint.com/numpy/numpy_statistical_functions.htm
    :param kernel_function: str
        (my_)gaussian, (my_)exponential, (my_)epanechnikov, (my_)linear, my_sinc(_sum)
    :param analyse_mode: str
        If the name of the trajectory is given than a plot of the gauss curve will be plotted with the given
    :param flattened: bool
        If True: runs the calculation in a way, where discontinuous input values are permitted.
    :param use_original_data: bool
        If True: uses only the original data without rescaling
    :return: ndarray (ndim=2)
        The gaussian kernel matrix
    """
    if not is_matrix_symmetric(matrix):
        raise ValueError(f'Input matrix to calculate the {kernel_function}-kernel has to be symmetric.')

    xdata = diagonal_indices(matrix)
    original_ydata = matrix_diagonals_calculation(matrix, np.mean)

    if use_original_data:
        rescaled_ydata = original_ydata
    else:
        if flattened:
            kernel_stat_func = np.min
            if kernel_function in [MY_COS]:
                interp_range = [-1, 1]
            else:
                interp_range = None
            rescaled_ydata = rescale_array(original_ydata, kernel_stat_func, interp_range)
        else:
            rescaled_ydata = rescale_center(original_ydata, kernel_stat_func)
    fit_y = _get_curve_fitted_y(matrix, kernel_function, xdata, rescaled_ydata)
    if flattened:  # bzw. reinterpolate
        fit_y = rescale_array(fit_y, lower_bound=kernel_stat_func(fit_y),
                              interp_range=[kernel_stat_func(original_ydata), np.max(original_ydata)])
    kernel_matrix = expand_diagonals_to_matrix(matrix, fit_y)

    if analyse_mode is not None:
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


def _get_rescaled_array(original_ydata, stat_func, flattened):
    """
    This function chooses the rescaling option depending on the flattened or unflattened input vairable.
    It the input-data-matrix was flattened, the array has to be rescaled on another way,
    since in this case, the original_ydata is not a continuous function.
    :param original_ydata:
    :param stat_func:
    :param flattened:
    :return:
    """
    if flattened:
        stat_func = np.min
        return rescale_array(original_ydata, stat_func)
    else:
        return rescale_center(original_ydata, stat_func)


# noinspection PyTupleAssignmentBalance
def _get_curve_fitted_y(matrix, kernel_name, xdata, rescaled_ydata):
    kernel_funcs = {MY_EXPONENTIAL: exponential_2d, MY_EPANECHNIKOV: epanechnikov_2d, MY_GAUSSIAN: gaussian_2d,
                    MY_SINC: my_sinc, MY_SINC + '_center': my_sinc, MY_SINC_SUM: my_sinc_sum, MY_COS: my_cos}

    if kernel_name in kernel_funcs.keys():
        if kernel_name in [MY_EPANECHNIKOV, MY_COS, MY_SINC + '_center']:
            non_zero_i = np.argmax(rescaled_ydata > 0)
            fit_y = rescaled_ydata.copy()
            if non_zero_i == 0 and kernel_name not in [MY_COS]:
                # If the rescaled_ydata does not reach bellow threshold, a different approach is needed
                fit_parameters, _ = curve_fit(kernel_funcs[kernel_name], xdata,
                                              rescaled_ydata)
                print(*fit_parameters)
                center_fit_y = kernel_funcs[kernel_name](xdata, *fit_parameters)
                fit_y = center_fit_y
            else:
                if kernel_name in [MY_COS]:
                    non_zero_i = (len(xdata) // 6)
                p0 = (len(xdata) // 2) - non_zero_i if kernel_name in [MY_COS] else 1
                fit_parameters, _ = curve_fit(kernel_funcs[kernel_name], xdata[non_zero_i:-non_zero_i],
                                              rescaled_ydata[non_zero_i:-non_zero_i], p0=p0, maxfev=5000)
                center_fit_y = kernel_funcs[kernel_name](xdata[non_zero_i:-non_zero_i], *fit_parameters)
                if kernel_name not in [MY_COS]:
                    center_fit_y = np.where(center_fit_y < 0, 0, center_fit_y)
                else:
                    fit_y = rescaled_ydata
                    fit_y = np.where(fit_y < 0, 0, fit_y)
                fit_y[non_zero_i:-non_zero_i] = center_fit_y

            return fit_y
        else:
            fit_parameters, _ = curve_fit(kernel_funcs[kernel_name], xdata, rescaled_ydata, maxfev=5000)
            return kernel_funcs[kernel_name](xdata, *fit_parameters)
    elif kernel_name.startswith('my_linear'):
        if kernel_name == MY_LINEAR_NORM:
            return np.concatenate((np.linspace(0, 1, len(matrix)), np.linspace(1, 0, len(matrix))[1:]))
        elif kernel_name == MY_LINEAR_INVERSE_NORM:
            return np.concatenate((np.linspace(1, 0, len(matrix)), np.linspace(0, 1, len(matrix))[1:]))
        elif kernel_name == MY_LINEAR:
            return np.concatenate((np.linspace(0, np.max(xdata), len(matrix)),
                                   np.linspace(np.max(xdata), 0, len(matrix))[1:]))
        else:  # kernel_name.startswith(MY_LINEAR_INVERSE):
            fit_y = np.abs(xdata)
            if kernel_name == MY_LINEAR_INVERSE_P1:
                fit_y -= 1
                return np.where(fit_y < 0, 1, fit_y)
    else:  # Try to use an implemented kernel from sklearn
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
