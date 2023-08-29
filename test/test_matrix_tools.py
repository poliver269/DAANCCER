import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy.testing as np_testing 

from utils.matrix_tools import *


class MyTestCase(unittest.TestCase):
    def test_co_mad(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(np.allclose(co_mad(matrix), np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])))

    def test_ensure_matrix_symmetry(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertTrue(np.allclose(ensure_matrix_symmetry(matrix), np.array([[1, 2.5], [2.5, 4]])))


if __name__ == '__main__':
    unittest.main()


class TestDiagonalIndices(unittest.TestCase):
    def test_square_matrix(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        expected_indices = np.array([-2, -1, 0, 1, 2])
        self.assertTrue(np.array_equal(diagonal_indices(matrix), expected_indices))

    def test_rectangular_matrix(self):
        matrix = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8]])
        expected_indices = np.array([-1, 0, 1, 2, 3])
        self.assertTrue(np.array_equal(diagonal_indices(matrix), expected_indices))

    def test_empty_matrix(self):
        matrix = np.empty((0, 0))
        expected_indices = np.array([])
        self.assertTrue(np.array_equal(diagonal_indices(matrix), expected_indices))


class TestMatrixDiagonalsCalculation(TestCase):
    def test_sum_diagonals(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        result = matrix_diagonals_calculation(matrix, func=np.sum)
        expected = np.array([7, 4 + 8, 1 + 5 + 9, 2 + 6, 3])
        np_testing.assert_array_equal(result, expected)

    def test_max_diagonals(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        result = matrix_diagonals_calculation(matrix, func=np.max)
        expected = np.array([7, 8, 9, 6, 3])
        np_testing.assert_array_equal(result, expected)

    def test_custom_function(self):
        def product_if_all_positive(diagonal):
            return np.prod(diagonal) if np.all(diagonal > 0) else diagonal[0]

        matrix = np.array([[1, 2, 3],
                           [4, 5, -6],
                           [-7, 8, 9]])
        result = matrix_diagonals_calculation(matrix, func=product_if_all_positive)
        expected = np.array([-7, 4 * 8, 1 * 5 * 9, 2, 3])
        np_testing.assert_array_equal(result, expected)


class TestExpandDiagonalsToMatrix(unittest.TestCase):
    def test_expand_diagonals(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        array = np.array([10, 20, 30, 40, 50])
        result = expand_diagonals_to_matrix(matrix, array)
        expected = np.array([[30, 40, 50],
                             [20, 30, 40],
                             [10, 20, 30]])
        np_testing.assert_array_equal(result, expected)

    def test_invalid_input_matrix(self):
        matrix = np.array([1, 2, 3])  # Not a 2D matrix
        array = np.array([10, 20, 30, 40, 50])
        with self.assertRaises(ValueError):
            expand_diagonals_to_matrix(matrix, array)

    def test_invalid_input_array_ndim(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        array = np.array([[10, 20, 30], [40, 50, 60]])
        with self.assertRaises(ValueError):
            expand_diagonals_to_matrix(matrix, array)

    def test_invalid_input_array(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        array = np.array([10, 20, 30])  # Length doesn't match number of diagonals
        with self.assertRaises(ValueError):
            expand_diagonals_to_matrix(matrix, array)


class TestDiagonalBlockExpand(unittest.TestCase):
    def test_diagonal_block_expand_identity(self):
        matrix = np.array([[1, 0],
                           [0, 2]])
        n_repeats = 2
        result = diagonal_block_expand(matrix, n_repeats)
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 2, 0],
                             [0, 0, 0, 2]])
        np_testing.assert_array_equal(result, expected)

    def test_diagonal_block_expand_non_identity(self):
        matrix = np.array([[1, 2],
                           [3, 4]])
        n_repeats = 3
        result = diagonal_block_expand(matrix, n_repeats)
        expected = np.array([[1, 0, 0, 2, 0, 0],
                             [0, 1, 0, 0, 2, 0],
                             [0, 0, 1, 0, 0, 2],
                             [3, 0, 0, 4, 0, 0],
                             [0, 3, 0, 0, 4, 0],
                             [0, 0, 3, 0, 0, 4]])
        np_testing.assert_array_equal(result, expected)

    def test_diagonal_block_expand_empty_matrix(self):
        matrix = np.array([])
        n_repeats = 2
        with self.assertRaises(ValueError):
            diagonal_block_expand(matrix, n_repeats)


class TestCalculateSymmetricalKernelMatrix(unittest.TestCase):
    def test_valid_input_gaussian(self):
        matrix = np.array([[1, .4, .2],
                           [.4, 1, .8],
                           [.2, .8, 1]])
        kernel_stat_func = np.mean
        kernel_function = 'gaussian'
        result = calculate_symmetrical_kernel_matrix(matrix, kernel_stat_func, kernel_function)
        rounded_expected = np.array([
            [1, 1.667e-01, 8e-04],
            [1.667e-01, 1, 1.667e-01],
            [8e-04, 1.667e-01, 1]
        ])
        self.assertIsInstance(result, np.ndarray)
        np_testing.assert_array_almost_equal(rounded_expected, np.round(result, decimals=4))

    def test_invalid_matrix_non_symmetric(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        with self.assertRaises(ValueError):
            calculate_symmetrical_kernel_matrix(matrix)

    def test_use_original_data(self):
        matrix = np.array([[1, .4, .2],
                           [.4, 1, .8],
                           [.2, .8, 1]])

        result = calculate_symmetrical_kernel_matrix(
            matrix, use_original_data=True
        )
        rounded_expected = np.array([[1, 0.6368, 0.1644],
                                     [0.6368, 1, 0.6368],
                                     [0.1644, 0.6368, 1]])
        np_testing.assert_array_almost_equal(rounded_expected, np.round(result, decimals=4))

    def test_flattened(self):
        matrix = np.array([[1, .4, .2],
                           [.4, 1, .8],
                           [.2, .8, 1]])

        result = calculate_symmetrical_kernel_matrix(
            matrix, flattened=True
        )
        rounded_expected = np.array([[1, 0.559, 0.2],
                                     [0.559, 1, 0.559],
                                     [0.2, 0.559, 1]])
        np_testing.assert_array_almost_equal(rounded_expected, np.round(result, decimals=4))

    def test_analyse_mode_kernel_compare(self):
        matrix = np.array([[1, .4, .2],
                           [.4, 1, .8],
                           [.2, .8, 1]])
        kernel_stat_func = np.median
        kernel_function = 'exponential'
        analyse_mode = KERNEL_COMPARE
        result = calculate_symmetrical_kernel_matrix(matrix, kernel_stat_func, kernel_function, analyse_mode)
        self.assertIsInstance(result, float)

    def test_analyse_mode_plot_3d_map(self):
        matrix = np.array([[1, .4, .2],
                           [.4, 1, .8],
                           [.2, .8, 1]])
        for analyse_mode in [PLOT_3D_MAP, PLOT_KERNEL_MATRIX_3D]:
            with patch.object(ArrayPlotter, 'matrix_plot') as mock_plotter:
                calculate_symmetrical_kernel_matrix(matrix, analyse_mode=analyse_mode)
                mock_plotter.assert_called_once()

    def test_analyse_mode_weighted_diagonal(self):
        matrix = np.array([[1, .4, .2],
                           [.4, 1, .8],
                           [.2, .8, 1]])
        for analyse_mode in [WEIGHTED_DIAGONAL, FITTED_KERNEL_CURVES]:
            with patch.object(ArrayPlotter, 'plot_gauss2d') as mock_plotter:
                result = calculate_symmetrical_kernel_matrix(matrix, analyse_mode=analyse_mode)
                self.assertIsInstance(result, np.ndarray)
                mock_plotter.assert_called_once()


class TestGetFittedYCurve(unittest.TestCase):
    def test_valid_kernel_name(self):
        kernel_name = 'gaussian'
        xdata = np.array([0, 1, 2, 3, 4])
        ydata = np.array([0.1, 0.2, 0.5, 0.3, 0.4])

        result = get_fitted_y_curve(kernel_name, xdata, ydata)
        self.assertTrue(isinstance(result, np.ndarray))

    def test_invalid_kernel_name(self):
        kernel_name = 'invalid_kernel'
        xdata = np.array([0, 1, 2, 3, 4])
        ydata = np.array([0.1, 0.2, 0.5, 0.3, 0.4])

        with self.assertRaises(InvalidKernelName):
            get_fitted_y_curve(kernel_name, xdata, ydata)

    def test_use_density_kernel(self):
        kernel_name = 'gaussian'
        xdata = np.array([0, 1, 2, 3, 4])
        ydata = np.array([0.1, 0.2, 0.5, 0.3, 0.4])

        result = get_fitted_y_curve(kernel_name, xdata, ydata, use_density_kernel=True)
        self.assertTrue(isinstance(result, np.ndarray))

    def test_use_density_kernel_invalid_kernel_name(self):
        kernel_name = 'invalid_kernel'
        xdata = np.array([0, 1, 2, 3, 4])
        ydata = np.array([0.1, 0.2, 0.5, 0.3, 0.4])

        with self.assertRaises(InvalidKernelName):
            get_fitted_y_curve(kernel_name, xdata, ydata, use_density_kernel=True)

    def test_cos_kernel_name(self):
        kernel_name = 'cos'
        xdata = np.array([-2, -1, 0, 1, 2])
        ydata = np.array([0.1, 0.2, 0.5, 0.2, 0.1])

        result = get_fitted_y_curve(kernel_name, xdata, ydata)
        rounded_expected = np.array([.1, .2, 1, .2, .1])
        np_testing.assert_array_almost_equal(rounded_expected, result)

    def test_epanechnikov_kernel_name(self):
        kernel_name = 'epanechnikov'
        xdata = np.array([-2, -1, 0, 1, 2])
        ydata = np.array([-0.1, 0.2, 0.5, 0.2, -0.1])

        result = get_fitted_y_curve(kernel_name, xdata, ydata)
        rounded_expected = np.array([-0.1, 0.2, 1, 0.2, -0.1])
        np_testing.assert_array_almost_equal(rounded_expected, result)

    def test_sinc_kernel_name(self):
        kernel_name = MY_SINC + '_center'
        xdata = np.array([-2, -1, 0, 1, 2])
        ydata = np.array([0.1, 0.2, 0.5, 0.2, 0.1])

        result = get_fitted_y_curve(kernel_name, xdata, ydata)
        rounded_expected = np.array([-0.0525, 0.0531, 1, 0.0531, -0.0525])
        np_testing.assert_array_almost_equal(rounded_expected, np.round(result, decimals=4))

    def test_linear_kernel_names(self):
        xdata = np.array([-2, -1, 0, 1, 2])
        ydata = np.array([0.1, 0.2, 0.5, 0.2, 0.1])
        for kernel_name, expected in [(MY_LINEAR, [0, 1, 2, 1, 0]),
                                      (MY_LINEAR_NORM, [0, 0.5, 1, 0.5, 0]),
                                      (MY_LINEAR_INVERSE, [2, 1, 0, 1, 2]),
                                      (MY_LINEAR_INVERSE_NORM, [1, 0.5, 0, 0.5, 1]),
                                      (MY_LINEAR_INVERSE_P1, [1, 0, 1, 0, 1])]:
            result = get_fitted_y_curve(kernel_name, xdata, ydata)
            rounded_expected = np.array(expected)
            np_testing.assert_array_almost_equal(rounded_expected, result)
            self.assertTrue(isinstance(result, np.ndarray))


class TestReconstructMatrix(unittest.TestCase):
    def test_reconstruct_orthogonal(self):
        projection = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        eigenvectors = np.array([[0.6, 0.8], [-0.8, 0.6]])

        result = reconstruct_matrix(projection, eigenvectors, dim=2, mean=0)

        expected_result = np.array([[-1, 2],
                                    [-1.4, 4.8],
                                    [-1.8, 7.6],
                                    [-2.2, 10.4]])
        np_testing.assert_array_almost_equal(expected_result, result)

    def test_reconstruct_non_orthogonal(self):
        projection = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        eigenvectors = np.array([[0.5, 0.8], [-0.8, 0.5]])

        result = reconstruct_matrix(projection, eigenvectors, dim=2, mean=1, std=2)

        expected_result = np.array([[-1.47191, 5.044944],
                                    [-2.820225, 10.88764],
                                    [-4.168539, 16.730337],
                                    [-5.516854, 22.573034]])
        np_testing.assert_array_almost_equal(expected_result, result)

    def test_reconstruct_pseudo_inverse(self):
        projection = np.array([[1], [3], [5], [7]])
        eigenvectors = np.array([[0.5, 0.8]])
        expected_result = np.array([[1.561798, 1.898876],
                                    [2.685393, 3.696629],
                                    [3.808989, 5.494382],
                                    [4.932584, 7.292135]])

        result = reconstruct_matrix(projection, eigenvectors, dim=2, mean=1)
        np_testing.assert_array_almost_equal(expected_result, result)
