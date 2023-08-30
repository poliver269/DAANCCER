import unittest
from unittest.mock import patch

import numpy.testing as np_testing

from utils.algorithms.dropp import *
from utils.errors import ModelNotFittedError
from utils.matrix_tools import co_mad


class TestDROPPInitialization(unittest.TestCase):
    def test_default_initialization(self):
        dropp = DROPP()
        self.assertIsInstance(dropp, DROPP)
        self.assertEqual(dropp.cov_stat_func, np.mean)
        self.assertEqual(dropp.algorithm_name, 'pca')
        self.assertEqual(dropp.ndim, 3)
        self.assertEqual(dropp.kernel_kwargs, {
            KERNEL_MAP: KERNEL_ONLY,
            KERNEL_FUNCTION: GAUSSIAN,
            KERNEL_STAT_FUNC: statistical_zero,
            USE_ORIGINAL_DATA: False,
            CORR_KERNEL: False,
            ONES_ON_KERNEL_DIAG: False
        })
        self.assertEqual(dropp.cov_function, np.cov)
        self.assertEqual(dropp.lag_time, 0)
        self.assertEqual(dropp.nth_eigenvector, 1)
        self.assertFalse(dropp.extra_dr_layer)
        self.assertTrue(dropp.abs_eigenvalue_sorting)
        self.assertEqual(dropp.analyse_plot_type, '')
        self.assertTrue(dropp.use_std)
        self.assertTrue(dropp.center_over_time)

    def test_custom_initialization(self):
        custom_kernel_kwargs = {
            KERNEL_MAP: KERNEL_DIFFERENCE,
            KERNEL_FUNCTION: 'my_custom_kernel',
            KERNEL_STAT_FUNC: np.median,
            USE_ORIGINAL_DATA: True,
            CORR_KERNEL: True,
            ONES_ON_KERNEL_DIAG: True
        }
        dropp = DROPP(
            cov_stat_func=co_mad,
            algorithm_name='tica',
            ndim=2,
            kernel_kwargs=custom_kernel_kwargs,
            cov_function=np.corrcoef,
            lag_time=5,
            nth_eigenvector=2,
            extra_dr_layer=True,
            abs_eigenvalue_sorting=False,
            analyse_plot_type='heatmap',
            use_std=False,
            center_over_time=False
        )

        self.assertEqual(dropp.cov_stat_func, co_mad)
        self.assertEqual(dropp.algorithm_name, 'tica')
        self.assertEqual(dropp.ndim, 2)
        self.assertEqual(dropp.kernel_kwargs, custom_kernel_kwargs)
        self.assertEqual(dropp.cov_function, np.corrcoef)
        self.assertEqual(dropp.lag_time, 5)
        self.assertEqual(dropp.nth_eigenvector, 2)
        self.assertTrue(dropp.extra_dr_layer)
        self.assertFalse(dropp.abs_eigenvalue_sorting)
        self.assertEqual(dropp.analyse_plot_type, 'heatmap')
        self.assertFalse(dropp.use_std)
        self.assertFalse(dropp.center_over_time)


class TestDROPPInitParamsCheck(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)

    def test_nth_eigenvector_adjustment(self):
        dropp = DROPP(nth_eigenvector=0)
        self.assertEqual(dropp.nth_eigenvector, 1)

    def test_extra_dr_layer_ignoring_nth_eigenvector(self):
        with self.assertWarns(UserWarning):
            dropp = DROPP(extra_dr_layer=True, nth_eigenvector=2)
        self.assertEqual(dropp.extra_dr_layer, True)
        self.assertEqual(dropp.nth_eigenvector, 2)

    def test_time_lagged_warning(self):
        with self.assertWarns(UserWarning):
            DROPP(algorithm_name='tica', lag_time=0)

    def test_kernel_stat_func_evaluation_warning(self):
        kernel_kwargs = {
            KERNEL_MAP: KERNEL_ONLY,
            KERNEL_FUNCTION: GAUSSIAN,
            KERNEL_STAT_FUNC: 'np.mean',  # A string that will be evaluated
            USE_ORIGINAL_DATA: False,
            CORR_KERNEL: False,
            ONES_ON_KERNEL_DIAG: False
        }
        with self.assertWarns(UserWarning):
            dropp = DROPP(kernel_kwargs=kernel_kwargs)
        self.assertTrue(callable(dropp.kernel_kwargs['kernel_stat_func']))


class TestDROPPProperties(unittest.TestCase):
    def setUp(self):
        self.dropp = DROPP()

    def test_is_matrix_model(self):
        self.assertFalse(self.dropp._is_matrix_model)

    def test_is_time_lagged_algorithm(self):
        self.assertFalse(self.dropp._is_time_lagged_model)

    def test_use_kernel_as_correlations_matrix(self):
        self.assertFalse(self.dropp._use_kernel_as_correlation_matrix())

    def test_use_correlations_matrix(self):
        self.assertFalse(self.dropp._use_correlation_matrix())

    def test_use_evs(self):
        self.assertFalse(self.dropp._use_evs)

    def test_combine_dim(self):
        with self.assertRaises(ModelNotFittedError):
            _ = self.dropp._combine_dim

    def test_feature_dim(self):
        with self.assertRaises(ModelNotFittedError):
            _ = self.dropp._feature_dim


class TestDROPPFitTransform(unittest.TestCase):
    def setUp(self):
        self.dropp = DROPP()

    def test_fit_transform(self):
        data_tensor = np.random.rand(100, 10, 5)
        transformed_data = self.dropp.fit_transform(data_tensor)
        self.assertEqual(transformed_data.shape, (100, self.dropp.n_components))


class TestDROPPFit(unittest.TestCase):
    def setUp(self):
        self.dropp = DROPP()
        self.data_tensor = np.random.rand(100, 10, 5)
        warnings.simplefilter("ignore", category=UserWarning)

    def test_fit_default(self):
        self.dropp.fit(self.data_tensor)
        self.assertEqual(self.dropp.n_components, 2)
        self.assertTrue(hasattr(self.dropp, 'components_'))

    def test_fit_with_params(self):
        self.dropp.fit(self.data_tensor, n_components=3)
        self.assertEqual(self.dropp.n_components, 3)

    def test_eigenvector_matrix_analyse(self):
        self.dropp = DROPP(analyse_plot_type='eigenvector_matrix_analyse')
        with patch.object(ArrayPlotter, 'matrix_plot') as mock_plotter:
            self.dropp.fit(self.data_tensor)
            mock_plotter.assert_called_once()

    def test_value_error(self):
        self.dropp = DROPP(ndim=2)
        with self.assertRaises(ValueError):
            self.dropp.fit(self.data_tensor)

    def test_standardize_data_matrix(self):
        self.dropp = DROPP(ndim=2, use_std=False)
        matrix_data = np.random.rand(100, 10)

        self.dropp.fit(matrix_data)

        self.assertTrue(np.allclose(np.mean(self.dropp._standardized_data, axis=0), np.zeros(10)))
        self.assertFalse(np.allclose(np.std(self.dropp._standardized_data, axis=0), np.ones(10)))
        self.assertEqual((100, 10), self.dropp._standardized_data.shape)

    def test_standardize_data_tensor(self):
        self.dropp.fit(self.data_tensor)

        self.assertTrue(np.allclose(np.mean(self.dropp._standardized_data, axis=0), np.zeros(5)))
        self.assertTrue(np.allclose(np.std(self.dropp._standardized_data, axis=0), np.ones(5)))
        self.assertEqual((100, 10, 5), self.dropp._standardized_data.shape)

    def test_standardized_error(self):
        with self.assertRaises(ModelNotFittedError):
            _ = self.dropp._standardized_data

        # TODO: test _get_eigenvectors: extra_dr_layer
        #       _get_correlation_matrix: CORR_KERNEL, ndim=2/3, CORRELATION_MATRIX_PLOT,


class TestDROPPGetCovarianceMatrix(unittest.TestCase):
    def setUp(self):
        self.dropp = DROPP()
        self.data_tensor = np.random.rand(100, 10, 5)

    def test_get_covariance_matrix_matrix_no_kernel(self):
        self.dropp = DROPP(ndim=2, kernel_kwargs={KERNEL_MAP: None})
        matrix_data = np.random.rand(100, 10)
        self.dropp.fit(matrix_data)

        cov_matrix = self.dropp.get_covariance_matrix()
        cov_expected = np.cov(self.dropp._standardized_data.T)

        self.assertEqual((10, 10), cov_matrix.shape)
        np_testing.assert_array_equal(cov_expected, cov_matrix)

    def test_get_covariance_matrix_matrix_with_kernel(self):
        self.dropp = DROPP(algorithm_name='tica', ndim=2, lag_time=10)
        matrix_data = np.random.rand(100, 10)
        self.dropp.fit(matrix_data)

        with patch('utils.algorithms.dropp.calculate_symmetrical_kernel_matrix',
                   return_value=np.random.rand(10, 10)) as mocker_patch:
            cov_matrix = self.dropp.get_covariance_matrix()
            raw_cov = np.cov(self.dropp._standardized_data[:-10].T)

            np_testing.assert_array_equal(raw_cov, mocker_patch.call_args[0][0])
            mocker_patch.assert_called_once()
        self.assertEqual((10, 10), cov_matrix.shape)

    def test_get_covariance_matrix_tensor_no_kernel(self):
        self.dropp = DROPP(kernel_kwargs={KERNEL_MAP: None})
        self.dropp.fit(self.data_tensor)

        cov_matrix = self.dropp.get_covariance_matrix()

        self.assertEqual((10 * 5, 10 * 5), cov_matrix.shape)
        self.assertEqual((10 * 5 * 10 * 5) - (10 * 10 * 5), np.count_nonzero(cov_matrix == 0))

    def test_get_covariance_matrix_tensor_with_kernel(self):
        self.dropp.fit(self.data_tensor)

        with patch('utils.algorithms.dropp.calculate_symmetrical_kernel_matrix',
                   return_value=np.random.rand(10, 10)) as mocker_patch:
            cov_matrix = self.dropp.get_covariance_matrix()

            mocker_patch.assert_called_once()
            self.assertEqual((10, 10), mocker_patch.call_args[0][0].shape)
        self.assertEqual((50, 50), cov_matrix.shape)

    # TODO: test _map__kernel_on with KERNEl_DIFFERENCE, KERNEL_MULTIPLICATION, ONES_ON_KERNEL_DIAG


class TestDROPPGetCombinedCovarianceMatrix(unittest.TestCase):
    def test_get_combined_covariance_matrix_tensor(self):
        self.dropp = DROPP(algorithm_name='tica', lag_time=10)
        self.tensor_data = np.random.rand(100, 10, 5)
        self.dropp.fit(self.tensor_data)

        with patch.object(MultiArrayPlotter, 'plot_tensor_layers') as plot_mocker:
            self.dropp.analyse_plot_type = COVARIANCE_MATRIX_PLOT
            combined_cov_matrix = self.dropp.get_combined_covariance_matrix()
            self.assertEqual((10, 10), combined_cov_matrix.shape)
            plot_mocker.assert_called_once()
