import unittest
import numpy as np

from utils.matrix_tools import co_mad, ensure_matrix_symmetry


class MyTestCase(unittest.TestCase):
    def test_co_mad(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(np.allclose(co_mad(matrix), np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])))

    def test_ensure_matrix_symmetry(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertTrue(np.allclose(ensure_matrix_symmetry(matrix), np.array([[1, 2.5], [2.5, 4]])))


if __name__ == '__main__':
    unittest.main()
