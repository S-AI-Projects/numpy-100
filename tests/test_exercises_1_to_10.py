import unittest
import numpy as np


class TestExercises1To10(unittest.TestCase):

    def test_ex01_import_numpy(self):
        """Exercise 1: Import numpy as np"""
        self.assertEqual(np.__name__, "numpy")

    def test_ex02_version_and_config(self):
        """Exercise 2: Version string and config callable"""
        self.assertIsInstance(np.__version__, str)
        self.assertTrue(callable(np.show_config))

    def test_ex03_null_vector_size_10(self):
        """Exercise 3: Null vector of size 10"""
        z = np.zeros(10)
        self.assertEqual(z.shape, (10,))
        self.assertTrue(np.all(z == 0))

    def test_ex04_memory_size(self):
        """Exercise 4: Memory size calculation"""
        z = np.zeros((10, 10))
        calculated_size = z.size * z.itemsize
        self.assertEqual(calculated_size, z.nbytes)
        self.assertEqual(calculated_size, 800)

    def test_ex05_numpy_info(self):
        """Exercise 5: np.info is callable for docs"""
        self.assertTrue(callable(np.info))

    def test_ex06_null_vector_fifth_value_1(self):
        """Exercise 6: 10-element zero vector with 5th element equal to 1"""
        z = np.zeros(10)
        z[4] = 1
        self.assertEqual(z.shape, (10,))
        self.assertEqual(z[4], 1)
        self.assertEqual(np.sum(z), 1)

    def test_ex07_range_10_to_49(self):
        """Exercise 7: Vector ranging from 10 to 49 inclusive"""
        z = np.arange(10, 50)
        self.assertEqual(len(z), 40)
        self.assertEqual(z[0], 10)
        self.assertEqual(z[-1], 49)
        self.assertTrue(np.array_equal(z, np.arange(10, 50)))

    def test_ex08_reverse_vector(self):
        """Exercise 8: Reverse a vector"""
        z = np.arange(50)
        z_rev = z[::-1]
        self.assertEqual(z_rev[0], 49)
        self.assertEqual(z_rev[-1], 0)
        self.assertTrue(np.array_equal(z_rev, np.arange(49, -1, -1)))

    def test_ex09_matrix_3x3_0_to_8(self):
        """Exercise 9: 3x3 matrix with values 0 to 8"""
        z = np.arange(9).reshape(3, 3)
        self.assertEqual(z.shape, (3, 3))
        self.assertEqual(z[0, 0], 0)
        self.assertEqual(z[2, 2], 8)
        self.assertTrue(np.array_equal(z[1], np.array([3, 4, 5])))

    def test_ex10_find_nonzero_indices(self):
        """Exercise 10: Find indices of non-zero elements from [1,2,0,0,4,0]"""
        raw = [1, 2, 0, 0, 4, 0]
        nz = np.nonzero(raw)
        self.assertTrue(np.array_equal(nz[0], np.array([0, 1, 4])))


if __name__ == "__main__":
    unittest.main()
