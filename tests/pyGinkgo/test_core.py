# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2025 pyGinkgo authors

# Most of the code is copilot generated

import pytest
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
from pyGinkgo.core import as_array, as_tensor, read, factor, eigen_solve


@pytest.mark.parametrize(
    "data, device, dtype", [([1, 2, 3], "cpu", "float"), ([4, 5, 6], "gpu", "int")]
)
def test_as_array_parametrized(data, device, dtype):
    with patch("pyGinkgo.core.pg.device") as mock_device, patch(
        "pyGinkgo.core.pGB.base.array_float"
    ) as mock_array_cls:
        mock_device.return_value = "mock_executor"
        mock_array = MagicMock()
        mock_array_cls.return_value = mock_array

        result = as_array(data, device=device, dtype=dtype)

        mock_device.assert_called_once_with(device)
        mock_array_cls.assert_called_once_with("mock_executor", data)
        assert result == mock_array


@pytest.mark.parametrize(
    "dim, device, dtype, fill",
    [((2, 2), "cpu", "float", 1.0), ((3, 3), "gpu", "int", 0.0)],
)
def test_as_tensor_parametrized(dim, device, dtype, fill):
    with patch("pyGinkgo.core.pg.device") as mock_device, patch(
        "pyGinkgo.core.pGB.matrix.dense_float"
    ) as mock_dense_cls:
        mock_device.return_value = "mock_executor"
        mock_tensor = MagicMock()
        mock_dense_cls.return_value = mock_tensor

        result = as_tensor(dim=dim, device=device, dtype=dtype, fill=fill)

        mock_device.assert_called_once_with(device)
        mock_dense_cls.assert_called_once_with("mock_executor", dim)
        mock_tensor.fill.assert_called_once_with(fill)
        assert result == mock_tensor


class TestCoreFunctions(unittest.TestCase):
    @pytest.mark.parametrize(
        "data, device, dtype", [([1, 2, 3], "cpu", "float"), ([4, 5, 6], "gpu", "int")]
    )
    @patch("pyGinkgo.core.pg.device")
    @patch("pyGinkgo.core.pGB.base.array_float")
    def test_as_array(self, mock_array_cls, mock_device, data, device, dtype):
        mock_device.return_value = "mock_executor"
        mock_array = MagicMock()
        mock_array_cls.return_value = mock_array

        result = as_array(data, device=device, dtype=dtype)

        mock_device.assert_called_once_with(device)
        mock_array_cls.assert_called_once_with("mock_executor", data)
        self.assertEqual(result, mock_array)

    @pytest.mark.parametrize(
        "dim, device, dtype, fill",
        [((2, 2), "cpu", "float", 1.0), ((3, 3), "gpu", "int", 0.0)],
    )
    @patch("pyGinkgo.core.pg.device")
    @patch("pyGinkgo.core.pGB.matrix.dense_float")
    def test_as_tensor(self, mock_dense_cls, mock_device, dim, device, dtype, fill):
        mock_device.return_value = "mock_executor"
        mock_tensor = MagicMock()
        mock_dense_cls.return_value = mock_tensor

        result = as_tensor(dim=dim, device=device, dtype=dtype, fill=fill)

        mock_device.assert_called_once_with(device)
        mock_dense_cls.assert_called_once_with("mock_executor", dim)
        mock_tensor.fill.assert_called_once_with(fill)
        self.assertEqual(result, mock_tensor)

    @patch("pyGinkgo.core.pg.device")
    @patch("pyGinkgo.core.pGB.matrix.read_dense_float")
    def test_read(self, mock_read_func, mock_device):
        mock_device.return_value = "mock_executor"
        mock_matrix = MagicMock()
        mock_read_func.return_value = mock_matrix

        result = read("test.mtx", format="dense", dtype="float", device="cpu")

        mock_device.assert_called_once_with("cpu")
        mock_read_func.assert_called_once_with("test.mtx", "mock_executor")
        self.assertEqual(result, mock_matrix)

    @patch("pyGinkgo.core.pg.device")
    @patch("pyGinkgo.core.pGB.factorization.factorization")
    def test_factor(self, mock_factorization, mock_device):
        mock_device.return_value = "mock_executor"
        mock_fact = MagicMock()
        mock_factorization.return_value = mock_fact
        mock_fact.get_upper_factor.return_value = "upper_factor"

        result = factor("A", kind="Upper", device="cpu")

        mock_device.assert_called_once_with("cpu")
        mock_factorization.assert_called_once_with("mock_executor", "A")
        self.assertEqual(result, "upper_factor")

    @patch("pyGinkgo.core.torch.linalg.eigh")
    @patch("pyGinkgo.core.pGB.matrix.dense_float")
    def test_eigen_solve(self, mock_dense_cls, mock_eigh):
        mock_eigh.return_value = (np.array([1.0, 2.0]), np.array([[1, 0], [0, 1]]))
        mock_dense = MagicMock()
        mock_dense_cls.return_value = mock_dense

        A = MagicMock()
        A.get_executor.return_value = "mock_executor"

        Lambda, Q = eigen_solve(A)

        mock_dense_cls.assert_any_call("mock_executor", np.array([1.0, 2.0]))
        mock_dense_cls.assert_any_call("mock_executor", np.array([[1, 0], [0, 1]]))
        self.assertEqual(Lambda, mock_dense)
        self.assertEqual(Q, mock_dense)


if __name__ == "__main__":
    unittest.main()
