# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import pytest
import sys
import numpy as np
import cupy

sys.path.append("../build")
import pyGinkgoBindings as pGB


@pytest.mark.gpu
def test_can_create_array_from_cupy():
    if pGB.CudaExecutor.get_num_devices() < 1:
        pytest.skip("CUDA is not available")

    executor = pGB.CudaExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cupy_array = cupy.array(np_array)
    arr = pGB.base.array(executor, cupy_array)
    arr_copy = pGB.base.array(executor, arr)
    assert arr.get_size() == arr_copy.get_size()
    assert pGB.base.reduce_add(arr, 0.0) == 15.0


@pytest.mark.gpu
def test_can_copy_array_to_cupy():
    if pGB.CudaExecutor.get_num_devices() < 1:
        pytest.skip("CUDA is not available")

    executor = pGB.CudaExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr = pGB.base.array(executor, cupy_array)
    arr_copy = pGB.base.array(executor, arr)
    cupy_array = cupy.array(arr)
