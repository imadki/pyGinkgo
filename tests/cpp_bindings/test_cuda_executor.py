# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import pytest
import sys
import numpy as np

sys.path.append("../../")
import pyGinkgoBindings as pGB


@pytest.mark.gpu
def test_array_cuda():
    if pGB.CudaExecutor.get_num_devices() < 1:
        pytest.skip("CUDA is not available")

    executor = pGB.CudaExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr = pGB.base.array(executor, np_array)
    arr_copy = pGB.base.array(executor, arr)
    assert arr.get_size() == arr_copy.get_size()
    assert pGB.base.reduce_add(arr, 0.0) == 15.0


@pytest.mark.gpu
def test_device_id():
    if pGB.CudaExecutor.get_num_devices() < 1:
        pytest.skip("CUDA is not available")

    executor = pGB.CudaExecutor(0)
    assert executor.device_id == 0
    with pytest.raises(AttributeError):
        executor.device_id = 1


@pytest.mark.gpu
def test_master():
    if pGB.CudaExecutor.get_num_devices() < 1:
        pytest.skip("CUDA is not available")

    master = pGB.ReferenceExecutor()
    executor = pGB.CudaExecutor(0, master)
    assert id(executor.master) == id(master)
    with pytest.raises(AttributeError):
        executor.master = pGB.ReferenceExecutor()
