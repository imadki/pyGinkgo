# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import pytest
import numpy as np

import pyGinkgo.pyGinkgoBindings as pGB

from test_utils import verify_dense_vec, d_type_map


@pytest.mark.parametrize("data_type", list(d_type_map.keys()))
class TestCuda:
    def test_array_cuda(self, data_type):
        if pGB.CudaExecutor.get_num_devices() < 1:
            pytest.skip("CUDA is not available")

        executor = pGB.CudaExecutor()
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=d_type_map[data_type])
        array_cls = getattr(pGB.base, "array_" + data_type)
        arr = array_cls(executor, np_array)
        arr_copy = array_cls(executor, arr)
        assert arr.get_size() == arr_copy.get_size()
        assert pGB.base.reduce_add(arr, 0.0) == 15.0

    def test_dense_copy_to_host(self, data_type):
        if pGB.CudaExecutor.get_num_devices() < 1:
            pytest.skip("CUDA is not available")

        master = pGB.ReferenceExecutor()
        executor = pGB.CudaExecutor(master=master)
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=d_type_map[data_type])
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(executor, np_array)
        dense_on_master = dense.copy_to_host()
        assert dense.get_executor() == executor
        assert dense_on_master.get_executor() == master
        assert id(dense) != id(dense_on_master)
        verify_dense_vec(dense_on_master, np_array)

    def test_device_id(self, data_type):
        if pGB.CudaExecutor.get_num_devices() < 1:
            pytest.skip("CUDA is not available")

        executor = pGB.CudaExecutor(0)
        assert executor.device_id == 0
        with pytest.raises(AttributeError):
            executor.device_id = 1

    def test_can_syncronize():
        if pGB.CudaExecutor.get_num_devices() < 1:
            pytest.skip("CUDA is not available")

        executor = pGB.CudaExecutor(0)
        executor.syncronize()


    def test_master(self, data_type):
        if pGB.CudaExecutor.get_num_devices() < 1:
            pytest.skip("CUDA is not available")
        master = pGB.ReferenceExecutor()
        executor = pGB.CudaExecutor(0, master)
        assert id(executor.master) == id(master)
        with pytest.raises(AttributeError):
            executor.master = pGB.ReferenceExecutor()
