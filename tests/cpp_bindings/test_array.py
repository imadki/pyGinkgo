# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors


import pytest
import numpy as np

import pyGinkgo.pyGinkgoBindings as pGB

d_type_map = {
    "int32": np.int32,
    "int64": np.longlong,
    "half": np.float16,
    "float": np.float32,
    "double": np.float64,
}


@pytest.mark.parametrize("data_type", list(d_type_map.keys()))
class TestArray:
    ref = pGB.ReferenceExecutor()
    size = 5
    lst = [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_init_empty_array(self, data_type):
        ctr = getattr(pGB.base, "array_" + data_type)
        arr = ctr(self.ref, self.size)
        assert arr.get_size() == self.size

    def test_can_fill_array(self, data_type):
        ctr = getattr(pGB.base, "array_" + data_type)
        arr = ctr(self.ref, self.size)
        fill_value = d_type_map[data_type](10)
        arr.fill(fill_value)
        assert arr.at(0) == fill_value

    def test_can_instantiate_array_from_numpy_array(self, data_type):
        np_array = np.array(self.lst, dtype=d_type_map[data_type])
        ctr = getattr(pGB.base, "array_" + data_type)
        arr = ctr(self.ref, np_array)
        assert arr.get_size() == len(np_array)
        for i in range(self.size):
            expect = d_type_map[data_type](self.lst[i])
            assert arr.at(i) == expect

    def test_can_copy_construct_array(self, data_type):
        ctr = getattr(pGB.base, "array_" + data_type)
        np_array = np.array(self.lst, dtype=d_type_map[data_type])
        arr = ctr(self.ref, np_array)
        arr_copy = ctr(self.ref, arr)

        assert arr.get_size() == arr_copy.get_size()

    def test_can_construct_np_array_from_gko_array(self, data_type):
        ctr = getattr(pGB.base, "array_" + data_type)
        np_array = np.array(self.lst, dtype=d_type_map[data_type])
        arr = ctr(self.ref, np_array)
        np_array_copy = np.array(arr)

        assert np.array_equal(np_array, np_array_copy)

    def test_can_reduce_add_array(self, data_type):
        ctr = getattr(pGB.base, "array_" + data_type)
        np_array = np.array(self.lst, dtype=d_type_map[data_type])
        arr = ctr(self.ref, np_array)
        init_value = d_type_map[data_type](0.0)
        expected_value = d_type_map[data_type](15.0)
        assert pGB.base.reduce_add(arr, init_value) == expected_value
