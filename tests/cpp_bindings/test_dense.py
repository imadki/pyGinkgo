# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import pytest
import numpy as np

import pyGinkgo.pyGinkgoBindings as pGB

from test_utils import verify_dense_vec, verify_within_precision

d_type_map = {
    "half": np.float16,
    "float": np.float32,
    "double": np.float64,
}

d_precision_map = {
    "half": 1e-3,
    "float": 1e-6,
    "double": 0,
}


@pytest.mark.parametrize("data_type", list(d_type_map.keys()))
class TestDense:
    values = [1.0, 2.0, -1.0, 3.0, 4.0, -1.0, 5.0, 6.0, -1.0]
    values2d = [[1.0, 2.0, -1.0], [3.0, 4.0, -1.0], [5.0, 6.0, -1.0]]
    ref = pGB.ReferenceExecutor()

    def test_can_create_dense_linop(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(self.ref)
        assert dense == dense

    def test_can_create_dense_linop_with_dim(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(self.ref, (len(self.values), 1))
        assert dense == dense

    def test_can_create_dense_linop_with_dim_stride(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(self.ref, (len(self.values), 1), 1)
        assert dense == dense

    def test_can_create_dense_from_1D_np_array_with_default_exec(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(np.array(self.values, dtype=d_type_map[data_type]))
        verify_dense_vec(dense, self.values)

    def test_can_copy_construct(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense_a = ctr(np.array(self.values, dtype=d_type_map[data_type]))
        dense_b = ctr(self.ref, dense_a)
        verify_dense_vec(dense_b, self.values)

    def test_can_create_dense_from_1D_np_array(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(self.ref, np.array(self.values, dtype=d_type_map[data_type]))
        verify_dense_vec(dense, self.values)

    def test_can_create_dense_from_1D_np_array_with_dim_stride(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(
            self.ref,
            (len(self.values), 1),
            np.array(self.values, dtype=d_type_map[data_type]),
            1,
        )
        verify_dense_vec(dense, self.values)

    def test_can_read_mtx_file(self, data_type):
        # Matrix market format stores column major order
        fn = os.path.dirname(os.path.realpath(__file__)) + "/dense_example.mtx"
        read_func = getattr(pGB.matrix, "read_dense_" + data_type)
        dense = read_func(fn, self.ref)

        assert dense.get_num_stored_elements() == 12
        verify_within_precision(dense.at(0, 2), 0.2785, d_precision_map[data_type])
        verify_within_precision(dense.at(2), 0.2785, d_precision_map[data_type])
        verify_within_precision(dense.at(2, 2), 0.9575, d_precision_map[data_type])

    def test_can_create_dense_from_2D_np_array_with_default_exec(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(np.array(self.values2d, dtype=d_type_map[data_type]))
        assert dense.at(0, 1) == 2.0
        assert dense.at(0, 2) == -1.0
        assert dense.at(1, 1) == 4.0

    def test_can_create_dense_from_1D_np_array_with_stride(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(
            self.ref, (3, 3), np.array([self.values], dtype=d_type_map[data_type]), 3
        )

        assert dense.get_num_stored_elements() == len(self.values)
        assert dense.at(2) == -1.0
        assert dense.at(0, 2) == dense.at(2)
        assert dense.at(2, 1) == 6

    def test_dense_support_basic_functionality(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(
            self.ref, (3, 3), np.array([self.values], dtype=d_type_map[data_type]), 3
        )

        dense.scale(5)
        assert dense.at(2) == -5.0
        assert dense.at(1, 2) == dense.at(5)

        dense.inv_scale(5)
        assert dense.at(2) == -1.0
        assert dense.at(1, 2) == dense.at(5)

    def test_can_apply_to_transpose(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        aT = np.array([self.values], dtype=d_type_map[data_type])
        a = aT.T
        dense_a = ctr(aT)
        dense_b = ctr(a)
        result = ctr(self.ref, (1, 1))
        dense_a.apply(dense_b, result)
        assert result.at(0) == aT.dot(a)[0, 0]

    def test_add_scaled(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        a = np.array(self.values, dtype=d_type_map[data_type])
        dense_a = ctr(a)
        alpha = 2.0
        dense_alpha = ctr(np.array([alpha], dtype=d_type_map[data_type]))
        dense_a.add_scaled(dense_alpha, dense_a)
        verify_dense_vec(dense_a, (1 + alpha) * a)

    def test_sub_scaled(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        a = np.array(self.values, dtype=d_type_map[data_type])
        dense_a = ctr(a)
        alpha = 0.5
        dense_alpha = ctr(np.array([alpha], dtype=d_type_map[data_type]))
        dense_a.sub_scaled(dense_alpha, dense_a)
        verify_dense_vec(dense_a, (1 - alpha) * a)

    def test_dense_can_return_size(self, data_type):
        ctr = getattr(pGB.matrix, "dense_" + data_type)
        dense = ctr(
            self.ref, (3, 3), np.array(self.values, dtype=d_type_map[data_type]), 3
        )
        assert dense.get_size()[0] == 3
        assert dense.get_size()[1] == 3
