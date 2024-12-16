# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import numpy as np

import pyGinkgo.pyGinkgoBindings as pGB

from test_utils import verify_dense_vec


class TestDense:
    values = [1.0, 2.0, -1.0, 3.0, 4.0, -1.0, 5.0, 6.0, -1.0]
    values2d = [[1.0, 2.0, -1.0], [3.0, 4.0, -1.0], [5.0, 6.0, -1.0]]
    ref = pGB.ReferenceExecutor()

    def test_can_create_dense_linop(self):
        dense = pGB.matrix.dense(self.ref)
        assert dense == dense

    def test_can_create_dense_linop_with_dim(self):
        dense = pGB.matrix.dense(self.ref, (len(self.values), 1))
        assert dense == dense

    def test_can_create_dense_linop_with_dim_stride(self):
        dense = pGB.matrix.dense(self.ref, (len(self.values), 1), 1)
        assert dense == dense

    def test_can_create_dense_from_1D_np_array_with_default_exec(self):
        dense = pGB.matrix.dense(np.array(self.values))
        verify_dense_vec(dense, self.values)

    def test_can_copy_construct(self):
        dense_a = pGB.matrix.dense(np.array(self.values))
        dense_b = pGB.matrix.dense(self.ref, dense_a)
        verify_dense_vec(dense_b, self.values)

    def test_can_create_dense_from_1D_np_array(self):
        dense = pGB.matrix.dense(self.ref, np.array(self.values))
        verify_dense_vec(dense, self.values)

    def test_can_create_dense_from_1D_np_array_with_dim_stride(self):
        dense = pGB.matrix.dense(
            self.ref, (len(self.values), 1), np.array(self.values), 1
        )
        verify_dense_vec(dense, self.values)

    def test_can_read_mtx_file(self):
        # Matrix market format stores column major order
        fn = os.path.dirname(os.path.realpath(__file__)) + "/dense_example.mtx"
        dense = pGB.matrix.read_dense(fn, self.ref)

        assert dense.get_num_stored_elements() == 12
        assert dense.at(0, 2) == 0.2785
        assert dense.at(2) == 0.2785
        assert dense.at(2, 2) == 0.9575

    def test_can_create_dense_from_2D_np_array_with_default_exec(self):
        dense = pGB.matrix.dense(np.array(self.values2d))
        assert dense.at(0, 1) == 2.0
        assert dense.at(0, 2) == -1.0
        assert dense.at(1, 1) == 4.0

    def test_can_create_dense_from_1D_np_array_with_stride(self):
        dense = pGB.matrix.dense(self.ref, (3, 3), np.array([self.values]), 3)

        assert dense.get_num_stored_elements() == len(self.values)
        assert dense.at(2) == -1.0
        assert dense.at(0, 2) == dense.at(2)
        assert dense.at(2, 1) == 6

    def test_dense_support_basic_functionality(self):
        dense = pGB.matrix.dense(self.ref, (3, 3), np.array([self.values]), 3)

        dense.scale(5)
        assert dense.at(2) == -5.0
        assert dense.at(1, 2) == dense.at(5)

        dense.inv_scale(5)
        assert dense.at(2) == -1.0
        assert dense.at(1, 2) == dense.at(5)

    def test_can_apply_to_transpose(self):
        aT = np.array([self.values])
        a = aT.T
        dense_a = pGB.matrix.dense(aT)
        dense_b = pGB.matrix.dense(a)
        result = pGB.matrix.dense(self.ref, (1, 1))
        dense_a.apply(dense_b, result)
        result.at(0) == aT.dot(a)[0, 0]

    def test_add_scaled(self):
        a = np.array(self.values)
        dense_a = pGB.matrix.dense(a)
        alpha = 2.0
        dense_alpha = pGB.matrix.dense(np.array([alpha]))
        dense_a.add_scaled(dense_alpha, dense_a)
        verify_dense_vec(dense_a, (1 + alpha) * a)

    def test_sub_scaled(self):
        a = np.array(self.values)
        dense_a = pGB.matrix.dense(a)
        alpha = 0.5
        dense_alpha = pGB.matrix.dense(np.array([alpha]))
        dense_a.sub_scaled(dense_alpha, dense_a)
        verify_dense_vec(dense_a, (1 - alpha) * a)

    def test_dense_can_return_size(self):
        dense = pGB.matrix.dense(
            self.ref, (3, 3), np.array(self.values), 3)
        assert dense.get_size()[0] == 3
        assert dense.get_size()[1] == 3



