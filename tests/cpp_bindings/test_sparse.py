# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import pytest
import numpy as np

import pyGinkgo.pyGinkgoBindings as pGB


def coo_rows_to_csr_rows(coo_rows: list) -> list:
    # https://stackoverflow.com/a/71339835/8302811
    n_rows = max(coo_rows) + 1
    ret = [0] * n_rows
    for i in coo_rows:
        ret[i] += 1
    for i in range(n_rows - 1):
        ret[i + 1] += ret[i]
    return [0] + ret


@pytest.mark.parametrize("matrix_format", ["Coo", "Csr"])
class TestSparseMatrix:
    # test a 5x5 symmetric matrix
    # A =
    # | 1  10  .  20  .  |
    # | 10  2  11  .  21 |
    # | .  11   3  12  . |
    # | 20  .  12  4  13 |
    # | .  21  .  13   5 |
    #

    coo_rows = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
    csr_rows = coo_rows_to_csr_rows(coo_rows)

    cols = [0, 1, 3, 0, 1, 2, 4, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4]
    values = [
        1.0,
        10.0,
        20.0,
        10.0,
        2.0,
        11.0,
        21.0,
        11.0,
        3.0,
        12.0,
        20.0,
        12.0,
        4.0,
        13.0,
        21.0,
        13.0,
        5.0,
    ]
    dense = [1, 1, 1, 1, 1]

    ref = pGB.ReferenceExecutor()

    def get_rows(self, matrix_format) -> list:
        if matrix_format == "Csr":
            return self.csr_rows
        else:
            return self.coo_rows

    def test_can_create_sparse_matrix(self, matrix_format):
        ctr = getattr(pGB.matrix, matrix_format)
        sparse = ctr(self.ref)
        assert sparse == sparse

    def test_can_create_sparse_from_np_arrays(self, matrix_format):
        ctr = getattr(pGB.matrix, matrix_format)
        coeffs = np.array(self.values, dtype=np.double)
        rows = np.array(self.get_rows(matrix_format), dtype=np.int32)
        cols = np.array(self.cols, dtype=np.int32)

        print(rows)
        sparse = ctr(self.ref, (5, 5), coeffs, cols, rows)
        assert sparse == sparse

    def test_can_create_sparse_from_gko_arrays(self, matrix_format):
        ctr = getattr(pGB.matrix, matrix_format)
        coeffs = pGB.base.array_double(self.ref, np.array(self.values, dtype=np.double))
        rows = pGB.base.array_int(
            self.ref, np.array(self.get_rows(matrix_format), dtype=np.int32)
        )
        cols = pGB.base.array_int(self.ref, np.array(self.cols, dtype=np.int32))

        sparse = ctr(self.ref, (5, 5), coeffs, cols, rows)
        assert sparse == sparse
        assert sparse.get_num_stored_elements() == coeffs.get_size()

    def test_can_apply_to_dense(self, matrix_format):
        ctr = getattr(pGB.matrix, matrix_format)
        coeffs = np.array(self.values, dtype=np.double)
        rows = np.array(self.get_rows(matrix_format), dtype=np.int32)
        cols = np.array(self.cols, dtype=np.int32)

        sparse = ctr(self.ref, (5, 5), coeffs, cols, rows)

        dense_b = pGB.matrix.dense(self.ref, np.array(self.dense, dtype=np.double))
        dense_x = pGB.matrix.dense(self.ref, np.array([0, 0, 0, 0, 0], dtype=np.double))

        sparse.apply(dense_b, dense_x)
        assert dense_x.at(0) == 31.0
        assert dense_x.at(1) == 44.0
        assert dense_x.at(2) == 26.0
        assert dense_x.at(3) == 49.0
        assert dense_x.at(4) == 39.0

    def test_can_read_from_mtx_file(self, matrix_format):
        reader = getattr(pGB.matrix, "read_" + matrix_format)
        fn = os.path.dirname(os.path.realpath(__file__)) + "/sparse_example.mtx"
        sparse = reader(fn, self.ref)

        assert sparse == sparse
