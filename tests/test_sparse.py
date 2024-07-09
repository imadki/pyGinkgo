# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import sys
import pyGinkgo
import pytest

sys.path.append("../build")


@pytest.mark.parametrize("matrix_format", ["Coo", "Csr"])
class TestSparseMatrix:
    ref = pyGinkgo.ReferenceExecutor()

    def test_can_create_sparse_matrix(self, matrix_format):
        ctr = getattr(pyGinkgo.matrix, matrix_format)
        sparse = ctr(self.ref)
        assert sparse == sparse
