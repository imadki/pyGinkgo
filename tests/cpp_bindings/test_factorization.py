# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import numpy as np

import pyGinkgo.pyGinkgoBindings as pgb


class TestFactorizationBinding:
    ref = pgb.ReferenceExecutor()
    values = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]

    def test_factorization(self):
        dense = pgb.matrix.dense_float(
            self.ref, np.array(self.values, dtype=np.float32)
        )
        factorization = pgb.factorization.factorization(
            self.ref, dense.convert_to_csr()
        )
        lower = factorization.get_lower_factor()
        assert lower.get_num_stored_elements() == 6
