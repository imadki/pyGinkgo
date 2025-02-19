# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import pytest
import numpy as np

import pyGinkgo.pyGinkgoBindings as pgb

from test_utils import d_type_map


@pytest.mark.parametrize("solver_name", ["direct"])
@pytest.mark.parametrize("data_type", list(d_type_map.keys()))
class TestDirectSolverBinding:
    ref = pgb.ReferenceExecutor()
    values = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    dim = (3, 3)

    solver_args = {"direct": {"factorization": "Cholesky"}}

    def test_direct_solver(self, solver_name, data_type):
        dense_cls = getattr(pgb.matrix, "dense_" + data_type)
        mtx = dense_cls(
            self.ref, (3, 3), np.array(self.values, dtype=d_type_map[data_type]), 3
        )
        exp = np.array([1, 1 / 2.0, 1 / 3.0], dtype=d_type_map[data_type])
        solver_cls = getattr(pgb.solver, f"{solver_name}_{data_type}_int32")
        args = self.solver_args[solver_name]
        solver = solver_cls(exec=self.ref, system_matrix=mtx, **args)

        rhs = dense_cls(mtx.get_executor(), (self.dim[0], 1))
        rhs.fill(1.0)
        x = dense_cls(mtx.get_executor(), (self.dim[0], 1))
        x.fill(0.0)
        solver.apply(rhs, x)

        res = np.array(x)
        assert np.all(exp) == np.all(res)
