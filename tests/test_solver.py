# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import sys
import pyGinkgoBindings as pgb
import pytest

sys.path.append("../build")


@pytest.mark.parametrize("solver_name", ["gmres"])
class TestSparseMatrix:
    ref = pgb.ReferenceExecutor()

    def test_unpreconditioned_solver(self, solver_name):
        reader = getattr(pgb.matrix, "read_Coo")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
        mtx = reader(fn, self.ref)
        solver_ctr = getattr(pgb.solver, solver_name)
        solver = solver_ctr(self.ref, mtx, 1000, 10, 1e-06)
        logger = solver.initialize_logger()
        assert logger.has_converged() == False

        dim = mtx.get_size()
        assert dim[0] == dim[1]
        rhs = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        rhs.fill(1.0)
        initial_guess = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        initial_guess.fill(0.0)
        solver = solver.apply(rhs, initial_guess)

        assert logger.has_converged() == True
