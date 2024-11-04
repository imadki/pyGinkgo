# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import sys
sys.path.append("../../")
import pyGinkgo as pg

import os
import pytest
import numpy as np

class TestSolve:
    executor = pg.ReferenceExecutor()
    fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
    mtx = pg.matrix.read_Coo(fn, executor)
    dim = mtx.get_size()
    rows =  dim[0]
    cols =  dim[1]
    rhs = pg.matrix.dense(executor, (rows, 1))
    rhs.fill(1.0)
    initial_guess = pg.matrix.dense(executor, (rows, 1))
    initial_guess.fill(0.0)

    def test_can_default_solve(self):
        logger, result = pg.solve(self.mtx, self.rhs, self.initial_guess)

        assert logger.has_converged()
        assert logger.get_num_iterations() < 1000

    def test_can_solve_with_solver_args_pcg(self):
        solver_args = {
            "type": "solver::Cg",
            "preconditioner": {
                "type": "preconditioner::Ilu",
                "l_solver_type": "solver::LowerTrs",
                "reverse_apply": False,
                "factorization": {"type": "factorization::ParIlu"},
            },
            "criteria": [
                {"type": "Iteration", "max_iters": 1000},
                {"type": "ResidualNorm", "reduction_factor": 1e-7},
            ],
        }
        logger, result = pg.solve(self.mtx, self.rhs, self.initial_guess, solver_args=solver_args)

        assert logger.has_converged()
        assert logger.get_num_iterations() < 1000

    def test_can_solve_with_solver_args_gmres(self):
        solver_args = {
            "type": "solver::Gmres",
            "preconditioner": {
                "type": "preconditioner::Ilu",
                "l_solver_type": "solver::LowerTrs",
                "reverse_apply": False,
                "factorization": {"type": "factorization::ParIlu"},
            },
            "criteria": [
                {"type": "Iteration", "max_iters": 1000},
                {"type": "ResidualNorm", "reduction_factor": 1e-7},
            ],
        }
        logger, result = pg.solve(self.mtx, self.rhs, self.initial_guess, solver_args=solver_args)

        assert logger.has_converged()
        assert logger.get_num_iterations() < 1000
