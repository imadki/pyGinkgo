# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import pytest
import numpy as np

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB

d_type_map = {
    "half": np.float16,
    "float": np.float32,
    "double": np.float64,
}


# TODO: it takes 6s to run test_can_solve_with_solver_args_gmres[float] and test_can_default_solve[float]
class TestSolve:
    # A bit better parametrization approach using fixtures
    # It allows to reduce amount of code repetition
    # https://stackoverflow.com/a/50135020/8302811
    @pytest.fixture(autouse=True, params=list(d_type_map.keys()))
    def __post_init__(self, request):
        data_type = request.param
        self.executor = pGB.ReferenceExecutor()
        self.fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"

        self.reader_cls = getattr(pGB.matrix, f"read_Coo_{data_type}_int32")
        self.mtx = self.reader_cls(self.fn, self.executor)
        self.dim = self.mtx.get_size()
        self.rows = self.dim[0]
        self.cols = self.dim[1]

        self.dense_cls = getattr(pGB.matrix, f"dense_{data_type}")
        self.rhs = self.dense_cls(self.executor, (self.rows, 1))
        self.rhs.fill(1.0)
        self.initial_guess = self.dense_cls(self.executor, (self.rows, 1))
        self.initial_guess.fill(0.0)

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
        logger, result = pg.solve(
            self.mtx, self.rhs, self.initial_guess, solver_args=solver_args
        )

        assert logger.has_converged()
        assert logger.get_num_iterations() < 1000

        # Check if result entries are non zero
        npresult = np.array(result)
        assert len(npresult) > 0
        assert len(np.nonzero(npresult)[0]) == len(npresult)

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
        logger, result = pg.solve(
            self.mtx, self.rhs, self.initial_guess, solver_args=solver_args
        )

        assert logger.has_converged()
        assert logger.get_num_iterations() < 1000

        # Check if result entries are non zero
        npresult = np.array(result)
        assert len(npresult) > 0
        assert len(np.nonzero(npresult)[0]) == len(npresult)
