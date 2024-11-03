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

    def test_can_default_solve(self):
        fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
        ref = pg.ReferenceExecutor()
        mtx = pg.matrix.read_Coo(fn, ref)

        dim = mtx.get_size()
        rhs = pg.matrix.dense(mtx.get_executor(), (dim[0], 1))
        rhs.fill(1.0)
        initial_guess = pg.matrix.dense(mtx.get_executor(), (dim[0], 1))
        initial_guess.fill(0.0)
        pg.solve(mtx, rhs, initial_guess)

