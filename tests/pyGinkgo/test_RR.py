# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import sys

sys.path.append("../../")
import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB

import numpy as np
import os
import pytest


class TestSolve:
    executor = pGB.ReferenceExecutor()
    # TODO replace all dense_float
    fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
    mtx = pGB.matrix.read_Coo_float_int32(fn, executor)
    dim = mtx.get_size()
    rows = dim[0]
    cols = dim[1]
    rhs = pGB.matrix.dense_float(executor, (rows, 1))
    rhs.fill(1.0)
    initial_guess = pGB.matrix.dense_float(executor, (rows, 1))
    initial_guess.fill(0.0)
    X = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    AX = [[9.0, 27.0], [12.0, 30.0], [15.0, 33.0]]
    BX = [[1.0, 4.0], [4.0, 10.0], [9.0, 18.0]]
    exphX = [[-0.6748703, -0.7149942], [0.2473979, 0.37255105]]
    expLambda = [1.32522729, 4.07477271]

    def test_can_default_solve(self):
        denseX = pGB.matrix.dense_float(np.array(self.X, dtype=np.float32))
        denseAX = pGB.matrix.dense_float(np.array(self.AX, dtype=np.float32))
        denseBX = pGB.matrix.dense_float(np.array(self.BX, dtype=np.float32))
        hX, Lambda = pg.RR1(denseX, denseAX, denseBX)
        reshX = np.array(hX, dtype=np.float32)
        assert pytest.approx(Lambda.at(0), 1e-6) == self.expLambda[0]
        assert pytest.approx(Lambda.at(1), 1e-6) == self.expLambda[1]
        assert pytest.approx(reshX[0, 0], 1e-6) == self.exphX[0][0]
        assert pytest.approx(reshX[0, 1], 1e-6) == self.exphX[0][1]
        assert pytest.approx(reshX[1, 0], 1e-6) == self.exphX[1][0]
        assert pytest.approx(reshX[1, 1], 1e-6) == self.exphX[1][1]
