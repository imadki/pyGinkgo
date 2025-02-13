# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import sys

sys.path.append("../../")
import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB

import numpy as np
import os


class TestSolve:
    executor = pGB.ReferenceExecutor()
    fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
    mtx = pGB.matrix.read_Coo(fn, executor)
    dim = mtx.get_size()
    rows = dim[0]
    cols = dim[1]
    rhs = pGB.matrix.dense(executor, (rows, 1))
    rhs.fill(1.0)
    initial_guess = pGB.matrix.dense(executor, (rows, 1))
    initial_guess.fill(0.0)
    X = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    AX = [[9.0, 27.0], [12.0, 30.0], [15.0, 33.0]]
    BX = [[1.0, 4.0], [4.0, 10.0], [9.0, 18.0]]

    def test_can_default_solve(self):
        denseX = pGB.matrix.dense(np.array(self.X))
        denseAX = pGB.matrix.dense(np.array(self.AX))
        denseBX = pGB.matrix.dense(np.array(self.BX))
        hX, Lambda = pg.RR1(denseX, denseAX, denseBX)
        assert hX[0][0] == 0
        assert Lambda[0][0] == 0
