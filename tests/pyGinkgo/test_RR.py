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
    X = [[1, 4], [2, 5], [3, 6]]
    AX = [[9, 27], [12, 30], [15, 33]]
    BX = [[1, 4], [4, 10], [9, 18]]

    def test_can_default_solve(self):
        denseX = pGB.matrix.dense(np.array(self.X))
        denseAX = pGB.matrix.dense(np.array(self.AX))
        denseBX = pGB.matrix.dense(np.array(self.BX))
        hX, Lambda = pg.RR1(denseX, denseAX, denseBX)
        assert hX[0][0] == 0
        assert Lambda[0][0] == 0
