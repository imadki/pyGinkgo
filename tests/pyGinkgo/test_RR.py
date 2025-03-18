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
    exphX = [[-0.65118902, -0.73662713], [0.2351141, 0.38042261]]
    expLambda = [1.38196601, 3.61803399]

    def test_can_default_solve(self):
        denseX = pGB.matrix.dense_float(np.array(self.X, dtype=np.float32))
        denseAX = pGB.matrix.dense_float(np.array(self.AX, dtype=np.float32))
        denseBX = pGB.matrix.dense_float(np.array(self.BX, dtype=np.float32))
        hX, Lambda = pg.RR1(denseX, denseAX, denseBX)
        assert Lambda.at(0) == self.expLambda[0]
        assert Lambda.at(1) == self.expLambda[1]
