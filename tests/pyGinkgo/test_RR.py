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
    values1 = [[1.0, 2.0, -1.0], [3.0, 4.0, -1.0], [5.0, 6.0, -1.0]]
    values2 = [[1.0, 2.0, -1.0], [3.0, 4.0, -1.0], [5.0, 6.0, -1.0]]
    values3 = [[1.0, 2.0, -1.0], [3.0, 4.0, -1.0], [5.0, 6.0, -1.0]]

    def test_can_default_solve(self):
        dense1 = pGB.matrix.dense(np.array(self.values1))
        dense2 = pGB.matrix.dense(np.array(self.values2))
        dense3 = pGB.matrix.dense(np.array(self.values3))
        pg.RR(dense1, dense2, dense3)
