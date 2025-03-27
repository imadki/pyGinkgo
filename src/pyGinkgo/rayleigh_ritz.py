# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import pyGinkgo as pg
from pyGinkgo import pyGinkgoBindings as pGB

import torch
import numpy as np


def mul(a, b, dtype="float", executor="Reference"):
    """ helper function to perform a @ b """
    dimRes = (a.get_size()[0], b.get_size()[1])
    # executor = a.get_executor()
    res = pg.as_tensor(executor=executor, dim=dimRes, dtype=dtype)
    a.apply(b, res)
    return res

def RR1(X, AX, BX, dtype="float", executor="Reference"):
    """
    Computes m least dominant generalized eigenpairs of
    (A,B) with respect to the range of X using a
    Rayleigh-Ritz projection.

    Parameters:
    - X, AX, BX : Input matrices (dense)

    Returns:
    - hX     : Eigenvectors
    - Lambda : Eigenvalues
    """
    XT = X.T()
    # compute G1 = X.T @ AX
    G1 = mul(XT, AX, dtype=dtype, executor=executor)
    # compute G2 = X.T @ BX
    G2 = mul(XT, BX, dtype=dtype, executor=executor)

    exec_obj = pGB.ReferenceExecutor()
    # compute G2P G2' = L^(-1) @ G1
    # Find L s.t. L @ L.T = G2
    L = pg.factor(G2, kind="Lower")
    _, G2P = pg.solve(L, G1, kind="triangular", solver_args={"type": "Lower"})
    LT = L.T()

    # compute G2PP G2'' = L^(-1) @ G2P.T = L^(-1) @ G1 @ L^(-T)
    _, G2PP = pg.solve(L, G2P.T(), kind="triangular", solver_args={"type": "Lower"})

    Lambda, hY = pg.eigen_solve(G2PP)

    _, hX = pg.solve(LT, hY, kind="triangular", solver_args={"type": "Upper"})
    return hX, Lambda
