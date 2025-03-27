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

def triangular_solve(a,b, executor="Reference", kind="Upper", dtype="float", itype="int32"):
    ctor = getattr(pGB.solver, f"{kind}Trs_{dtype}_{itype}")
    exec_obj = getattr(pGB, executor + "Executor")()
    trs = ctor(exec_obj, a)
    dim = (a.get_size()[1], b.get_size()[1])
    res = pg.as_tensor(executor=executor, dim=dim, dtype=dtype)
    #G2P = pGB.matrix.dense_float(executor, dim)
    trs.apply(b, res)
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
    G2P = triangular_solve(L, G1, executor=executor,  kind="Lower", dtype=dtype)
    LT = L.T()

    # compute G2PP G2'' = L^(-1) @ G2P.T = L^(-1) @ G1 @ L^(-T)
    G2PP = triangular_solve(L, G2P.T(), executor=executor,  kind="Lower", dtype=dtype)

    torchG2 = torch.as_tensor(np.array(G2PP))
    L, Q = torch.linalg.eigh(torchG2)
    Lambda = pGB.matrix.dense_float(exec_obj, L.__array__())
    hY = pGB.matrix.dense_float(exec_obj, Q.__array__())

    hX = triangular_solve(LT, hY, executor=executor,  kind="Upper", dtype=dtype)
    return hX, Lambda
