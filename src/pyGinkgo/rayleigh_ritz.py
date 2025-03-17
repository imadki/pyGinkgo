# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

from pyGinkgo import pyGinkgoBindings as pGB

import torch
import numpy as np


def RR1(X, AX, BX):
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
    executor = pGB.ReferenceExecutor()
    XT = X.T()

    # compute G1 = X.T @ AX
    G1 = pGB.matrix.dense(X)
    XT.apply(AX, G1)

    # compute G2 = X.T @ BX
    G2 = pGB.matrix.dense(X)
    XT.apply(BX, G2)

    # compute G2P G2' = L^(-1) @ G1
    # Find L s.t. L @ L.T = G2
    direct1 = pGB.solver.direct(executor, G2, factorization="Cholesky")
    G2P = pGB.matrix.dense(X)
    direct1.apply(G1, G2P)
    G2PT = G2P.T()

    # TODO: remove
    # direct2 = pGB.solver.direct(executor, G2, factorization="Cholesky")

    # compute G2PP G2'' = L^(-1) @ G2P.T = L^(-1) @ G1 @ L^(-T)
    G2PP = pGB.matrix.dense(X)
    direct1.apply(G2PT, G2PP)

    torchG2 = torch.as_tensor(np.array(G2PP))
    L, Q = torch.linalg.eigh(torchG2)
    Lambda = pGB.matrix.dense(executor, L.__array__())
    hY = pGB.matrix.dense(executor, Q.__array__())

    G2T = G2.T()
    direct3 = pGB.solver.direct(executor, G2T, factorization="Cholesky")
    hX = pGB.matrix.dense(hY)
    direct3.apply(hY, hX)

    return hX, Lambda
