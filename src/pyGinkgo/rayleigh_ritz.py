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

    G1 = pGB.matrix.dense(X)
    XT.apply(AX, G1)

    G2 = pGB.matrix.dense(X)
    XT.apply(BX, G2)

    # Find L s.t. L @ L.T = G2
    direct1 = pGB.solver.direct(executor, G2, factorization="Cholesky")
    G2P = pGB.matrix.dense(X)
    direct1.apply(G1, G2P)
    G2PT = G2P.T()
    direct2 = pGB.solver.direct(executor, G2, factorization="Cholesky")
    G2PP = pGB.matrix.dense(X)
    direct2.apply(G2PT, G2PP)

    torchG2 = torch.as_tensor(np.array(G2PP))
    L, Q = torch.linalg.eigh(torchG2)
    Lambda = pGB.matrix.dense(executor, L.__array__())
    hY = pGB.matrix.dense(executor, Q.__array__())

    G2T = G2.T()
    direct3 = pGB.solver.direct(executor, G2T, factorization="Cholesky")
    hX = pGB.matrix.dense(hY)
    direct3.apply(hY, hX)

    return hX, Lambda
