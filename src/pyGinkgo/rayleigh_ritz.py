# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

from pyGinkgo import pyGinkgoBindings as pGB

import torch


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
    executor = pgb.ReferenceExecutor()
    XT = X.T

    G1 = pGB.matrix.dense(X)
    XT.apply(AX, G1)

    G2 = pGB.matrix.dense(X)
    XT.apply(BX, G2)

    # Find L s.t. L @ L.T = G2
    direct1 = pgb.direct(G1)
    direct1.apply(L, G2)

    torchG2 = torch.astensor(G2)
    L, Q = eigh(torchG2)
    Lambda = pGB.matrix.dense(executor, L.__array__())
    hY = pGB.matrix.dense(executor, Q.__array__())

    direct2 = pgb.direct(Lambda)
    direct2.apply(hY, hX)

    return hX, Lambda
