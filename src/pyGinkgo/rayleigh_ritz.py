# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

from pyGinkgo import pyGinkgoBindings as pGB


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
    XT = X.T

    G1 = pGB.matrix.dense(X)
    XT.apply(AX, G1)

    G2 = pGB.matrix.dense(X)
    XT.apply(BX, G2)

    # Find L s.t. L @ L.T = G2
    # TODO check on Ginkgo how cholesky works
    # TODO needs cholesky bindings
    L = cholesky(G2, lower=True)

    # TODO use view from Ginkgo here
    # G2[:] = L^(-1) @ G1 @ L^(-T)
    # TODO needs triangular bindings
    # G2[:] = solve_triangular(L, G1, lower=True, overwrite_b=True)
    # G2[:] = solve_triangular(L, G2.T, lower=True)

    # # TODO eigh -> use pytorch
    # Lambda, hY = eigh(G2)

    # # hX = L^(-T) @ hY
    # hX = solve_triangular(L, hY, trans="T", lower=True)

    return hX, Lambda
