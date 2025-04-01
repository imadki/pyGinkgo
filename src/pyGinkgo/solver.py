# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

from pyGinkgo import pyGinkgoBindings as pGB
import numpy as np


def lobpcg_basic_standard_impl_(A, X0, nev,
                          T, itmax, tol,
                          A_products):
  """
  Knyazev, A. V. (2001)
  Toward the optimal preconditioned eigensolver: Locally optimal block preconditioned conjugate gradient method
  SIAM journal on scientific computing, 23(2), 517-541

  Parameters:
  A          : left  hand-side operator, symmetric positive definite, n-by-n
  B          : right hand-side operator, symmetric positive definite, n-by-n
  X0         : initial iterates, n-by-m (m < n)
  nev        : number of wanted eigenpairs, nev <= m
  T          : precondontioner, symmetric positive definite, n-by-n
  itmax      : maximum number of iterations
  tol        : tolerance used for convergence criterion
  A_products : if :implicit, the matrix products with A are updated implicitly
  B_products : if :implicit, the matrix products with B are updated implicitly

  Returns:
  Lambda : last iterates of least dominant eigenvalues, m-by-1
  X      : last iterates of least dominant eigenvectors, n-by-m
  res    : normalized norms of eigenresiduals, m-by-it
  """
  pass


def lobpcg(A, X0, nev,
           B=None, T=None, itmax=200, tol=1e-6,
           method='Skip_ortho',
           A_products='implicit',
           B_products='implicit'):
  """
  LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) method.
    
  Parameters:
  A          : left  hand-side operator, symmetric positive definite, n-by-n
  X0         : initial iterates, n-by-m (m < n)
  nev        : number of eigenvalues to compute.
  B          : right hand-side operator, symmetric positive definite, n-by-n
  T          : precondontioner, symmetric positive definite, n-by-n
  itmax      : maximum number of iterations
  tol        : tolerance used for convergence criterion
  method     : type of LOBPCG iterations among ('Basic', 'BLOPEX', 'Ortho', 'Skip_ortho')
  A_products : if "implicit", the matrix products with A are updated implicitly
  B_products : if "implicit", the matrix products with B are updated implicitly
  
  Returns:
  Lambda : last iterates of least dominant eigenvalues, m-by-1
  X      : last iterates of least dominant eigenvectors, n-by-m
  res    : normalized norms of eigenresiduals, m-by-it
  """
  pass
