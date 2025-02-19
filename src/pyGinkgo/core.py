# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

from pyGinkgo import pyGinkgoBindings as pGB
import json

try:
    import torch

    torch_avail = True
except ImportError:
    torch_avail = False

valid_value_types = ["half", "float", "double"]
valid_index_types = ["int32", "int64"]
valid_dtypes = valid_value_types + valid_index_types
valid_executor = ["Reference", "Cuda"]


def as_array(obj, executor: str = "Reference", dtype="float"):
    """create a ginkgo array from a given object"""
    if not dtype in valid_dtypes:
        raise ValueError(
            "Not a valid dtype: "
            + dtype
            + " possible choices are: "
            + str(valid_dtypes)
        )
    if not executor in valid_executor:
        raise ValueError(
            "Not a valid executor: "
            + dtype
            + " possible choices are: "
            + str(valid_executor)
        )

    ctr = getattr(pGB.base, "array_" + dtype)
    executor = getattr(pGB, executor + "Executor")()
    return ctr(executor, obj)


def as_tensor(obj, executor: str = "Reference", dtype="float"):
    """create a ginkgo array from a given object"""
    if not dtype in valid_value_types:
        raise ValueError(
            "Not a valid dtype: "
            + dtype
            + " possible choices are: "
            + str(valid_value_types)
        )
    if not executor in valid_executor:
        raise ValueError(
            "Not a valid executor: "
            + dtype
            + " possible choices are: "
            + str(valid_executor)
        )

    if torch_avail:
        if isinstance(obj, torch.Tensor):
            obj = obj.__array__()

    ctr = getattr(pGB.base, "dense_" + dtype)
    executor = getattr(pGB, executor + "Executor")()
    return ctr(executor, obj)


def solve(A, b, initial_guess=None, solver_args: dict = dict()):
    """Solve a given linear system, where A is the system matrix and b the RHS

    Parameters: A - The system matrix
                b - The right hand side vector
                initial_guess - The initial guess
                solver - The solver
                solver_args - A dictionary that is forwarded to the solver containing
                    arguments, eg {'max_iters': 100, 'tolerance': 1e-6}
    Returns: tuple of a logger object and solution vector
    """

    if not solver_args:
        solver_args = {
            "type": "solver::Gmres",
            "preconditioner": {
                "type": "preconditioner::Ilu",
                "l_solver_type": "solver::LowerTrs",
                "reverse_apply": False,
                "factorization": {"type": "factorization::ParIlu"},
            },
            "criteria": [
                {"type": "Iteration", "max_iters": 1000},
                {"type": "ResidualNorm", "reduction_factor": 1e-7},
            ],
        }
    solver_executor = A.get_executor()

    # TODO: Create a better way to check the dtype of the matrix
    dtype = str(type(A)).split('_')[1]
    dense_ctr = getattr(pGB.matrix, f"dense_{dtype}")

    if not initial_guess:
        initial_guess = dense_ctr(b.get_executor(), (b.dim[0], 1))
        initial_guess.fill(0.0)

    solver_ctr = getattr(pGB.solver, "config_solve_" + dtype)
    logger = solver_ctr(
        solver_executor, A, b, initial_guess, json.dumps(solver_args)
    )
    return logger, initial_guess
