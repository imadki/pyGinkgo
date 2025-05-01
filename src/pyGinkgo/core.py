# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

from pyGinkgo import pyGinkgoBindings as pGB
import json
import numpy as np
import pyGinkgo as pg

try:
    import torch

    torch_avail = True
except ImportError:
    torch_avail = False

valid_value_types = ["half", "float", "double"]
valid_index_types = ["int32", "int64"]
valid_dtypes = valid_value_types + valid_index_types


def as_array(obj, device: str | pGB.Executor = "cpu", dtype="float"):
    """create a ginkgo array from a given object"""
    if not dtype in valid_dtypes:
        raise ValueError(
            "Not a valid dtype: "
            + dtype
            + " possible choices are: "
            + str(valid_dtypes)
        )
    if isinstance(device, str):
        executor = pg.device(device)
    else:
        executor = device
    
    array_cls = getattr(pGB.base, "array_" + dtype)
    return array_cls(executor, obj)


def as_tensor(obj=None, dim=None, device: str | pGB.Executor = "cpu", dtype="float"):
    """create a ginkgo array from a given object"""
    if not dtype in valid_value_types:
        raise ValueError(
            "Not a valid dtype: "
            + dtype
            + " possible choices are: "
            + str(valid_value_types)
        )
    if isinstance(device, str):
        executor = pg.device(device)
    else:
        executor = device

    if torch_avail:
        if isinstance(obj, torch.Tensor):
            obj = obj.__array__()

    array_cls = getattr(pGB.matrix, "dense_" + dtype)
    if obj:
        return array_cls(executor, obj)
    else:
        return array_cls(executor, dim)


def factor(A, kind="Upper", device: str | pGB.Executor = "cpu"):
    if isinstance(device, str):
        executor = pg.device(device)
    else:
        executor = device
    
    factorization = pGB.factorization.factorization(executor, A)
    if kind == "Upper":
        return factorization.get_upper_factor()
    if kind == "Lower":
        return factorization.get_lower_factor()


def eigen_solve(A,solver_args=None):
    exec_obj = A.get_executor()
    torchA = torch.as_tensor(np.array(A))
    dtype = str(type(A)).split('_')[1]
    L, Q = torch.linalg.eigh(torchA)
    dense_cls = getattr(pGB.matrix, f"dense_float")
    Lambda = dense_cls(exec_obj, L.__array__())
    hY = dense_cls(exec_obj, Q.__array__())
    return Lambda, hY

def generate_solver(A, solver_args: dict = dict()):
    """Generate a solver based on the system matrix A

    Parameters: A - The system matrix
                solver_args - A dictionary that is forwarded to the solver containing
                    arguments, eg {'type': 'solver::Cg', 'criteria': {'max_iters': 100}}
    Returns: the solver
    """

    if not solver_args:
        solver_args = {
            "type": "solver::Gmres",
            "preconditioner": {
                "type": "preconditioner::Ilu",
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
    solver_cls = getattr(pGB.solver, "config_solver_" + dtype)
    solver = solver_cls(
        solver_executor, A, json.dumps(solver_args)
    )
    return solver

def config_solve(A,b,x,solver_args=None):
    if not solver_args:
        solver_args = {
            "type": "solver::Gmres",
            "preconditioner": {
                "type": "preconditioner::Ilu",
                "reverse_apply": False,
                "factorization": {"type": "factorization::ParIlu"},
            },
            "criteria": [
                {"type": "Iteration", "max_iters": 1000},
                {"type": "ResidualNorm", "reduction_factor": 1e-7},
            ],
        }

    solver_executor = A.get_executor()
    dtype = str(type(A)).split('_')[1]
    # TODO: Create a better way to check the dtype of the matrix
    solver_cls = getattr(pGB.solver, "config_solve_" + dtype)
    logger = solver_cls(
        solver_executor, A, b, x, json.dumps(solver_args)
    )

    return logger, x

def triangular_solve(A,b,x,solver_args):
    kind = solver_args["type"]
    dtype = str(type(A)).split('_')[1]
    itype = str(type(A)).split('_')[2]
    s = f"{kind}Trs_{dtype}_int32" # TODO why does it fail for + itype
    ctor = getattr(pGB.solver, s)
    exec_obj = A.get_executor()
    trs = ctor(exec_obj, A)
    trs.apply(b, x)
    return None, x

def solve(A, b, initial_guess=None, solver_args: dict = dict(), kind="config"):
    """Solve a given linear system, where A is the system matrix and b the RHS

    Parameters: A - The system matrix
                b - The right hand side vector
                initial_guess - The initial guess
                solver - The solver
                solver_args - A dictionary that is forwarded to the solver containing
                    arguments, eg {'type': 'solver::Cg', 'criteria': {'max_iters': 100}}
                kind - the underlying solver, eg. config
    Returns: tuple of a logger object and solution vector
    """

    ctor = globals()[kind+"_solve"]

    if not initial_guess:
        dtype = str(type(A)).split('_')[1]
        dense_cls = getattr(pGB.matrix, f"dense_{dtype}")
        dim = (A.get_size()[1], b.get_size()[1])
        initial_guess = dense_cls(b.get_executor(), dim)
        initial_guess.fill(0.0)

    return ctor(A,b,x=initial_guess,solver_args=solver_args)
