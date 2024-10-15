# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

from pyGinkgo import pyGinkgoBindings as pGB
import numpy as np


def asarray(obj, executor: str = "Reference", dtype="float"):
    """create a ginkgo array from a given object"""
    valid_dtypes = ["int", "float", "double"]
    if not dtype in valid_dtypes:
        raise ValueError(
            "Not a valid dtype: " + dtype + " possible choices are: " + valid_dtypes
        )
    valid_executor = ["Reference", "Cuda"]
    if not executor in valid_executor:
        raise ValueError(
            "Not a valid executor: "
            + dtype
            + " possible choices are: "
            + valid_executor
        )

    ctr = getattr(pGB.base, "array_" + dtype)
    executor = getattr(pGB, executor + "Executor")()
    return ctr(executor, obj)
