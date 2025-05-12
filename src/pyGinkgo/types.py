# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2025 pyGinkgo authors

import numpy as np
from typing import TypeAlias
from enum import ReprEnum, auto

from . import pyGinkgoBindings as pGB


class StrEnum(str, ReprEnum):
    """
    Enum where members are also (and must be) strings
    """

    def __new__(cls, value: str):
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        """
        Return the lower-cased version of the member name.
        """
        return name


class DatatypeEnum(StrEnum):
    def __new__(cls, name: str, _):
        member = str.__new__(cls, name)
        member._value_ = name
        return member
    
    def __init__(self, _: str, numpy_type):
        self.numpy_type = numpy_type

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]
    
    @classmethod
    def numpy_types(cls) -> list:
        return [member.numpy_type for member in cls]

    @classmethod
    def type_map(cls) -> dict:
        return {member.value: member.numpy_type for member in cls}


class ValueType(DatatypeEnum):
    half = (auto(), np.float16)
    float = (auto(), np.float32)
    double = (auto(), np.float64)

class IndexType(DatatypeEnum):
    int32 = (auto(), np.int32)
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.longlong
    int64 = (auto(), np.longlong)

dtype: list[DatatypeEnum] = [*ValueType, *IndexType]

class MatrixFormat(StrEnum):
    dense = auto()
    # TODO: it makes sense to make them the same casing
    Csr = "Csr"
    Coo = "Coo"
    

class ExecutorType(StrEnum):
    cpu = auto()
    omp = auto()
    cuda = auto()
    hip = auto()
    dpcpp = auto()

DeviceType = ExecutorType | pGB.Executor | str
