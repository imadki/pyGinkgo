from enum import StrEnum, auto

from . import pyGinkgoBindings as pGB

class ValueType(StrEnum):
    half = auto()
    float = auto()
    double = auto()

class IndexType(StrEnum):
    int32 = auto()
    int64 = auto()

dtype = {*ValueType, *IndexType}

class MatrixFormat(StrEnum):
    dense = auto()
    # TODO: it makes sense to make them lower case
    Csr = auto()
    Coo = auto()
    

class ExecutorType(StrEnum):
    cpu = auto()
    omp = auto()
    cuda = auto()
    hip = auto()
    dpcpp = auto()

DeviceType = ExecutorType | pGB.Executor | str
