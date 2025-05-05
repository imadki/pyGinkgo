from enum import StrEnum, auto

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
    