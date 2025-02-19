# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import numpy as np
from typing import Union


def verify_dense_vec(mtx, values: Union[list, np.ndarray]):
    """ """
    assert mtx.get_num_stored_elements() == len(values)
    for i in range(len(values)):
        assert mtx.at(i) == values[i]

    assert mtx.at(2, 0) == mtx.at(2)
    # test if it can be called several times
    assert mtx.at(2, 0) == mtx.at(2)


def verify_within_precision(a, b, precision=1e-3):
    assert abs(a - b) <= precision


d_type_map = {
    "half": np.float16,
    "float": np.float32,
    "double": np.float64,
}

i_type_map = {
    "int32": np.int32,
    "int64": np.longlong,
}

di_type_map = {**d_type_map, **i_type_map}
