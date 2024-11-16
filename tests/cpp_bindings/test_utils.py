# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import numpy as np
from typing import Union

import pyGinkgo.pyGinkgoBindings as pGB


def verify_dense_vec(mtx: pGB.matrix.dense, values: Union[list, np.ndarray]):
    """ """
    assert mtx.get_num_stored_elements() == len(values)
    for i in range(len(values)):
        assert mtx.at(i) == values[i]

    assert mtx.at(2, 0) == mtx.at(2)
    # test if it can be called several times
    assert mtx.at(2, 0) == mtx.at(2)
