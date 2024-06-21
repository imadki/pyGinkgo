# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import sys
import pyGinkgo
import numpy as np

sys.path.append("../build")


def test_init_empty_coo():
    executor = pyGinkgo.ReferenceExecutor()
    dim = pyGinkgo.dim(5,5)
    mtx = pyGinkgo.matrix.coo(executor, dim)

def test_read_coo_from_mtx():
    executor = pyGinkgo.ReferenceExecutor()
    dim = pyGinkgo.dim(5,5)
    mtx = pyGinkgo.matrix.mmread("test.mtx")
    assert mtx.get_num_stored_elments() != 0
