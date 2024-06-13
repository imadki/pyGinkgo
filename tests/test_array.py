# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

# tests/test_array.py

import sys
import pyGinkgo

sys.path.append("../build")


def test_init_empty_array():
    executor = pyGinkgo.Executor()
    size = 5
    arr = pyGinkgo.base.array(executor, size)
    assert arr.get_size() == size
