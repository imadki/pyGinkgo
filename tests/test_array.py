# SPDX - License - Identifier : MIT
#
# SPDX - FileCopyrightText : 2024 pyGinkgo authors


import sys
import pyGinkgo

sys.path.append("../build")


def test_init_empty_array():
    executor = pyGinkgo.ReferenceExecutor()
    size = 5
    arr = pyGinkgo.base.array(executor, size)
    assert arr.get_size() == size
