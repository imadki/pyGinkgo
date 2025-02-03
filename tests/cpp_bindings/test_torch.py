# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import numpy as np

try:
    import torch

    torch_avail = True
except ImportError:
    torch_avail = False

import pyGinkgo.pyGinkgoBindings as pgb
import pytest


@pytest.mark.skipif(not torch_avail, reason="requires pytorch")
def test_can_create_array_from_torch():
    executor = pgb.ReferenceExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    torch_array = torch.asarray(np_array)
    arr = pgb.base.array_float(executor, torch_array)
    arr_copy = pgb.base.array_float(executor, arr)
    assert arr.get_size() == arr_copy.get_size()
    assert pgb.base.reduce_add(arr, 0.0) == 15.0


@pytest.mark.skipif(not torch_avail, reason="requires pytorch")
def test_can_create_torch_array_from_gko_array():
    executor = pgb.ReferenceExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr = pgb.base.array_float(executor, torch_array)
    torch_array = torch.asarray(arr)


@pytest.mark.skipif(not torch_avail, reason="requires pytorch")
def test_can_create_dense_from_torch_tensor():
    executor = pgb.ReferenceExecutor()

    data = [[1, 2], [3, 4]]
    torch_tensor = torch.tensor(data)

    arr = pgb.matrix.dense(executor, torch_tensor)
