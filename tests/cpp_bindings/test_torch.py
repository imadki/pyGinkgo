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
    arr = pgb.base.array_float(executor, np_array)
    torch_array = torch.asarray(arr)
    assert torch_array.size(dim=0) == np_array.size


@pytest.mark.skipif(not torch_avail, reason="requires pytorch")
def test_can_create_dense_from_torch_tensor():
    executor = pgb.ReferenceExecutor()
    data = [[1.0, 2.0], [3.0, 4.0]]
    torch_tensor = torch.tensor(data, dtype=float)
    dense = pgb.matrix.dense(executor, torch_tensor.__array__())
    assert dense.get_num_stored_elements() == 4
    assert dense.at(0, 1) == 2.0
    assert dense.at(1, 1) == 4.0
    assert dense.get_size()[0] == 2
    assert dense.get_size()[1] == 2
