# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import sys
import numpy as np
import torch

sys.path.append("../build")
import pyGinkgoBindings as pGB


def test_can_create_array_from_torch():
    executor = pGB.ReferenceExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    torch_array = torch.asarray(np_array)
    arr = pGB.base.array_float(executor, torch_array)
    arr_copy = pGB.base.array_float(executor, arr)
    assert arr.get_size() == arr_copy.get_size()
    assert pGB.base.reduce_add(arr, 0.0) == 15.0


def test_can_create_torch_array_from_gko_array():
    executor = pGB.ReferenceExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr = pGB.base.array_float(executor, torch_array)
    torch_array = torch.asarray(arr)
