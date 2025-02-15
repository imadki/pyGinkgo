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

d_type_map = {
    "half": np.float16,
    "float": np.float32,
    "double": np.float64,
}

torch_d_type_map = {
    "half": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
}


@pytest.mark.skipif(not torch_avail, reason="requires pytorch")
@pytest.mark.parametrize("data_type", list(d_type_map.keys()))
class TestTorchInteroperability:
    def test_can_create_array_from_torch(self, data_type):
        executor = pgb.ReferenceExecutor()
        ctr = getattr(pgb.base, "array_" + data_type)
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=d_type_map[data_type])
        torch_array = torch.asarray(np_array)
        arr = ctr(executor, torch_array)
        arr_copy = ctr(executor, arr)
        assert arr.get_size() == arr_copy.get_size()
        assert pgb.base.reduce_add(arr, 0.0) == 15.0

    def test_can_create_torch_array_from_gko_array(self, data_type):
        executor = pgb.ReferenceExecutor()
        ctr = getattr(pgb.base, "array_" + data_type)
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=d_type_map[data_type])
        arr = ctr(executor, np_array)
        torch_array = torch.asarray(arr)
        assert torch_array.size(dim=0) == np_array.size

    def test_can_create_dense_from_torch_tensor(self, data_type):
        executor = pgb.ReferenceExecutor()
        ctr = getattr(pgb.matrix, "dense_" + data_type)
        data = [[1.0, 2.0], [3.0, 4.0]]
        # TODO: the created tensor is broken for data_type = "half" and "double"
        torch_tensor = torch.tensor(data, dtype=torch_d_type_map[data_type])
        dense = ctr(executor, torch_tensor.__array__())
        assert dense.get_num_stored_elements() == 4
        assert dense.at(0, 1) == 2.0
        assert dense.at(1, 1) == 4.0
        assert dense.get_size()[0] == 2
        assert dense.get_size()[1] == 2
