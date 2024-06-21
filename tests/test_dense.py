import sys
import pyGinkgo
import numpy as np

sys.path.append("../build")

values = [1, 2, -1, 3, 4, -1, 5, 6, -1]
array_length = 10


def test_can_create_dense_linop():
    ref = pyGinkgo.ReferenceExecutor()
    dense = pyGinkgo.matrix.dense(self.ref, length)

def test_can_create_dense_from_array():
    arr = pyGinkgo.array(ref, 9, np.array([self.values]))
    dense = pyGinkgo.matrix.Dense(ref, (3, 2), arr, 3)

    assert dense[2] == 5
    assert dense[2, 1] == 6

def test_can_create_dense_from_list():
    dense = pyGinkgo.matrix.Dense(ref, (3, 2), values, 3)

    assert dense.at(2, 1) == 6

def test_can_create_dense_from_nparray():
    dense = pyGinkgo.matrix.Dense(ref, (3, 2), np.array(values), 3)

    assert dense.at(2, 1) == 6

def test_dense_support_basic_functionality():
    np_array = np.array(values)
    dense = pyGinkgo.matrix.Dense(ref, (9, 1), np_array, 1)

    result = pyGinkgo.matrix.Dense(ref, (1, 1))

    dense.compute_norm1(result)
    assert result[0, 0] == np.linalg.norm(np_array)

    dense.compute_norm2(result)
    assert result[0, 0] == np.linalg.norm(np_array)

    # TODO for now just test if callable add checking of results
    dense.scale(result)
    dense.inv_scale(result)
    dense.add_scale(result)
    dense.sub_scale(result)
    dense.compute_dot(dense, result)
