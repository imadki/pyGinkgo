# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

add_test(NAME pyginkgo_import_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_import.py)
add_test(NAME pyginkgo_array_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_array.py)
add_test(NAME pyginkgo_dense_matrix_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_dense.py)
add_test(NAME pyginkgo_sparse_matrix_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse.py)
add_test(NAME pyginkgo_solver_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_solver.py)
