# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

add_test(
  NAME pyginkgo_bindings_import_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_import.py
  WORKING_DIRECTORY $<TARGET_FILE_DIR:pyGinkgoBindings>)
add_test(
  NAME pyginkgo_bindings_array_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_array.py
  WORKING_DIRECTORY $<TARGET_FILE_DIR:pyGinkgoBindings>)
add_test(
  NAME pyginkgo_bindings_dense_matrix_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_dense.py
  WORKING_DIRECTORY $<TARGET_FILE_DIR:pyGinkgoBindings>)
add_test(
  NAME pyginkgo_bindings_sparse_matrix_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse.py
  WORKING_DIRECTORY $<TARGET_FILE_DIR:pyGinkgoBindings>)
add_test(
  NAME pyginkgo_bindings_solver_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_solver.py
  WORKING_DIRECTORY $<TARGET_FILE_DIR:pyGinkgoBindings>)

# add CUDA specific tests, if building with CUDA enabled
if(GINKGO_BUILD_CUDA)
  add_test(
    NAME pyginkgo_bindings_cuda_executor_test
    COMMAND ${PYTHON_EXECUTABLE} -m pytest
            ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_executor.py
    WORKING_DIRECTORY $<TARGET_FILE_DIR:pyGinkgoBindings>)
endif()