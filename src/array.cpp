// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"

namespace py = pybind11;

void init_array(py::module_ &module_base) {
  py::class_<gko::array<ValueType>>(module_base, "array")
      .def(py::init<std::shared_ptr<const gko::Executor>, int>())
      .def("fill", &gko::array<ValueType>::fill,
           "Fill the array with the given value.")
      .def("get_size", &gko::array<ValueType>::get_num_elems);
}
