// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "pybind11/numpy.h"
#include "python.hpp"

namespace py = pybind11;

void init_array(py::module_ &module_base) {
  py::class_<gko::array<ValueType>>(module_base, "array")
      .def(py::init<std::shared_ptr<const gko::Executor>, int>())
      .def(py::init<std::shared_ptr<const gko::Executor>, gko::array<ValueType>&>())
      .def(py::init(
          [](std::shared_ptr<gko::Executor> exec,
             py::array_t<double, py::array::c_style | py::array::forcecast> b) {
            // for documentation of the second argument see
            // see
            // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#arrays
            /* Request a buffer descriptor from Python */
            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<ValueType>::format()) {
              throw std::runtime_error("Incompatible dtype");
            }

            auto ref = gko::ReferenceExecutor::create();

            if (info.ndim != 1) {
              throw std::runtime_error("Only 1D arrays are supported");
            }

            auto elems = info.shape[0];

            return gko::array<ValueType>(exec, (ValueType *)info.ptr,
                                         (ValueType *)info.ptr + elems);
          }))
      .def("fill", &gko::array<ValueType>::fill,
           "Fill the array with the given value.")
      .def("get_size", &gko::array<ValueType>::get_num_elems)
      .def("at", [](const gko::array<ValueType> &arr, int idx) {
        return arr.get_const_data()[idx];
      });
}
