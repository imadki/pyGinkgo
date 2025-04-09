// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

// TODO: might require placement in a separate dedicated folder

#include "python.hpp"

namespace py = pybind11;

void add_dpcpp_queue_property_enum(py::module_ &root_module)
{
#ifdef GINKGO_BUILD_SYCL
    py::enum_<dpcpp_queue_property>(root_module, "dpcpp_queue_property")
        .value("in_order", dpcpp_queue_property::in_order)
        .value("enable_profiling", dpcpp_queue_property::enable_profiling);
#endif
}
