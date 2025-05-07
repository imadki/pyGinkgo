// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include <pybind11/numpy.h>
#include "python.hpp"
#include "utils.hpp"

template <typename ValueType>
void init_array(py::module_ &module, const std::string typestr)
{
    std::string pyclass_name = std::string("array_") + typestr;

    py::class_<gko::array<ValueType>>(module, pyclass_name.c_str(),
                                      py::buffer_protocol())
        .def(py::init<std::shared_ptr<const gko::Executor>, int>())
        .def(py::init<std::shared_ptr<const gko::Executor>,
                      gko::array<ValueType> &>())
        .def(py::init(
            [](std::shared_ptr<gko::Executor> exec,
               py::array_t<ValueType, py::array::c_style | py::array::forcecast>
                   b) {
                // for documentation of the second argument see
                // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#arrays
                /* Request a buffer descriptor from Python */
                py::buffer_info info = b.request();
                check_buffer_dtype<ValueType>(info);

                if (info.ndim != 1) {
                    throw std::runtime_error("Only 1D arrays are supported");
                }

                auto elems = info.shape[0];
                return gko::array<ValueType>(exec, (ValueType *)info.ptr,
                                             (ValueType *)info.ptr + elems);
            }))
        .def_buffer([](gko::array<ValueType> &a) -> py::buffer_info {
            return py::buffer_info(
                a.get_data(),      /* Pointer to buffer */
                sizeof(ValueType), /* Size of one scalar */
                py::format_descriptor<
                    ValueType>::format(), /* Python struct-style
                                             format descriptor */
                1,                        /* Number of dimensions */
                {a.get_size()},           /* Buffer dimensions */
                {
                    sizeof(ValueType) /* Strides (in bytes) for each index */
                });
        })
        .def("fill", &gko::array<ValueType>::fill,
             "Fill the array with the given value.")
        .def("get_size", &gko::array<ValueType>::get_size)
        .def_property_readonly("size", &gko::array<ValueType>::get_size)
        .def("at", [](const gko::array<ValueType> &arr,
                      int idx) { return arr.get_const_data()[idx]; })
        .def("__repr__", [=](const gko::array<ValueType> &arr) {
            auto str = "pygko.base." + pyclass_name + " object of size ";
            auto elems = arr.get_size();
            str += std::to_string(elems);
            if (arr.get_executor() == arr.get_executor()->get_master()) {
                str += " on host";
                if (elems < 10) {
                    str += " [ ";
                    auto data = arr.get_const_data();
                    for (int i = 0; i < elems; i++) {
                        str += std::to_string(data[i]);
                        str += " ";
                    }
                    str += " ] ";
                }
            }
            return str;
        });

    module.def(
        "reduce_add",
        pybind11::overload_cast<const gko::array<ValueType> &, const ValueType>(
            &gko::reduce_add<ValueType>));
}

void init_array_all_types(py::module_ &module)
{
#define DECLARE_ARRAY_VALUE(ValueType) \
    init_array<ValueType>(module, #ValueType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_ARRAY_VALUE);
    PYGKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(DECLARE_ARRAY_VALUE);
}
