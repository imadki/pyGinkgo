// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"

namespace py = pybind11;

void init_logger(py::module_ &logger_module)
{
    py::class_<gko::log::Convergence<ValueType>,
               std::shared_ptr<gko::log::Convergence<ValueType>>>(
        logger_module, "convergence_logger")
        .def(
            py::init([]() {
                return gko::share(gko::log::Convergence<ValueType>::create());
            }),
            "Creates an empty stream wrapper, representing the default stream.")
        .def("has_converged", &gko::log::Convergence<ValueType>::has_converged,
             "Returns whether the solver has converged")
        .def(
            "get_residual_norm",
            [](gko::log::Convergence<ValueType> &o) {
                auto res_norm = gko::as<gko::matrix::Dense<ValueType>>(
                    o.get_residual_norm());
                auto res_norm_host = gko::matrix::Dense<ValueType>::create(
                    res_norm->get_executor()->get_master(), gko::dim<2>{1});
                res_norm_host->copy_from(res_norm);
                return res_norm_host->at(0);
            },
            "Returns the residual norm")
        .def("get_num_iterations",
             &gko::log::Convergence<ValueType>::get_num_iterations,
             "Returns whether the number of iterations");
}
