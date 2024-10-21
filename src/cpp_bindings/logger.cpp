// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"

namespace py = pybind11;

// /**
//  * Returns the number of iterations
//  *
//  * @return the number of iterations
//  */
// const size_type& get_num_iterations() const noexcept
// {
//     return num_iterations_;
// }

// /**
//  * Returns the residual
//  *
//  * @return the residual
//  */
// const LinOp* get_residual() const noexcept { return residual_.get(); }

// /**
//  * Returns the residual norm
//  *
//  * @return the residual norm
//  */
// const LinOp* get_residual_norm() const noexcept
// {
//     return residual_norm_.get();
// }

// /**
//  * Returns the implicit squared residual norm
//  *
//  * @return the implicit squared residual norm
//  */
// const LinOp* get_implicit_sq_resnorm() const noexcept
// {
//     return implicit_sq_resnorm_.get();
// }

void init_logger(py::module_ &logger_module)
{
    py::class_<gko::log::Convergence<ValueType>,
               std::shared_ptr<gko::log::Convergence<ValueType>>>(
        logger_module, "convergence_logger")
        // The comments are copied from the source of the class at
        // ginkgo-src/include/ginkgo/core/base/stream.hpp
        .def(
            py::init([]() {
                return gko::share(gko::log::Convergence<ValueType>::create());
            }),
            "Creates an empty stream wrapper, representing the default stream.")
        .def("has_converged", &gko::log::Convergence<ValueType>::has_converged,
             "Returns whether the solver has converged")
        .def("get_num_iterations",
             &gko::log::Convergence<ValueType>::get_num_iterations,
             "Returns whether the number of iterations");
}
