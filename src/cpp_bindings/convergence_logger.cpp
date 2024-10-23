// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

// TODO: might require placement in a separate dedicated folder

#include "python.hpp"

namespace py = pybind11;


bool has_converged() const noexcept { return convergence_status_; }


/**
 * Returns the number of iterations
 *
 * @return the number of iterations
 */
const size_type &get_num_iterations() const noexcept { return num_iterations_; }

/**
 * Returns the residual
 *
 * @return the residual
 */
const LinOp *get_residual() const noexcept { return residual_.get(); }

/**
 * Returns the residual norm
 *
 * @return the residual norm
 */
const LinOp *get_residual_norm() const noexcept { return residual_norm_.get(); }

/**
 * Returns the implicit squared residual norm
 *
 * @return the implicit squared residual norm
 */
const LinOp *get_implicit_sq_resnorm() const noexcept
{
    return implicit_sq_resnorm_.get();
}

void init_logger(py::module_ &logger_module)
{
    py::class_<gko::convergence_logger, std::shared_ptr<gko::cuda_stream>>(
        logger_module, "cuda_stream")
        // The comments are copied from the source of the class at
        // ginkgo-src/include/ginkgo/core/base/stream.hpp
        .def(
            py::init(),
            "Creates an empty stream wrapper, representing the default stream.")
        .def("get_num_stored_elements",
             &gko::matrix::Dense<ValueType>::get_num_stored_elements,
             "Returns the number of elements explicitly stored in the "
             "matrix.");
}
