// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include <pybind11/pybind11.h>
#include "pybind11/numpy.h"

#include "python.hpp"

#include "ginkgo/ginkgo.hpp"
#include <ginkgo/extensions/config/json_config.hpp>

namespace py = pybind11;

gko::log::Convergence<ValueType> config_solve(
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<gko::LinOp> A,
    std::shared_ptr<gko::LinOp> b,
    std::shared_ptr<gko::LinOp> x,
    std::string configfile){
    auto config = gko::ext::config::parse_json_file(configfile);
    // Create the registry, which allows passing the existing data into config
    // This example does not use existing data.
    auto reg = gko::config::registry();
    // Create the default type descriptor, which gives the default common type
    // (value/index) for solver generation. If the solver does not specify value
    // type, the solver will use these types.
    auto td = gko::config::make_type_descriptor<ValueType, IndexType>();
    // generate the linopfactory on the given executors
    auto solver_gen = gko::config::parse(config, reg, td).on(exec);

    // Create solver
    auto solver = solver_gen->generate(A);
    solver->apply(b, x);

    // Add logger
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(logger);
    return logger;
}

template <typename ValueType>
void init_file_config(py::module_ &m)
{
        m.def("config_solve", &config_solve, "wrapper for config solve");
}
