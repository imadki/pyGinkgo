// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include <pybind11/pybind11.h>
#include "pybind11/numpy.h"

#include "../python.hpp"
#include "../utils.hpp"

#include <ginkgo/extensions/config/json_config.hpp>
#include "ginkgo/ginkgo.hpp"

namespace py = pybind11;


std::shared_ptr<gko::LinOp> config_solver(std::shared_ptr<gko::Executor> exec,
                                          std::shared_ptr<gko::LinOp> A,
                                          std::string json)
{
    auto config = gko::ext::config::parse_json(nlohmann::json::parse(json));
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

    return solver;
}

template <typename ValueType>
std::shared_ptr<const gko::log::Convergence<ValueType>> config_solve(
    std::shared_ptr<gko::Executor> exec, std::shared_ptr<gko::LinOp> A,
    std::shared_ptr<gko::LinOp> b, std::shared_ptr<gko::LinOp> x,
    std::string json)
{
    // Create solver
    auto solver = config_solver(exec, A, json);

    // Add logger
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(logger);

    solver->apply(b, x);

    return logger;
}

template <typename ValueType>
void init_config_solver(py::module_ &m, const std::string value_type)
{
    std::string pyfunc_name = "config_solve_" + value_type;
    m.def(pyfunc_name.c_str(), &config_solve<ValueType>,
          "wrapper for config solve");
}

void init_config_solver_all_types(py::module_ &m)
{
#define DECLARE_CONFIG_SOLVER(ValueType) \
    init_config_solver<ValueType>(m, #ValueType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_CONFIG_SOLVER);
#undef DECLARE_CONFIG_SOLVER

    m.def("config_solver", &config_solver,
          "wrapper for generating config solver");
}
