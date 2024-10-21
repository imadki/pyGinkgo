// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "../python.hpp"

void init_gmres(py::module_ &module_solver)
{
    py::class_<gko::solver::Gmres<ValueType>,
               std::shared_ptr<gko::solver::Gmres<ValueType>>, gko::LinOp>(
        module_solver, "gmres")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix,
                         size_t max_iters, size_t krylov_dim,
                         ValueType reduction_factor, bool relative_stop_mode) {
                 auto stop_mode = (relative_stop_mode)
                                      ? gko::stop::mode::rhs_norm
                                      : gko::stop::mode::absolute;
                 auto fact = gko::share(
                     gko::solver::Gmres<ValueType>::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(
                                 max_iters),
                             gko::stop::ResidualNorm<ValueType>::build()
                                 .with_baseline(stop_mode)
                                 .with_reduction_factor(reduction_factor))
                         .with_krylov_dim(krylov_dim)
                         .on(exec));
                 return gko::share(fact->generate(system_matrix));
             }),
             py::arg("exec"), py::arg("system_matrix"), py::arg("max_iters"),
             py::arg("krylov_dim"), py::arg("reduction_factor"),
             py::arg("relative_stop_mode"))
        .def("initialize_logger",
             [](gko::solver::Gmres<ValueType> &o) {
                 std::shared_ptr<gko::log::Convergence<ValueType>>
                     convergence_logger =
                         gko::log::Convergence<ValueType>::create();
                 o.add_logger(convergence_logger);
                 return convergence_logger;
             })
        .def("__repr__",
             [](const gko::solver::Gmres<ValueType> &o) {
                 auto str = std::string("pygko.solver.Gmres object");
                 return str;
             })
        .def(
            "apply",
            [](const gko::solver::Gmres<ValueType> &d,
               std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> x) { d.apply(b, x); },
            "");
}
