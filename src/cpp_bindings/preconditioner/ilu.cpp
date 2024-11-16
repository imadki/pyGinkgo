// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "../python.hpp"

using Ilu =
    gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
                             gko::solver::UpperTrs<ValueType, IndexType>,
                             false>;

void init_ilu(py::module_ &module_preconditioner)
{
    py::class_<Ilu, std::shared_ptr<Ilu>, gko::LinOp>(module_preconditioner,
                                                      "Ilu")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix) {
            auto par_ilu_fact =
                gko::factorization::ParIlu<ValueType, IndexType>::build().on(
                    exec);
            // Generate concrete factorization for input matrix
            auto par_ilu = gko::share(par_ilu_fact->generate(system_matrix));

            // Generate an ILU preconditioner factory by setting lower and upper
            // triangular solver - in this case the exact triangular solves
            auto ilu_pre_factory = Ilu::build().on(exec);

            // Use incomplete factors to generate ILU preconditioner
            return gko::share(ilu_pre_factory->generate(par_ilu));
        }))
        .def("__repr__", [](const gko::solver::Gmres<ValueType> &o) {
            auto str = std::string("pygko.preconditioner.Ilu object");
            return str;
        });
}
