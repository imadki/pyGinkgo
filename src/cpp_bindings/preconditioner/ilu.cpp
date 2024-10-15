// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "../python.hpp"

using Ilu =
    gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
                             gko::solver::UpperTrs<ValueType, IndexType>,
                             false>;

void init_ilu(py::module_ &module_preconditioner)
void init_ilu(py::module_ &module_solver)
{
    py::class_<gko::solver::Ilu, std::shared_ptr<gko::preconditioner::Ilu>,
               gko::LinOp>(module_solver, "Ilu")
{
    py::class_<Ilu, std::shared_ptr<Ilu>, gko::LinOp>(module_preconditioner,
                                                      "Ilu")
            auto ilu_pre_factory =
                gko::preconditioner::Ilu<
                    gko::solver::LowerTrs<ValueType, IndexType>,
                    gko::solver::UpperTrs<ValueType, IndexType>, false>::build()
                    .on(exec);
            auto ilu_pre_factory = Ilu::build().on(exec);

            // Use incomplete factors to generate ILU preconditioner
            return gko::share(ilu_pre_factory->generate(par_ilu));
        }))
        .def("__repr__", [](const gko::solver::Gmres<ValueType> &o) {
            auto str = std::string("pygko.preconditioner.Ilu object");
            return str;
        });
}
