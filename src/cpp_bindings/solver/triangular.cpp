// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "../python.hpp"
#include "../utils.hpp"

template <typename ValueType, typename IndexType>
void init_UpperTrs(py::module_ &module_solver, const std::string value_type,
                   const std::string index_type)
{
    std::string pyclass_name = "UpperTrs_" + value_type + "_" + index_type;
    std::string repr_str = "pygko.solver." + pyclass_name + " object";
    py::class_<gko::solver::UpperTrs<ValueType, IndexType>,
               std::shared_ptr<gko::solver::UpperTrs<ValueType, IndexType>>,
               gko::LinOp>(module_solver, pyclass_name.c_str())
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix) {
                 auto solver_fact = gko::share(
                     gko::solver::UpperTrs<ValueType, IndexType>::build().on(
                         exec));
                 return gko::share(solver_fact->generate(system_matrix));
             }),
             py::arg("exec"), py::arg("system_matrix"))
        .def("__repr__",
             [=](const gko::solver::UpperTrs<ValueType, IndexType> &o) {
                 return repr_str;
             })
        .def(
            "apply",
            [](const gko::solver::UpperTrs<ValueType, IndexType> &d,
               std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> x) { d.apply(b, x); },
            "");
}

void init_UpperTrs_all_types(py::module_ &module_solver)
{
#define DECLARE_TRS_SOLVER(ValueType, IndexType) \
    init_UpperTrs<ValueType, IndexType>(module_solver, #ValueType, #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_TRS_SOLVER);
}

template <typename ValueType, typename IndexType>
void init_LowerTrs(py::module_ &module_solver, const std::string value_type,
                   const std::string index_type)
{
    std::string pyclass_name = "LowerTrs_" + value_type + "_" + index_type;
    std::string repr_str = "pygko.solver." + pyclass_name + " object";
    py::class_<gko::solver::LowerTrs<ValueType, IndexType>,
               std::shared_ptr<gko::solver::LowerTrs<ValueType, IndexType>>,
               gko::LinOp>(module_solver, pyclass_name.c_str())
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix) {
                 auto solver_fact = gko::share(
                     gko::solver::LowerTrs<ValueType, IndexType>::build().on(
                         exec));
                 return gko::share(solver_fact->generate(system_matrix));
             }),
             py::arg("exec"), py::arg("system_matrix"))
        .def("__repr__",
             [=](const gko::solver::LowerTrs<ValueType, IndexType> &o) {
                 return repr_str;
             })
        .def(
            "apply",
            [](const gko::solver::LowerTrs<ValueType, IndexType> &d,
               std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> x) { d.apply(b, x); },
            "");
}

void init_LowerTrs_all_types(py::module_ &module_solver)
{
#define DECLARE_TRS_SOLVER(ValueType, IndexType) \
    init_LowerTrs<ValueType, IndexType>(module_solver, #ValueType, #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_TRS_SOLVER);
}
