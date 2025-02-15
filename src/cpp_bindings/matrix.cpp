// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"
#include "utils.hpp"

namespace py = pybind11;

template <template <typename ValueType, typename IndexType> class MatrixType,
          typename ValueType, typename IndexType>
void init_matrix(py::module_ &module, const std::string matrix_type,
                 const std::string value_type, const std::string index_type)
{
    std::string matrix_type_repr = "pygko.matrix." + matrix_type + " object";
    std::string value_index_str = value_type + "_" + index_type;
    std::string pyclass_name = matrix_type + "_" + value_index_str;

    py::class_<MatrixType<ValueType, IndexType>,
               std::shared_ptr<MatrixType<ValueType, IndexType>>, gko::LinOp>(
        module, pyclass_name.c_str(), py::buffer_protocol())
        .def(py::init([](std::shared_ptr<gko::Executor> exec) {
            return MatrixType<ValueType, IndexType>::create(exec);
        }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,
                         gko::array<ValueType> &vals,
                         gko::array<IndexType> &cols,
                         gko::array<IndexType> &rows) {
            return gko::share(MatrixType<ValueType, IndexType>::create(
                exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},
                vals, cols, rows));
        }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,
                         py::buffer data, py::buffer cols, py::buffer rows) {
            /* Request a buffer descriptor from Python */
            py::buffer_info data_info = data.request();
            check_buffer_dtype<ValueType>(data_info);

            py::buffer_info cols_info = cols.request();
            check_buffer_dtype<IndexType>(cols_info);

            py::buffer_info rows_info = rows.request();
            check_buffer_dtype<IndexType>(rows_info);

            auto ref = gko::ReferenceExecutor::create();

            /* create a view into numpy data */
            auto nnz = data_info.shape[0];

            auto data_view = gko::array<ValueType>::view(
                ref, nnz, (ValueType *)data_info.ptr);
            auto cols_view = gko::array<IndexType>::view(
                ref, nnz, (IndexType *)cols_info.ptr);

            auto rows_view = gko::array<IndexType>::view(
                ref, rows_info.shape[0], (IndexType *)rows_info.ptr);

            return gko::share(MatrixType<ValueType, IndexType>::create(
                exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},
                data_view, cols_view, rows_view));
        }))
        .def("__repr__",
             [=](const MatrixType<ValueType, IndexType> &o) {
                 return matrix_type_repr;
             })
        .def("get_num_stored_elements",
             &MatrixType<ValueType, IndexType>::get_num_stored_elements,
             "Get the number of non zero elements")
        .def("get_size", &MatrixType<ValueType, IndexType>::get_size,
             "Get the size of the matrix");

    std::string read_fn = "read_" + matrix_type + "_" + value_index_str;
    module.def(read_fn.c_str(), [](const std::string &fn,
                                   std::shared_ptr<gko::Executor> exec) {
        return gko::share(gko::read<MatrixType<ValueType, IndexType>>(
            std::ifstream(fn), exec));
    });
}

void init_sparse_all_types(py::module_ &module)
{
#define DECLARE_COO_MATRIX(ValueType, IndexType)         \
    init_matrix<gko::matrix::Coo, ValueType, IndexType>( \
        module, "Coo", #ValueType, #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_COO_MATRIX);

#define DECLARE_CSR_MATRIX(ValueType, IndexType)         \
    init_matrix<gko::matrix::Csr, ValueType, IndexType>( \
        module, "Csr", #ValueType, #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_CSR_MATRIX);
}