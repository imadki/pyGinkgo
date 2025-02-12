#pragma once

#include <pybind11/numpy.h>
#include <ginkgo/core/base/types.hpp>
#include "half.hpp"
#include "python.hpp"

template <typename ValueType>
void check_buffer_dtype(const py::buffer_info &info)
{
    if (info.format != py::format_descriptor<ValueType>::format())
        throw std::runtime_error("Incompatible dtypes: " + info.format +
                                 " vs " +
                                 py::format_descriptor<ValueType>::format());
}

/**
 * Instantiates a template for each non-complex value type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(_macro) \
    _macro(float)
#else
#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(_macro) \
    _macro(float);                                                     \
    _macro(double)
#endif

#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro) \
    _macro(half);                                                 \
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(_macro)