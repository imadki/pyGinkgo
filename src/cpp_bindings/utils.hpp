#pragma once

#include <ginkgo/core/base/types.hpp>
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

// cuda half operation is supported from arch 5.3
#if GINKGO_ENABLE_HALF && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
#define PYGKO_ADAPT_HF(_macro) _macro
#else
#define PYGKO_ADAPT_HF(_macro)
#endif

#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro) \
    PYGKO_ADAPT_HF(_macro(half));                                 \
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(_macro)

/**
 * Instantiates a template for each index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define PYGKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(_macro) \
    _macro(int32);                                    \
    _macro(int64)
