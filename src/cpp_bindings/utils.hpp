#pragma once

#include <pybind11/numpy.h>
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

using half = gko::half;
// https://github.com/PaddlePaddle/Paddle/blob/fb0a9a5c3b22e02143fcb3cd568237b27e6cb825/paddle/fluid/pybind/inference_api.cc#L67-L85
namespace pybind11::detail {

// Note: use same enum number of float16 in numpy.
// import numpy as np
// print np.dtype(np.float16).num  # 23
constexpr int NPY_FLOAT16_ = 23;
constexpr int NPY_UINT16_ = 4;

// Note: Since float16 is not a builtin type in C++, we register
// phi::dtype::float16 as numpy.float16.
// Ref: https://github.com/pybind/pybind11/issues/1776
template <>
struct npy_format_descriptor<half> {
    static py::dtype dtype()
    {
        handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16_);
        return reinterpret_borrow<py::dtype>(ptr);
    }
    static std::string format()
    {
        // Note: "e" represents float16.
        // Details at:
        // https://docs.python.org/3/library/struct.html#format-characters.
        return "e";
    }
    static constexpr auto name = _("float16");
};

}  // namespace pybind11::detail

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
    GKO_ADAPT_HF(_macro(half));                                   \
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(_macro)