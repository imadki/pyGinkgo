// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <ginkgo/ginkgo.hpp>

namespace py = pybind11;

using ValueType = double;
using IndexType = int;

using half = gko::half;
using int32 = gko::int32;
using int64 = gko::int64;

// Declaration of half-precision type for the numpy
// Used solution from:
// https://github.com/PaddlePaddle/Paddle/blob/d2a6b77dd90fa749cf2602378b3b506d628f1061/paddle/fluid/pybind/inference_api.cc#L59-L85
namespace PYBIND11_NAMESPACE {
namespace detail {
// Note: use same enum number of float16 in numpy.
// import numpy as np
// print np.dtype(np.float16).num  # 23
constexpr int NPY_FLOAT16_ = 23;

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
        // https://docs.python.org/3/library/struct.html#:~:text=(3)-,e
        return "e";
    }
    static constexpr auto name = _("float16");
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
