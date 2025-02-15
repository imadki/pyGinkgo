// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#pragma once

#include <pybind11/pybind11.h>
#include <ginkgo/ginkgo.hpp>

namespace py = pybind11;

using ValueType = double;
using IndexType = int;

using half = gko::half;

// Declaration of half-precision type for the numpy
// Used solution from
// https://github.com/pybind/pybind11/issues/1776#issuecomment-492742167
// Another option:
// https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/inference_api.cc#L67-L85
namespace PYBIND11_NAMESPACE {
template <>
struct format_descriptor<half> {
    // https://docs.python.org/3/library/struct.html#:~:text=(3)-,e
    static std::string format() { return "e"; }
};
}  // namespace PYBIND11_NAMESPACE
