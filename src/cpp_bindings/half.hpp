#pragma once

#include <pybind11/pybind11.h>
#include <ginkgo/core/base/types.hpp>

using half = gko::half;

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

// https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html
namespace PYBIND11_NAMESPACE {
namespace detail {
template <>
struct type_caster<half> {
public:
    /**
     * This macro establishes the name 'half' in
     * function signatures and declares a local variable
     * 'value' of type half
     */
    PYBIND11_TYPE_CASTER(half, const_name("half"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a half
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool)
    {
        /* Extract PyObject from handle */
        PyObject *source = src.ptr();

        /* Try converting into a Python float value */
        PyObject *tmp = PyNumber_Float(source);
        if (!tmp) return false;

        // TODO: fix double conversion on ginkgo side
        value = (float)PyFloat_AsDouble(source);
        Py_DECREF(tmp);

        /* Ensure return code was OK (to avoid out-of-range errors etc) */
        return !(value == -1 && !PyErr_Occurred());
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an half instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(half src, return_value_policy /* policy */,
                       handle /* parent */)
    {
        return PyFloat_FromDouble((double)src);
    }
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE