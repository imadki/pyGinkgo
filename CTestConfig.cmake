# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2025 pyGinkgo authors

# This file added as per:
# https://cmake.org/pipermail/cmake/2015-January/059742.html : >> Do you have a
# CTestConfig.cmake file in your source tree? >> >> If you do, then ctest will
# load that instead of looking for the >> DartConfiguration.tcl file in your
# build tree... Even if it's empty >> because you don't submit to a CDash
# server, the presence of >> CTestConfig.cmake in your top level source tree
# should suppress this >> error message.

# In such a way to suppress "Cannot find file:
# /sourcerepo/build/DartConfiguration.tcl"
