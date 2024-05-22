![pyGinkgo Logo](./logo.jpg)
# PyGinkgo: Python Binding for Ginkgo

[![License: MIT](badge)](https://opensource.org/license/mit)


PyGinkgo is a Python binding for the Ginkgo framework, providing access to Ginkgo's powerful linear algebra capabilities from Python. Ginkgo is a high-performance numerical linear algebra library for sparse systems, primarily designed for developing efficient iterative solvers on complex HPC architectures.

## Installation

### Prerequisites

- Python 3.x
- Ginkgo 
- Pybind11

### Building the module

1. **Clone the repository**: <git clone>

2. **Build using CMake**: Y

```bash
# Make a build directory in the project directory
mkdir build && cd build

# Run CMake configuration
cmake ..

# Build the project using the specified number of cores (replace "number of cores" with the desired value)
cmake --build . -j=number_of_cores
```

###Running the tests
For now, there is just one test implemented. You would need to install pytest to be able to run this test as follows:
```bash
ctest -R pyginkgo_import_test
```

## Usage

Here's a simple example demonstrating how to use PyGinkgo to perform matrix-vector multiplication:

## Benchmarking

