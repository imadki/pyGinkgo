# tests/test_import.py

import sys

sys.path.append("../build")


def test_import():
    try:
        import pyGinkgo  # noqa: F401

        assert True  # If import succeeds, the test passes

    except ImportError:
        assert False  # If import fails, the test fails
