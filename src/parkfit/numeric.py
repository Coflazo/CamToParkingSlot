"""Bounding the thread pools that numeric libraries allocate on import.

OpenBLAS, MKL and OpenMP each reserve per-thread scratch buffers the first time they are
touched, sized by the core count rather than by the work. On a sixteen-core machine that
is a large allocation made before a single row has been read, and on a machine already
under memory pressure it fails outright::

    OpenBLAS error: Memory allocation still failed after 10 retries, giving up.

This project does not need those pools. The heavy numeric work is either in the C++ core,
which manages its own threads, or in LightGBM, which takes an explicit ``num_threads``.
What numpy does here is means, sorts and comparisons over one-dimensional arrays, work
that a threaded BLAS cannot speed up and only reserves memory for.

**Import order matters.** The variables are read when the native library loads, which
happens at ``import numpy``. Calling :func:`limit_numeric_threads` afterwards has no
effect, so it runs at the top of the entry points, before any import that pulls numpy in.
An explicit value already in the environment is always respected.
"""

from __future__ import annotations

import os

#: The variables each library reads at load time.
_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_numeric_threads(threads: int = 1) -> dict[str, str]:
    """Cap BLAS/OpenMP thread pools, leaving any explicit setting alone.

    Returns the variables this call actually set, so a caller can report what it changed
    rather than claiming credit for the environment it inherited.
    """
    applied: dict[str, str] = {}
    value = str(max(1, threads))
    for name in _THREAD_VARS:
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied


def numpy_already_imported() -> bool:
    """True when it is too late for :func:`limit_numeric_threads` to have any effect.

    Used by the tests, which assert the entry point calls it early enough rather than
    trusting that it does.
    """
    import sys

    return "numpy" in sys.modules
