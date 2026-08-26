"""Access to the compiled C++ core, with a graceful fallback.

``parkfit_native`` is built by CMake into this package directory. It carries the work
that runs per candidate on every search, coordinate transforms, the radius sweep over
a quarter of a million bays, vehicle fit and the generalised-cost ranking.

The module is genuinely optional. Everything it provides also exists in pure Python
under :mod:`parkfit.geo` and :mod:`parkfit.domain`, so a checkout that has never been
compiled still runs, just slower. What must never differ is the *answer*: a bay that
fits in C++ has to fit in Python, which is what ``tests/contract/test_native_parity.py``
enforces.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

native: Any | None
try:
    from parkfit import parkfit_native as native  # type: ignore[attr-defined]

    HAS_NATIVE = True
except ImportError:  # pragma: no cover - depends on whether the build has been run
    native = None
    HAS_NATIVE = False
    log.info(
        "parkfit_native is not built; falling back to pure Python. "
        "Run tasks.ps1 build for the compiled path."
    )


def require_native() -> Any:
    """Return the native module, or explain precisely how to get it."""
    if native is None:
        raise RuntimeError(
            "The compiled parkfit_native module is required here but was not found. "
            "Build it with:  .\\tasks.ps1 build"
        )
    return native


def native_version() -> str | None:
    return getattr(native, "__version__", None) if native is not None else None
