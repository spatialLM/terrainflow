"""pysheds on NumPy 2 — restore the alias it still calls, and hand back ``Grid``.

pysheds 0.5 calls ``np.in1d`` at **nine** sites in ``sgrid.py`` (``:904``, ``:1152``,
``:1325``, ``:1424``, ``:1507``, ``:1571``, ``:1666``, ``:1821``, ``:1923``). NumPy removed
that name in 2.0; this environment runs NumPy 2.5.2. Every one of those sites sits on a
**D8** code path, so the plugin only met it where it asked pysheds for D8 — which is
exactly what the Routing combo's second entry does. Selecting it took the baseline run down
with ``AttributeError: module 'numpy' has no attribute 'in1d'`` (`ANALYSIS_DEFECTS.md`
§8.2, `FLA-26`).

**Why a shim rather than a wider ``except``.** The obvious-looking fix — widening
``flow_analysis.py``'s ``except TypeError`` to catch ``AttributeError`` — does not work, and
believing it does is worse than leaving the bug alone. ``Grid.accumulation``'s signature
defaults to ``routing='d8'`` (``sgrid.py:822``), so the fallback call in that ``except``
block re-enters ``_d8_accumulation`` and raises the identical error one line further down.
It would have moved the crash, not removed it.

**Why this substitution is exact rather than approximate.** ``np.isin`` is NumPy's own
documented replacement for ``np.in1d``. The two differ in one respect: ``in1d`` always
returned a flat array while ``isin`` preserves the input's shape. Every pysheds call site
above passes ``fdir.ravel()`` — already one-dimensional — and immediately calls
``.reshape(fdir.shape)`` on the result, so the shapes agree at every site and the values
were never in question.

**Scope.** ``np.in1d`` does not exist on NumPy 2, so defining it *adds* a name rather than
replacing one: nothing else in the process can be relying on different behaviour from it,
because anything calling it today already fails. On NumPy 1.x the attribute is present and
this module leaves it alone. The guard is ``hasattr``, not a version comparison, so a future
NumPy that restores the name gets its own implementation back.

Import ``Grid`` **from here** rather than from ``pysheds.grid``. That makes the dependency
structural instead of a bare side-effecting import a tidy-up could delete, and it means
every pysheds entry point in the tree is shimmed by construction rather than by whichever
module happened to be imported first.
"""

import numpy as np


def apply():
    """Restore ``np.in1d`` if this NumPy has removed it. Idempotent."""
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    return hasattr(np, "in1d")


apply()

from pysheds.grid import Grid  # noqa: E402  — must follow apply()

__all__ = ["Grid", "apply"]
