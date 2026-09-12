"""p_crest_split_ledger — the crest split's conservation ledger, on a real burn.

Moved here from ``CLudeDocs/wip/`` (H-7), which is a docs folder and outside both
test globs, so nothing ever ran it and nothing ever would. It uses the shipped
``FlowAnalysis.run(crest_split=...)`` API, not the superseded draft that sat
beside it — that draft was `plan_crest_split`/`CrestSplit`, which shipped as
`plan_crest_absorption`/`CrestPlan`/`CrestSpread`, and has been deleted.

**It needs artefacts this repo does not carry** (`burned.tif`, `own.npy`,
`burned.npy` from Round 14), so it is a probe you point at a working directory
rather than one that runs on the fixture. It records; it does not assert.

Run it from the directory holding the Round 14 artefacts (``burned.tif``, ``own.npy``,
``burned.npy``) under plain CPython — no QGIS needed:

    python -B p_crest_split_ledger.py

What it checks, and the figures it should print (Round 15, 858x1027, 1 m, dinf):

    ponds 53, crest cells 309, passes 9, residual 0
    W / baseline / spread all 877,400.0000          <- the ledger, exact
    negatives 0, fdir identical True
    busiest cell 101,654 -> 95,899.7 (-5.661%), terminal delta +6,113.7 / -6,113.7
    Dam 15 band 36,781.50x -> 1.00x

The **ledger** is the acceptance test, not the busiest cell. Water leaves the domain at
cells that pass nothing on, so the sum of accumulation over those cells is the domain
weight — exactly, and it was exactly that before the change too. The busiest cell *moves*,
because two dams whose level crests straddle a divide now spill along their whole length
and part of their discharge leaves by the other coast; the paired terminal deltas show it
is relocation, and they cancel.
"""
import sys
import time
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, r"C:\Users\liamm\Documents\TerrainFlow")
for _m in ("qgis", "qgis.core", "qgis.gui", "qgis.PyQt", "qgis.PyQt.QtCore",
           "qgis.PyQt.QtGui", "qgis.PyQt.QtWidgets"):
    sys.modules.setdefault(_m, MagicMock())

from pysheds import _sgrid as _S  # noqa: E402
from scipy.ndimage import label as _label  # noqa: E402

from terrainflow_assessment.modules.flow_analysis import FlowAnalysis  # noqa: E402

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)


def terminals(fdir):
    """Cells pysheds routes into themselves — where every unit of weight ends up."""
    f = np.ascontiguousarray(np.asarray(fdir, dtype="float64"))
    a, b, _, _ = _S._angle_to_d8_numba(f, DIRMAP, np.ascontiguousarray(~np.isfinite(f)))
    e0 = _S._flatten_fdir_numba(np.ascontiguousarray(a), DIRMAP).ravel()
    e1 = _S._flatten_fdir_numba(np.ascontiguousarray(b), DIRMAP).ravel()
    idx = np.arange(e0.size)
    return (e0 == idx) & (e1 == idx)


plain = FlowAnalysis()
plain.load_dem("burned.tif")
t = time.time()
plain.run(routing="dinf", crest_split=False)
t_plain = time.time() - t
before = np.asarray(plain.acc, dtype="float64")

split = FlowAnalysis()
split.load_dem("burned.tif")
t = time.time()
result = split.run(routing="dinf")
t_split = time.time() - t
after = np.asarray(split.acc, dtype="float64")

# The pond regions again, so pond cells can be left out of the ledger: their value is the
# running total of everything that ever arrived, and on a chain the same water arrives at
# several ponds in turn.
pit = split.grid.fill_pits(split.dem)
ground = np.array(pit, dtype="float64", copy=True)
filled = split.grid.fill_depressions(pit)
ponded = np.array(filled, dtype="float64", copy=True)
split.grid.resolve_flats(filled)
plan = split._crest_plan(ponded, ground)
held = plan.absorb.ravel()

fdir_plain = np.asarray(plain.fdir, dtype="float64")
domain = float(np.isfinite(fdir_plain).sum())
absorbing = np.array(split.fdir, copy=True)
absorbing[plan.absorb] = -1.0

t_before = terminals(fdir_plain)
t_after = terminals(absorbing)
flux = after.ravel()[t_after & ~held].sum()
delta = after.ravel()[t_after & ~held] - before.ravel()[t_after & ~held]
busiest = int(np.argmax(before))

print(f"runtime      {t_plain:.1f}s -> {t_split:.1f}s")
print(f"ponds {result['crest_ponds']}  crest cells {result['crest_cells']}  "
      f"passes {result['crest_passes']}  residual {result['crest_residual']:.2e}  "
      f"skipped {len(result['crest_skipped'])}")
print(f"W                     {domain:>14,.4f}")
print(f"baseline -> terminals {before.ravel()[t_before].sum():>14,.4f}")
print(f"spread   -> terminals {flux:>14,.4f}     "
      f"ledger {flux + result['crest_residual']:,.4f}")
print(f"negatives {int((after < -1e-9).sum())}   acc.min {after.min():.6f}   "
      f"fdir identical "
      f"{np.array_equal(fdir_plain, np.asarray(split.fdir, dtype='float64'), equal_nan=True)}")
print(f"busiest cell {before.ravel()[busiest]:,.0f} -> {after.ravel()[busiest]:,.1f} "
      f"({100 * (after.ravel()[busiest] - before.ravel()[busiest]) / before.ravel()[busiest]:+.3f}%)"
      f"   terminal delta +{delta[delta > 0].sum():,.1f} / {delta[delta < 0].sum():,.1f} "
      f"/ net {delta.sum():+.2e}")

own = np.load("own.npy")
burned = np.load("burned.npy").astype("float64")
pour = float(burned[own].max())
labels, _ = _label(np.isclose(burned, pour, atol=1e-6))
band = np.isin(labels, list(set(labels[own].ravel().tolist()) - {0}))
for name, field in (("BEFORE", before), ("AFTER ", after)):
    v = field[band]
    v = v[v > 0]
    print(f"{name} Dam 15 band ({v.size} cells): min {v.min():,.1f} "
          f"median {np.median(v):,.1f} max {v.max():,.1f}  "
          f"ratio {v.max() / np.median(v):,.2f}x")
print(f"streams over 0.5 ha: {int((before > 5000).sum()):,} -> {int((after > 5000).sum()):,}")
