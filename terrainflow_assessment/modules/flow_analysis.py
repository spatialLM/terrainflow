"""
flow_analysis.py — Pysheds-based flow analysis for TerrainFlow Assessment.

Adapts the base plugin's FlowAnalysis class with one addition:
  exit point dicts now also include ``volume_m3`` (total runoff volume that
  passed through the exit over the analysis period) rather than flow velocity.

Key classes / functions
-----------------------
FlowAnalysis          — flow direction, accumulation, stream network, catchments
AnalysisWorker        — QThread wrapper for background analysis
"""

import numpy as np
import rasterio
from pysheds.grid import Grid

from terrainflow_assessment.modules.footprint import xy_to_rc

# Below this share of the site's runoff, unrouted cells are not worth interrupting for:
# a handful of nodata holes on the edge of a tile is normal and says nothing about the
# design. Set from the Quail Island measurement, where 1.4% was worth reporting and a
# tenth of that would not have been.
UNROUTED_WARN_FRACTION = 0.005


def fdir_nodata(routing):
    """The value that means "no direction here" in a flow-direction raster.

    Routing-dependent, and getting it wrong is not a cosmetic error. Under D8 the
    ESRI codes are the eight powers of two, so ``0`` is genuinely spare and is what
    pysheds itself uses. Under D-infinity the value *is* an angle in radians and
    ``0.0`` means "flows due east" — an ordinary cell on any east-facing slope — so
    the only safe sentinel is NaN. Declare 0 there and every east-flowing cell comes
    back as no-data, routing to itself.

    Module-level rather than a method because both ends of the round trip need it:
    ``AnalysisWorker`` when it writes the raster, ``simulation`` when it reads one
    back. A hand-copy in the second place is how the two ends drift apart.
    """
    return np.nan if routing == 'dinf' else 0


# The step pysheds inflates a flat by, and the smallest one worth keeping. `eps` is a
# height, and the danger is entirely that it is *large*: the Barnes gradient is an integer
# BFS distance, so on a big flat the inflation reaches eps x (thousands) and can lift a
# flat cell above a neighbour that was genuinely lower than it. See `safe_flat_epsilon`.
FLAT_EPS_CEILING = 1e-5
# Floor, in float64 spacings at the surface's own elevation. The synthetic gradient only
# has to be *strictly* positive to route — `d8_from_dem` tests `> 0` — so a thousand-odd
# ULPs is ample, and stopping there keeps one pathological pair from flattening the DEM.
FLAT_EPS_FLOOR_ULPS = 1024.0

# Neighbour offsets. The whole 8-neighbourhood, because a drop is a drop whichever
# direction it runs and D-infinity reads all of them.
_NEIGHBOURS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0))


def _neighbour_views(a, dr, dc):
    """Aligned ``(here, there)`` views of *a* for the neighbour at offset *(dr, dc)*.

    Sliced rather than rolled. ``np.roll`` wraps, which would pair the top row of the
    grid with the bottom as though they touched — inventing a drop across the whole tile
    and letting it bind a bound that only real neighbours should set.
    """
    rows, cols = a.shape
    r0, r1 = max(0, -dr), rows - max(0, dr)
    c0, c1 = max(0, -dc), cols - max(0, dc)
    return (a[r0:r1, c0:c1],
            a[r0 + dr:r1 + dr, c0 + dc:c1 + dc])


def safe_flat_epsilon(surface, gradient,
                      ceiling=FLAT_EPS_CEILING, floor_ulps=FLAT_EPS_FLOOR_ULPS):
    """The largest flat-inflation step that cannot reverse a drop *surface* already has.

    ``resolve_flats`` returns ``surface + eps * gradient`` where ``gradient`` is the
    integer Barnes drainage gradient — zero off the flats, and growing with distance to
    the flat's outlet. So for neighbours A and B with ``z[A] > z[B]``, the inflated
    surface keeps that order only while::

        eps * (g[B] - g[A]) < z[A] - z[B]

    which is the whole of this function. pysheds' default ``eps = 1e-5`` respects no such
    bound, and on a flat big enough it does not hold: measured on the Quail Island tile,
    70.4% of which is one connected harbour plane, the gradient reaches **1842** — an
    inflation of 1.84 cm — while the tightest drop it had to respect was 7.1e-07 m per
    gradient unit. The default was 14x too large, and 516 low bumps standing inside that
    plane were lifted under their own neighbours and became pits: cells with a real way
    downhill that the conditioning took away, absorbing 12.5% of the tile's drainage into
    self-loops that nothing downstream ever heard about.

    The answer is halved for margin and clamped at both ends: never above pysheds' own
    default, so this can only ever be gentler than what it replaces, and never so small
    that the gradient falls inside float64's spacing at the surface's own elevation.

    Returns ``(eps, n_inversions_remaining)``. The second is 0 unless the floor bound,
    and is the count of neighbour pairs the step still reverses — reported rather than
    hidden, because it is exactly the residue this function exists to remove.
    """
    z = np.asarray(surface, dtype="float64")
    g = np.asarray(gradient, dtype="float64")
    if z.shape != g.shape:
        raise ValueError("surface and gradient must have the same shape")

    scale = float(np.nanmax(np.abs(z))) if z.size else 0.0
    floor = floor_ulps * float(np.spacing(scale if np.isfinite(scale) else 1.0))
    if not np.any(g > 0):
        return float(ceiling), 0          # no flats: nothing to inflate, nothing at risk

    def binding_pairs(dr, dc):
        """``(drop, lift)`` for the pairs this offset could invert, both > 0."""
        z_here, z_there = _neighbour_views(z, dr, dc)
        g_here, g_there = _neighbour_views(g, dr, dc)
        drop = z_here - z_there          # > 0 where this cell is above its neighbour
        lift = g_there - g_here          # > 0 where the neighbour is inflated past it
        binding = (drop > 0) & (lift > 0)
        return drop[binding], lift[binding]

    tightest = np.inf
    for dr, dc in _NEIGHBOURS:
        drop, lift = binding_pairs(dr, dc)
        if drop.size:
            tightest = min(tightest, float(np.min(drop / lift)))

    eps = ceiling if not np.isfinite(tightest) else min(ceiling, 0.5 * tightest)
    eps = max(eps, floor)

    # Only when the floor overrode the bound. Counted rather than assumed zero: it is
    # exactly the residue this function exists to remove, and a silent 0 would read as
    # "nothing was buried" on the one surface where something was.
    residual = 0
    if np.isfinite(tightest) and eps >= tightest:
        for dr, dc in _NEIGHBOURS:
            drop, lift = binding_pairs(dr, dc)
            residual += int((eps * lift >= drop).sum())
    return float(eps), residual


def resolve_flats_safely(grid, filled):
    """``grid.resolve_flats`` with an inflation step scaled to the surface it is given.

    ``resolve_flats(filled, eps=1.0)`` is ``filled + drainage_gradient`` — pysheds does
    nothing else with ``eps`` (``sgrid.py``: ``inflated_dem = dem + eps *
    drainage_gradient``) — so calling it that way hands back the integer Barnes gradient
    for free and the surface is rebuilt here at the step :func:`safe_flat_epsilon`
    derives. **One flood, not two**: this replaces the default call rather than following
    it.

    The rebuilt array is re-wrapped with *filled*'s own viewfinder before it goes back to
    pysheds, which asserts ``isinstance(data, Raster)`` on input and would otherwise read
    the surface with the wrong nodata.

    Returns ``(inflated_raster, eps, n_inversions_remaining)``.
    """
    from pysheds.sview import Raster

    base = np.array(filled, dtype="float64", copy=True)
    unit = np.asarray(grid.resolve_flats(filled, eps=1.0), dtype="float64")
    # Integral by construction; rounded because `base + g` and the subtraction back off it
    # are each one float64 op away from exact, and a gradient of 1841.9999999 would make
    # the bound below meaningless.
    gradient = np.rint(unit - base)

    eps, residual = safe_flat_epsilon(base, gradient)
    inflated = base + eps * gradient
    viewfinder = getattr(filled, "viewfinder", None)
    if viewfinder is not None:
        inflated = Raster(np.ascontiguousarray(inflated), viewfinder=viewfinder)
    return inflated, eps, residual


def unrouted_flow_warning(n_cells, flow_held, total,
                          threshold=UNROUTED_WARN_FRACTION):
    """Advisory when water stops inside the site with nowhere to go.

    Counts only cells away from the edge of the data — see
    :meth:`FlowAnalysis.unrouted_flow` for why water stopping at a coastline or a clip
    boundary has left the site rather than gone missing.

    ``flow_held`` and ``total`` must be **the same quantity in the same units**, and
    both must be measured over the same ground. They were neither. The numerator came
    off the whole raster (:meth:`FlowAnalysis.unrouted_flow` took no domain mask) while
    the denominator was the site's cell count, and it was a cell *count* while the
    sentence called it runoff. A field run duly reported **105.0%**, which is what sent
    someone looking for a hydrology bug that was substantially a measurement bug.

    An impossible share is now **printed, not clamped.** A ``min(share, 1.0)`` here
    would have hidden the very thing that got this looked at; a reader who sees 105%
    knows something is wrong with the instrument, and a reader who sees a smooth 100%
    does not. Returns ``None`` when there is nothing to say.
    """
    if not n_cells or flow_held <= 0 or total <= 0:
        return None
    share = flow_held / float(total)
    if share < threshold:
        return None
    msg = (
        f"{n_cells:,} cell{'s' if n_cells != 1 else ''} inside the site have nowhere "
        f"downhill to send water, and hold about {share * 100:.1f}% of its runoff where "
        f"the flow map cannot follow it. These are not ponds — hollows are already filled "
        f"to their spill level — but a hole or a spike the conditioning could not resolve. "
        f"Treat streams and exit volumes below those points as under-reported."
    )
    if share > 1.0:
        msg += (
            " More than 100% is not possible and means this figure is being measured "
            "wrongly, not that the site has lost all its water — see the unrouted "
            "diagnostic written beside the run's rasters."
        )
    return msg


def format_unrouted_diagnostics(diag):
    """Render :meth:`FlowAnalysis.unrouted_diagnostics` as a readable block.

    Walks the dict rather than naming its keys, so a measurement added there cannot be
    silently dropped from the report — which is the failure mode that matters for a
    diagnostic nobody reads until something is already wrong.
    """
    if not diag:
        return "unrouted diagnostics: nothing measured"

    lines = ["TerrainFlow — unrouted cell diagnostics", "=" * 46]

    def emit(key, value, indent=0):
        pad = "  " * indent
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            for k, v in value.items():
                emit(k, v, indent + 1)
        elif isinstance(value, float):
            lines.append(f"{pad}{key}: {value:,.6g}")
        else:
            lines.append(f"{pad}{key}: {value}")

    for key, value in diag.items():
        emit(key, value)

    lines += [
        "",
        "Reading it:",
        "  flat_resolution.eps      — the step the flats were inflated by. Well below",
        "                             pysheds_default_eps means a big flat forced it",
        "                             down; inversions_remaining > 0 means even the",
        "                             floor could not save every drop, and that many",
        "                             cells were pushed under a neighbour.",
        "  location.outside_domain  — counted in the numerator, never in the",
        "                             denominator. Large here means the share was",
        "                             inflated by ground outside the site.",
        "  neighbour_drop.conditioned.has_lower",
        "                           — a lower neighbour exists and the router still",
        "                             called it stuck. That is a comparison bug, not",
        "                             terrain.",
        "  components.singletons vs largest",
        "                           — scattered spikes want a different fix from one",
        "                             unresolved plateau.",
        "  nodata_distance          — mass at 2-4 means the 3x3 nodata halo is too",
        "                             narrow to catch them.",
        "  ratios.acc_max_over_domain_cells",
        "                           — over 1 is impossible for a single-pass D8 and",
        "                             demonstrates the pond re-emission double-count.",
    ]
    return "\n".join(lines)


def crest_spread_warning(unplaced_cells, domain_cells, threshold=UNROUTED_WARN_FRACTION):
    """Advisory when the crest split ran out of passes before a pond chain drained.

    A pond hands its overflow to the next pond down, which hands on what it in turn
    receives, and each pass of the spread moves the water one link — so a long enough chain
    of dams needs more passes than the budget allows. Reported rather than absorbed, because
    the loop is monotone from below: it can only ever *under*-emit, so the number below is
    exactly how much the map is understating and its sign is known.

    (A pond with no way out at all is a different thing and never gets here — it keeps the
    default routing, where ``unrouted_flow`` already reports water with nowhere to go.)

    Same units and the same threshold as :func:`unrouted_flow_warning`: cell-units against
    the domain, so the ratio is the share of the site's runoff. ``None`` when there is
    nothing to say.
    """
    if unplaced_cells <= 0 or domain_cells <= 0:
        return None
    share = unplaced_cells / float(domain_cells)
    if share < threshold:
        return None
    return (
        f"About {share * 100:.1f}% of the site's runoff is still held in ponds: the chain "
        f"of ponds here runs deeper than the crest spreading resolved, and each pass carries "
        f"water only one pond further down. Streams and exit volumes below them are "
        f"under-reported by that much; nothing is over-reported."
    )


# ---------------------------------------------------------------------------
# Core flow analysis
# ---------------------------------------------------------------------------

def pad_for_seeded_walk(fdir):
    """A one-cell-padded copy of *fdir*, and the viewfinder that addresses it.

    Split out from :func:`catchment_from_seed` so a caller delineating many
    outlets off one flow-direction raster pays for the copy once. Under D-infinity
    that copy is a float64 array the size of the DEM.
    """
    from affine import Affine
    from pysheds.sview import Raster, ViewFinder

    view = fdir.viewfinder
    padded = np.pad(np.asarray(fdir), 1, mode="constant", constant_values=0)
    aff = view.affine
    # One cell up and one cell left in *pixel* space. Written out via the full
    # affine rather than as ``c - a`` / ``f - e`` so a rotated or sheared raster
    # steps along its own axes instead of along north and east.
    shifted = Affine(aff.a, aff.b, aff.c - aff.a - aff.b,
                     aff.d, aff.e, aff.f - aff.d - aff.e)
    padded_view = ViewFinder(affine=shifted, shape=padded.shape,
                             nodata=view.nodata, crs=view.crs,
                             mask=np.ones(padded.shape, dtype=bool))
    return Raster(padded, viewfinder=padded_view), padded_view


def catchment_from_seed(grid, fdir, x, y, routing=None, snap=None, padded=None):
    """``grid.catchment``, safe to seed with a cell on the edge of the grid.

    pysheds' catchment kernels are numba ``nopython`` code with bounds checking
    off, and they walk the *flattened* array: the eight neighbours of a cell are
    ``parent ± {1, ncols, ncols ± 1}``, read straight out of ``fdir.flat`` before
    anything asks whether they exist. For a seed on row 0 the three northern
    neighbours are negative flat indices; on the last row the three southern ones
    are past the end. Neither is an ``IndexError`` to catch — each is a read
    outside the allocation, and the process dies with an access violation the
    moment the array is large enough for that read to leave the mapped page. A
    3,000-cell-square DEM is large enough. The 25-cell fixtures in the checks are
    not, and neither is the synthetic DEM the QGIS suite runs on, which is why
    208 green checks sat on top of a crash that killed QGIS on real terrain.

    A bounds test on the seed cannot help here. Every boundary outlet is a rim
    cell *by construction* — :meth:`FlowAnalysis._find_boundary_outlets` only ever
    considers row 0, row -1, column 0 and column -1 — so these seeds are not off
    the grid, they are legitimately on its edge, and the kernel still reads past
    it.

    So the walk runs on a one-cell-padded copy, where every real cell is interior,
    and the mask is cropped back. Padding rather than nudging the seed inland is
    what keeps the answer exact: a rim cell can be fed by three interior
    neighbours at once, and a nudge would keep only the branch it stepped into.
    The padding's own values never matter — ``_pop_rim`` zeroes the outermost ring
    before the walk, which is exactly why the ring has to be there to spare.

    Raises ``ValueError`` for a seed whose cell falls outside *fdir*, which is
    what pysheds already does for a point outside the bounding box; the padding
    is a margin for the kernel to read into, not licence to seed off the raster.
    """
    from pysheds.sview import View

    if padded is None:
        padded = pad_for_seeded_walk(fdir)
    padded_fdir, padded_view = padded

    rows, cols = np.asarray(fdir).shape
    seed_col, seed_row = View.nearest_cell(x, y, affine=fdir.viewfinder.affine,
                                           snap=(snap or "corner"))
    if not (0 <= seed_row < rows and 0 <= seed_col < cols):
        raise ValueError(
            f"Pour point ({x}, {y}) resolves to cell ({seed_row}, {seed_col}), "
            f"outside a raster of shape {(rows, cols)}."
        )

    kwargs = {"x": x, "y": y, "fdir": padded_fdir, "xytype": "coordinate"}
    if snap is not None:
        kwargs["snap"] = snap

    original_view = grid.viewfinder
    grid.viewfinder = padded_view
    try:
        try:
            catch = grid.catchment(routing=routing, **kwargs) if routing \
                else grid.catchment(**kwargs)
        except TypeError:
            catch = grid.catchment(**kwargs)
    finally:
        grid.viewfinder = original_view

    return np.asarray(catch)[1:-1, 1:-1]


class FlowAnalysis:
    """
    Pysheds-based flow direction, accumulation, and catchment delineation.

    Usage::

        fa = FlowAnalysis()
        fa.load_dem(dem_path)
        result = fa.run(routing='dinf')
        exit_pts = fa.get_boundary_exit_points(boundary_path, threshold, runoff_mm, hours)
    """

    def __init__(self):
        self.grid = None
        self.dem = None
        self.conditioned = None
        self.fdir = None
        self.acc = None
        self.crs = None
        self.transform = None
        self.nodata = None
        self.routing = 'dinf'
        # Set by :func:`resolve_flats_safely` on every run. Declared here so the
        # diagnostics can report "not run" rather than raise on a fresh instance.
        self.flat_eps = None
        self.flat_inversions = None

    def load_dem(self, dem_path):
        """Load DEM from GeoTIFF and initialise pysheds grid."""
        self.grid = Grid.from_raster(dem_path)
        self.dem = self.grid.read_raster(dem_path)
        with rasterio.open(dem_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.nodata = src.nodata
        return True

    def run(self, routing='dinf', runoff_weights=None, crest_split=True):
        """
        Run the full flow analysis pipeline:
          1. Fill pits
          2. Fill depressions (priority-flood)
          3. Resolve flats
          4. Compute flow direction (D-infinity or D8)
          5. Compute flow accumulation (optionally weighted by per-cell runoff),
             with every pond contracted to a mixing node so it spills evenly along
             its crest — see :meth:`_spread_crests`.

        Step 2 is a **fill**, not a breach. ``breach_depressions`` does not exist in
        pysheds 0.5 — a grep over the installed package finds no definition — so the
        ``except AttributeError`` branch below has always been the one that runs, and
        filling is what we want anyway. The old name and comment here claimed otherwise
        for long enough to mislead a diagnosis (Round 14), which is why they now say
        what the code does.

        Parameters
        ----------
        routing : str — 'dinf' or 'd8'
        runoff_weights : numpy array or None
            Per-cell runoff volume (m³) for weighted accumulation.
        crest_split : bool
            Contract ponds and spread their overflow along the crest. ``False`` restores
            the pipeline exactly as it was before Round 15 — kept so the before/after can
            be measured from one process, and so "no ponds means nothing moved" is
            assertable rather than argued.

        Returns dict with 'flow_direction', 'flow_accumulation', 'conditioned_dem',
        'unrouted_cells', 'unrouted_flow' and the 'crest_*' counters, and optionally
        'runoff_accumulation'.
        """
        if self.grid is None:
            raise RuntimeError("DEM not loaded. Call load_dem() first.")

        self.routing = routing

        pit_filled = self.grid.fill_pits(self.dem)
        # ``fill_depressions`` writes into its input buffer and hands the same array back:
        # ``np.shares_memory(filled, pit_filled)`` is True, so the pre-fill surface has to be
        # copied out *here*. Measured against it afterwards, every pond is 0.00 m deep and
        # none is ever found (Round 14).
        ground = np.array(pit_filled, dtype="float64", copy=True)
        try:
            filled = self.grid.breach_depressions(pit_filled)
            self.depression_method = "breach"
        except AttributeError:
            filled = self.grid.fill_depressions(pit_filled)
            # Recorded rather than assumed. pysheds 0.5 has no `breach_depressions`, so
            # this is the branch that always runs — but "always" is a claim about an
            # installed version, and the diagnostic should report which one it got
            # rather than restate the claim.
            self.depression_method = "fill"
        # ``resolve_flats`` inflates the flats it is given, and a pond's rim is a flat, so
        # the level surface has to be copied before it too.
        ponded = np.array(filled, dtype="float64", copy=True)

        # Retained. Standing water is ``ponded - ground``, and the analysis worker
        # used to get that number by building a second DEMBurner and re-running
        # fill_pits and fill_depressions over the same DEM — two more full reads
        # and two more floodings, for a surface already sitting in this frame.
        self.ground = ground
        self.ponded = ponded

        # Not ``grid.resolve_flats(filled)``. Its default step inflates a flat by enough
        # to lift cells over neighbours that were genuinely lower, which is what put 516
        # cells and 12.5% of the tile's drainage into self-loops on the Quail Island run
        # — see :func:`safe_flat_epsilon`. The step is derived from this surface instead.
        inflated, self.flat_eps, self.flat_inversions = resolve_flats_safely(
            self.grid, filled)
        # Keep the conditioned surface: it is what flow_graph derives its D8 pointers
        # from. Steepest descent on this array is provably acyclic (pits filled,
        # depressions filled, flats resolved), unlike rounding the D-infinity angles.
        self.conditioned = inflated

        try:
            self.fdir = self.grid.flowdir(inflated, routing=routing)
        except TypeError:
            self.routing = 'd8'
            self.fdir = self.grid.flowdir(inflated)

        plan = self._crest_plan(ponded, ground) if crest_split else None
        crest = self._spread_crests(plan) if plan is not None else None
        if crest is not None:
            self.acc = crest.accumulation
        else:
            try:
                self.acc = self.grid.accumulation(self.fdir, routing=self.routing)
            except TypeError:
                self.acc = self.grid.accumulation(self.fdir)

        unrouted_cells, unrouted_flow = self.unrouted_flow()

        result = {
            "flow_direction": self.fdir,
            "flow_accumulation": self.acc,
            "conditioned_dem": inflated,
            "unrouted_cells": unrouted_cells,
            "unrouted_flow": unrouted_flow,
            "crest_ponds": crest.ponds if crest else 0,
            "crest_cells": crest.outlet_cells if crest else 0,
            "crest_passes": crest.passes if crest else 0,
            "crest_residual": crest.residual if crest else 0.0,
            "crest_skipped": list(crest.skipped) if crest else [],
            # Each pool painted with its pond's whole throughput, in the same cell-units as
            # the accumulation. Inside a contracted pond the accumulation is no longer
            # contributing area — it is what arrived at that cell and stopped — so anything
            # reading it as catchment size needs this instead.
            "pond_flow": crest.pond_flow if crest else None,
        }

        if runoff_weights is not None:
            if plan is not None:
                # The ponds and their exits are a fact about the terrain, so the plan is
                # reused rather than rebuilt. Spreading this field too is not optional: it
                # is what ``throughflow_*.tif`` and the exit volumes are read off, and
                # leaving it unspread would have them disagree with the accumulation beside
                # them by the whole of the crest correction.
                #
                # ``retain=True`` only here. This field is in m³, so a pond can be held to
                # its measured storage and the raster becomes the water that actually gets
                # downstream rather than the water that would if nothing were held. The
                # cell-count field above keeps every drop moving, because a contributing-area
                # count is not a volume and streams/keypoints/Tc are asking a different
                # question of it.
                spread = self._spread_crests(
                    plan, base_weights=np.asarray(runoff_weights, dtype="float64"),
                    retain=True,
                )
                result["runoff_accumulation"] = spread.accumulation
                result["crest_retained_m3"] = spread.retained
                # The residual off *this* pass, and so already m³ — unlike
                # ``crest_residual`` above, which comes off the cell-count spread and
                # has to be scaled by cell area and depth before it can be printed as a
                # volume. Two numbers that look alike and are not, so the one that needs
                # no conversion says so in its name.
                result["crest_residual_m3"] = spread.residual
            else:
                try:
                    from pysheds.sview import Raster as _PR
                    weights_raster = _PR(
                        runoff_weights.astype("float64"),
                        viewfinder=self.fdir.viewfinder,
                    )
                except Exception:
                    weights_raster = runoff_weights
                try:
                    weighted = self.grid.accumulation(
                        self.fdir, weights=weights_raster, routing=self.routing
                    )
                except TypeError:
                    weighted = self.grid.accumulation(self.fdir, weights=weights_raster)
                result["runoff_accumulation"] = weighted

        return result

    def _crest_plan(self, filled, ground):
        """Find the ponds and work out which of their cells discharge. ``None`` if none do.

        ``filled`` is the depression-filled surface and ``ground`` the same surface *before*
        the fill — both copies taken in :meth:`run`, because pysheds fills in place.
        """
        from terrainflow_assessment.modules import crest_routing
        from terrainflow_assessment.modules.flow_graph import d8_from_dem

        cell_w = abs(self.transform.a) if self.transform is not None else 1.0
        cell_h = abs(self.transform.e) if self.transform is not None else 1.0

        impoundments, skipped = crest_routing.find_impoundments(
            filled, ground, cell_area_m2=cell_w * cell_h)
        if not impoundments:
            return None

        surface = self.conditioned if self.conditioned is not None else self.dem
        next_flat, is_sink = d8_from_dem(
            np.asarray(surface, dtype="float64"),
            cell_w=cell_w, cell_h=cell_h, nodata=self.nodata,
        )
        # Returned even when nothing survived the exit test: the plan still carries *why*,
        # and a pond that kept the default routing is worth saying out loud. Spreading an
        # empty plan absorbs nothing, so the accumulation comes back unchanged.
        return crest_routing.plan_crest_absorption(
            impoundments, next_flat, is_sink,
            np.asarray(self.fdir).shape, skipped=skipped,
        )

    def _spread_crests(self, plan, base_weights=None, retain=False):
        """Accumulate with every pond contracted to a mixing node.

        ``retain=True`` also has each pond hold back what it can store, so the field is the
        water that actually gets downstream. **Only ever for a runoff-weighted pass**: the
        capacities on the plan are m³, and the engine's default field is a cell count, so
        retaining against that would cap a count with a volume. It is off by default for
        exactly that reason, and ``run`` sets it on the one call whose units are cubic
        metres — which is also why streams, keypoints and the time-of-concentration path,
        all of which read the cell-count field, are untouched by any of this.

        A pool spills along its whole level crest at once, and one pointer per cell cannot
        divide the load of one cell, so the split is done in the **flux field** instead —
        see :mod:`terrainflow_assessment.modules.crest_routing` for the whole argument and
        the measurements behind it.

        **The flow directions are not touched.** The absorbing map is a private copy, so
        ``feature_inflow_m3``, the catchment labelling and capture % are unaffected by
        construction — and ``unrouted_flow`` keeps reading the real directions rather than
        counting every pond cell as a flat with nowhere to go.
        """
        from pysheds.sview import Raster as _PR

        from terrainflow_assessment.modules import crest_routing

        # Wrap with the *direction* raster's viewfinder, not the DEM's: for D-infinity its
        # nodata is NaN while the DEM's is a real sentinel, and borrowing the wrong one hands
        # the nodata border a weight of 1 and silently changes the domain.
        viewfinder = self.fdir.viewfinder
        fdir_abs = np.array(self.fdir, copy=True)
        fdir_abs[np.asarray(plan.absorb, dtype=bool)] = self.FDIR_FLAT
        fdir_abs = _PR(fdir_abs, viewfinder=viewfinder)

        def accumulate(weights):
            if weights is None:
                try:
                    return self.grid.accumulation(fdir_abs, routing=self.routing)
                except TypeError:
                    return self.grid.accumulation(fdir_abs)
            w = _PR(np.ascontiguousarray(weights, dtype="float64"), viewfinder=viewfinder)
            try:
                return self.grid.accumulation(fdir_abs, weights=w, routing=self.routing)
            except TypeError:
                return self.grid.accumulation(fdir_abs, weights=w)

        return crest_routing.spread_crests(
            plan, accumulate, base_weights=base_weights,
            capacities=plan.capacities if retain else None)

    # Sentinels pysheds writes into a flow-direction array for a cell it could not route:
    # ``flat`` when the best available slope is exactly zero, ``pit`` when every neighbour
    # is higher. Both are the library's own defaults (``_sgrid._d8_flowdir_numba``).
    FDIR_FLAT = -1
    FDIR_PIT = -2

    def ponding_depth(self):
        """Standing-water depth (m), from the surfaces :meth:`run` conditioned.

        Full resolution, unlike ``DEMBurner.get_ponding_layer``, which downsamples
        past a cell cap and resamples back. Returns None before a run.
        """
        ground = getattr(self, "ground", None)
        ponded = getattr(self, "ponded", None)
        if ground is None or ponded is None:
            return None
        depth = np.asarray(ponded, dtype="float64") - np.asarray(ground, dtype="float64")
        return np.where(np.isfinite(depth), np.maximum(depth, 0.0), 0.0)

    def unrouted_flow(self, domain=None, field=None):
        """Water that stops **inside** the site because the routing could not place it.

        A cell pysheds marks ``flat`` or ``pit`` is dropped from the direction map before
        accumulation (``sgrid.py`` sets it to 0, which ``_flatten_fdir_numba`` maps to the
        cell itself). It becomes a **self-loop**: everything upstream arrives, nothing
        leaves, and the water is absent from every downstream figure with nothing saying
        so. Under D-infinity it is the same — a flat angle decomposes to two zero-proportion
        directions rewritten to 0.5/0.5 back into the same cell.

        **These are not ponds.** ``fill_depressions`` has already raised every hollow to its
        spill level, which is exactly the mechanism that makes a filling pond overflow to
        the next cell. What is left over is conditioning residue, and on a real site almost
        all of it sits against **nodata**: Quail Island is an island, so 35 of its 36 stuck
        cells are on the coastline, where the only downhill neighbour is sea that carries no
        elevation. That water has *left the analysed area* — it is not lost, it is simply
        unlabelled on this tier, and warning about it would fire on every clipped or coastal
        DEM for a benign reason.

        So the split is the same one the design tier already makes, ``LABEL_EXIT`` at the
        edge of the data against ``LABEL_SINK`` in an interior pit, and only the second is
        reported here. On Quail Island that is **1 cell, not 36** — which is the honest
        number, and quiet.

        Nothing is re-routed: a cell with no lower neighbour genuinely has nowhere to send
        water, and inventing a destination would be worse than reporting the fact. Because
        such a cell absorbs its whole upstream, its own accumulation *is* the held load.

        ``domain`` — the analysed area, as a boolean mask. **Pass it.** Without it the
        count and the load are taken over the whole raster while the caller divides them
        by the *site's* cell count, so a pit in the DEM buffer outside the boundary
        contributes its entire upstream to a ratio it is not part of. That is most of
        how a field run reported "105.0% of runoff" held in 516 cells: on a DEM 2.8 M
        cells wide against a much smaller drawn site, the numerator was counting ground
        the denominator had never heard of. Optional only because the mask needs file
        I/O the analysis itself does not do.

        ``field`` — the raster to measure the held load in, defaulting to ``self.acc``.
        ``acc`` is a **cell count**, so a share of it is a share of contributing area
        and calling that "% of runoff" is a category error; pass ``runoff_accumulation``
        and the ratio is m³ over m³ and the sentence is literally true.

        Returns ``(n_cells, flow)`` for the interior cells only. ``(0, 0.0)`` when the
        analysis has not run.
        """
        if self.fdir is None or self.acc is None:
            return 0, 0.0
        fdir = np.asarray(self.fdir, dtype="float64")
        stuck = (fdir == self.FDIR_FLAT) | (fdir == self.FDIR_PIT)
        if not stuck.any():
            return 0, 0.0
        stuck &= ~self._at_data_edge()
        if domain is not None:
            stuck &= np.asarray(domain, dtype=bool)
        if not stuck.any():
            return 0, 0.0
        weights = np.asarray(self.acc if field is None else field, dtype="float64")
        held = float(np.nansum(weights[stuck]))
        return int(stuck.sum()), held

    def unrouted_diagnostics(self, domain=None, field=None, runoff_volume_m3=None):
        """Measure *why* cells are unrouted, rather than only how many there are.

        A field run reported 516 stuck cells holding "105.0% of runoff". Three separate
        defects in the ratio account for the percentage (see
        :func:`unrouted_flow_warning` and :meth:`unrouted_flow`), and those are fixed —
        but the 516 are real and the cause is genuinely unknown. Several mechanisms
        could produce them and they need different fixes, so this measures which one it
        is instead of picking the most plausible:

        * a cell two or more cells in from a **nodata hole**. ``_at_data_edge`` dilates
          by a single 3x3, so anything further in survives the filter, and
          ``fill_depressions`` cannot flood across nodata — a hollow whose only outlet
          crosses a hole never fills. → ``nodata_distance``, ``nodata_neighbours``.
        * a **plateau interior** ``resolve_flats`` did not reach. → ``components``: 516
          singletons is spikes, three blobs of ~170 is unreached flats, and the two want
          opposite fixes.
        * a genuine **pit that survived the fill**, which should be impossible unless
          nodata blocked the flood. → ``neighbour_drop``, the sharpest number here. On
          the *conditioned* surface, ``>0`` means every neighbour is higher; ``==0`` is
          an exact tie no flat resolution broke; ``<0`` means a lower neighbour exists
          and pysheds still called it a pit, which would be a comparison bug of the same
          class as the strict ``>`` in ``flow_graph.d8_from_dem``.

        ``ratios`` computes the share several ways side by side, so how much of the 105%
        each defect owned is a matter of reading rather than of argument. ``field_ink``
        counts how much of the runoff field is non-zero and how much clears the log
        ramp's first stop — evidence for the separate complaint that Surface Runoff
        "stops" below a pond, at the cost of two lines while everything else is loaded.

        ``runoff_volume_m3`` is **the denominator the warning uses**: the rain that fell
        on the domain. Pass it. Without it the m³ ratios below divide by the sum of the
        throughflow field over *every* cell, which is not a volume of water at all — it
        is each cubic metre counted once per cell it passes — and on the Quail Island run
        it read 0.000596 against a warning that correctly said 13.4%. A file written to
        check the warning cannot understate it 225-fold; the keys now say what they
        divide by.

        Pure numpy/scipy and returns a plain dict; the worker writes it beside the run's
        rasters and the controllers print the path. Nothing here feeds an analysis.
        """
        from scipy.ndimage import distance_transform_edt, label, minimum_filter

        out = {"conditioning": getattr(self, "depression_method", "unknown"),
               "routing": self.routing}
        # The step the flats were inflated by, and what it could not save. This is where
        # the 516 came from, so the run says which one it used rather than leaving the
        # next reader to assume pysheds' default.
        out["flat_resolution"] = {
            "eps": getattr(self, "flat_eps", None),
            "pysheds_default_eps": FLAT_EPS_CEILING,
            "inversions_remaining": getattr(self, "flat_inversions", None),
        }
        if self.fdir is None or self.acc is None:
            out["error"] = "analysis has not run"
            return out

        fdir = np.asarray(self.fdir, dtype="float64")
        stuck = (fdir == self.FDIR_FLAT) | (fdir == self.FDIR_PIT)
        at_edge = self._at_data_edge()
        dom = (np.ones(fdir.shape, dtype=bool) if domain is None
               else np.asarray(domain, dtype=bool))

        out["totals"] = {
            "grid_cells": int(fdir.size),
            "domain_cells": int(dom.sum()),
            "stuck_any": int(stuck.sum()),
            "stuck_flat": int((fdir == self.FDIR_FLAT).sum()),
            "stuck_pit": int((fdir == self.FDIR_PIT).sum()),
        }

        # Where they are. `outside_domain` is the one that sized the ratio bug: those
        # cells were in the numerator and their ground was never in the denominator.
        interior = stuck & ~at_edge
        out["location"] = {
            "at_data_edge": int((stuck & at_edge).sum()),
            "outside_domain": int((interior & ~dom).sum()),
            "interior_in_domain": int((interior & dom).sum()),
        }

        focus = interior & dom
        n_focus = int(focus.sum())
        out["focus_cells"] = n_focus
        if n_focus:
            dem = np.asarray(self.dem, dtype="float64")
            invalid = ~np.isfinite(dem)
            if self.nodata is not None:
                invalid |= dem == self.nodata

            if invalid.any():
                dist = distance_transform_edt(~invalid)
                d = dist[focus]
                buckets = {}
                for lo, hi, name in ((0, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 3.5, "3"),
                                     (3.5, 4.5, "4"), (4.5, 5.5, "5"),
                                     (5.5, 10.5, "6-10"), (10.5, np.inf, ">10")):
                    buckets[name] = int(((d >= lo) & (d < hi)).sum())
                out["nodata_distance"] = buckets
            else:
                out["nodata_distance"] = "no nodata in the DEM"

            nb_invalid = np.zeros(fdir.shape, dtype="int16")
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb_invalid += np.roll(np.roll(invalid, dr, 0), dc, 1).astype("int16")
            counts = nb_invalid[focus]
            out["nodata_neighbours"] = {
                str(k): int((counts == k).sum()) for k in range(9)
                if int((counts == k).sum())
            }

            lab, n_comp = label(focus, structure=np.ones((3, 3), dtype=bool))
            sizes = np.bincount(lab.ravel())[1:]
            out["components"] = {
                "count": int(n_comp),
                "singletons": int((sizes == 1).sum()),
                "largest": int(sizes.max()) if sizes.size else 0,
                "sizes_top10": [int(s) for s in np.sort(sizes)[::-1][:10]],
            }

            # Steepest available drop on each conditioned surface. `minimum_filter`
            # includes the cell itself, so a cell that IS its own minimum reads 0 — and
            # 0 is exactly the "no strictly lower neighbour" case being looked for, so
            # that is the right degenerate answer rather than a bug to work around.
            for name, surf in (("conditioned", getattr(self, "conditioned", None)),
                               ("filled", getattr(self, "ponded", None)),
                               ("ground", getattr(self, "ground", None))):
                if surf is None:
                    continue
                s = np.asarray(surf, dtype="float64")
                lowest = minimum_filter(s, size=3, mode="nearest")
                drop = s[focus] - lowest[focus]     # >0 means a lower neighbour exists
                out.setdefault("neighbour_drop", {})[name] = {
                    "has_lower": int((drop > 0).sum(),),
                    "exactly_level": int((drop == 0).sum()),
                }

            for name, mask in (("in_pond_pool", getattr(self, "_last_pools", None)),):
                if mask is not None:
                    out[name] = int((focus & np.asarray(mask, dtype=bool)).sum())

        # The same share four ways, so the size of each defect is read rather than
        # argued. `today` reproduces the pre-fix figure exactly, on purpose.
        acc = np.asarray(self.acc, dtype="float64")
        dom_cells = float(dom.sum()) or 1.0
        ratios = {
            "today_unmasked_count_over_domain_cells":
                float(np.nansum(acc[interior])) / dom_cells,
            "domain_masked_count_over_domain_cells":
                float(np.nansum(acc[focus])) / dom_cells,
        }
        if field is not None:
            f = np.asarray(field, dtype="float64")
            # Over the rain that fell, which is what the warning divides by and what
            # "% of runoff" means. Summing the field itself totals throughflow — every
            # cubic metre counted once per cell it passes — and is reported below as
            # `field_ink.total_throughflow_m3` under a name that says so.
            total = float(runoff_volume_m3 or 0.0)
            if total > 0:
                ratios["m3_over_m3_unmasked"] = float(np.nansum(f[interior])) / total
                ratios["m3_over_m3_domain_masked"] = float(np.nansum(f[focus])) / total
                ratios["runoff_volume_m3"] = total
            out["field_ink"] = {
                "total_throughflow_m3": float(np.nansum(f[dom])),
                "non_zero_cells": int(np.count_nonzero(np.nan_to_num(f[dom]))),
                "above_log_first_stop": int(
                    (np.nan_to_num(f[dom]) > 1e-4 * float(np.nanmax(f))).sum())
                if np.isfinite(np.nanmax(f)) and np.nanmax(f) > 0 else 0,
                "max": float(np.nanmax(f)) if f.size else 0.0,
            }
        # Neither can exceed 1 under a single-pass D8. `acc` under crest contraction is
        # a sum over passes and a pond re-emits at its exits, so a value over 1 here is
        # the double-count demonstrating itself, with no second run needed.
        ratios["acc_max_over_domain_cells"] = float(np.nanmax(acc)) / dom_cells
        out["ratios"] = ratios
        return out

    def _at_data_edge(self):
        """Cells on the grid border or touching nodata — where the site's data runs out."""
        from scipy.ndimage import binary_dilation

        dem = np.asarray(self.dem, dtype="float64")
        invalid = ~np.isfinite(dem)
        if self.nodata is not None:
            invalid |= dem == self.nodata
        edge = binary_dilation(invalid, structure=np.ones((3, 3), dtype=bool))
        edge[0, :] = edge[-1, :] = True
        edge[:, 0] = edge[:, -1] = True
        return edge

    def delineate_catchment(self, x, y):
        """Return boolean mask for the catchment draining to (x, y)."""
        if self.fdir is None:
            raise RuntimeError("Run flow analysis first.")
        return catchment_from_seed(self.grid, self.fdir, x, y, routing=self.routing)

    def get_stream_network(self, accumulation_threshold=1000):
        """Boolean mask where accumulation > threshold."""
        if self.acc is None:
            raise RuntimeError("Run flow analysis first.")
        return self.acc > accumulation_threshold

    def get_runoff_volume_raster(self, runoff_mm, cell_area_m2):
        """
        Runoff volume raster (m³): each cell = upstream_cells × runoff_m × cell_area.
        """
        if self.acc is None:
            raise RuntimeError("Run flow analysis first.")
        runoff_m = runoff_mm / 1000.0
        return np.array(self.acc, dtype="float32") * runoff_m * cell_area_m2

    def get_boundary_exit_points(self, boundary_path, min_flow_ls,
                                 runoff_mm, duration_hours, volume_raster=None):
        """
        Find flow exit points where the drainage network crosses the site boundary.

        An exit is a boundary crossing carrying at least *min_flow_ls* litres per
        second (event-average). A **physical L/s criterion** is used instead of a
        raw upstream-cell count so the threshold is scale-appropriate: a small
        permaculture plot and a large catchment use the same meaningful number.

        Robust to D-infinity flow splitting: the per-cell flow raster is 3×3
        max-pooled before sampling the boundary, so a channel whose accumulation is
        spread across adjacent near-boundary cells still registers its true peak
        flow at the crossing (otherwise each split cell can fall under the threshold
        and the crossing disappears).

        Each exit point dict contains: x, y, flow_ls, volume_m3, label.

        Parameters
        ----------
        boundary_path : str
        min_flow_ls : float — minimum event-average flow (L/s) for an exit to show
        runoff_mm : float
        duration_hours : float
        volume_raster : ndarray or None — per-cell runoff volume (m³) over the
            event; when None it is derived as acc × cell_area × runoff_m (exact for
            a spatially-uniform storm).

        Returns list of dicts sorted by flow_ls descending.
        """
        import geopandas as gpd
        from rasterio.features import rasterize
        from scipy.ndimage import maximum_filter

        if self.acc is None:
            raise RuntimeError("Run flow analysis first.")

        duration_s = duration_hours * 3600.0
        if duration_s <= 0:
            return []

        gdf = gpd.read_file(boundary_path)
        gdf = gdf.to_crs(self.crs.to_wkt())

        boundary_lines = [
            geom.exterior for geom in gdf.geometry
            if geom is not None and hasattr(geom, "exterior")
        ]
        if not boundary_lines:
            return []

        boundary_mask = rasterize(
            [(line, 1) for line in boundary_lines],
            out_shape=self.grid.shape,
            transform=self.transform,
            fill=0, all_touched=True, dtype="uint8",
        ).astype(bool)

        cell_w = abs(self.transform.a)
        cell_h = abs(self.transform.e)
        cell_area_m2 = cell_w * cell_h
        runoff_m = runoff_mm / 1000.0

        # Per-cell runoff volume passing each cell over the event (m³), → L/s.
        if volume_raster is not None:
            vol = np.array(volume_raster, dtype="float64")
        else:
            vol = np.array(self.acc, dtype="float64") * cell_area_m2 * runoff_m
        flow_ls = vol * 1000.0 / duration_s
        # Recover D-infinity-split channels: peak flow in each cell's 3×3 window.
        flow_ls_pooled = maximum_filter(flow_ls, size=3)

        exit_mask = boundary_mask & (flow_ls_pooled >= min_flow_ls)
        rows, cols = np.where(exit_mask)
        if len(rows) == 0:
            return []

        # Cell-centre coordinates ((row/col + 0.5) handles the negative y pixel size).
        xs = self.transform.c + (cols + 0.5) * self.transform.a
        ys = self.transform.f + (rows + 0.5) * self.transform.e
        flows = flow_ls_pooled[rows, cols]

        # Group qualifying boundary cells into crossings: seed at the highest-flow
        # cell, absorb every not-yet-used cell within ~20 cells, and place the exit
        # at the crossing's FLOW-WEIGHTED centroid so the marker sits on the channel
        # thread — not at a corner of the max-pooled plateau (which a spatial
        # tie-break would pick).
        min_dist = cell_w * 20
        order = list(np.argsort(flows)[::-1])
        used = np.zeros(len(flows), dtype=bool)

        results = []
        for i in order:
            if used[i]:
                continue
            dx = xs - xs[i]
            dy = ys - ys[i]
            near = (~used) & (dx * dx + dy * dy < min_dist ** 2)
            used[near] = True
            w = flows[near]
            if w.sum() > 0:
                cx = float(np.average(xs[near], weights=w))
                cy = float(np.average(ys[near], weights=w))
            else:
                cx = float(xs[near].mean())
                cy = float(ys[near].mean())
            peak = float(w.max())
            volume_m3 = peak * duration_s / 1000.0
            results.append({
                "x": cx,
                "y": cy,
                "flow_ls": round(peak, 2),
                "volume_m3": round(volume_m3, 1),
                "label": "",
            })

        results.sort(key=lambda r: r["flow_ls"], reverse=True)
        for i, r in enumerate(results):
            r["label"] = (
                f"Exit {i + 1}: {r['flow_ls']:,.1f} L/s "
                f"({r['volume_m3']:,.0f} m³ over event)"
            )

        return results

    def boundary_outflow_total(self, boundary_path, runoff_mm, duration_hours,
                               volume_raster=None):
        """Total water leaving the polygon in *boundary_path* over the event.

        The companion to :meth:`get_boundary_exit_points`, and deliberately not built
        from it. That method exists to **place markers**, so it drops crossings under a
        user threshold, 3×3 max-pools to recover D-infinity splits, and collapses each
        crossing to its single peak cell. Summing its results therefore under-reports
        the real total — badly, on a site with many small crossings — and lowering the
        threshold to zero would only add spurious crossings without fixing the peak-cell
        collapse. This measures the flux instead, and applies **no threshold at all**.

        Method: steepest-descent D8 pointers over the conditioned surface, via the same
        ``flow_graph.d8_from_dem`` the catchment labelling uses. A cell counts as leaving
        when it sits inside the polygon and the cell it drains into does not, so only the
        last inside-cell of a flow path contributes and a channel crossing several
        boundary cells is counted once rather than once per cell. Interior pits are
        ponding rather than outflow and are excluded; a pit on the DEM edge is counted as
        leaving, matching ``flow_graph``'s ``LABEL_EXIT`` convention.

        Note this is water *leaving*, not runoff *generated* inside: flow that entered
        from upstream terrain outside the polygon is included, which is what "water
        leaving the site" should mean.

        Returns ``{"volume_m3": float, "flow_ls": float}``.
        """
        import geopandas as gpd
        from rasterio.features import rasterize

        from terrainflow_assessment.modules.flow_graph import d8_from_dem

        if self.acc is None:
            raise RuntimeError("Run flow analysis first.")

        empty = {"volume_m3": 0.0, "flow_ls": 0.0}
        duration_s = duration_hours * 3600.0
        if duration_s <= 0:
            return empty

        gdf = gpd.read_file(boundary_path)
        gdf = gdf.to_crs(self.crs.to_wkt())
        # Polygons only. A line has no interior, so "what leaves it" is meaningless —
        # but rasterize() would happily burn cells along it and return a plausible
        # number, which is worse than returning nothing.
        polys = [
            g for g in gdf.geometry
            if g is not None and not g.is_empty
            and g.geom_type in ("Polygon", "MultiPolygon")
        ]
        if not polys:
            return empty

        # all_touched=False so the mask is the polygon interior; a cell straddling the
        # edge belongs outside, which keeps the inside→outside test unambiguous.
        inside = rasterize(
            [(g, 1) for g in polys],
            out_shape=self.grid.shape,
            transform=self.transform,
            fill=0, all_touched=False, dtype="uint8",
        ).astype(bool)
        if not inside.any():
            return empty

        cell_w = abs(self.transform.a)
        cell_h = abs(self.transform.e)
        if volume_raster is not None:
            vol = np.array(volume_raster, dtype="float64")
        else:
            vol = (np.array(self.acc, dtype="float64")
                   * cell_w * cell_h * (runoff_mm / 1000.0))

        surface = self.conditioned if self.conditioned is not None else self.dem
        next_flat, is_sink = d8_from_dem(
            np.asarray(surface, dtype="float64"),
            cell_w=cell_w, cell_h=cell_h, nodata=self.nodata,
        )

        inside_flat = inside.ravel()
        leaves = inside_flat & ~is_sink & ~inside_flat[next_flat]

        # A pit on the DEM edge has nowhere lower to point, so it never registers as
        # draining outward; flow_graph calls the grid edge an exit and so do we.
        edge = np.zeros(inside.shape, dtype=bool)
        edge[0, :] = edge[-1, :] = True
        edge[:, 0] = edge[:, -1] = True
        leaves |= inside_flat & is_sink & edge.ravel()

        total_m3 = float(vol.ravel()[leaves].sum())
        return {
            "volume_m3": round(total_m3, 1),
            "flow_ls": round(total_m3 * 1000.0 / duration_s, 2),
        }

    def get_catchment_polygons(self, outlet_points=None, stream_threshold=500):
        """
        Delineate non-overlapping sub-catchment polygons.

        Parameters
        ----------
        outlet_points : list of (x, y) or None (auto-detected from grid edge)
        stream_threshold : int

        Returns list of dicts: {id, geometry, area_m2, area_ha, flow_bearing, label}
        """
        from rasterio.features import shapes as rasterio_shapes
        from shapely.geometry import shape as shapely_shape
        from shapely.ops import unary_union

        if self.fdir is None or self.acc is None:
            raise RuntimeError("Run flow analysis first.")

        acc_array = np.array(self.acc)

        if not outlet_points:
            outlet_points = self._find_boundary_outlets(acc_array, stream_threshold)
            if not outlet_points:
                max_idx = np.unravel_index(np.argmax(acc_array), acc_array.shape)
                row, col = max_idx
                # transform.e is negative (north-up), so the centre of a cell is half a
                # cell *down* from its top edge: (row + 0.5) * e. Adding +cell_h/2
                # instead seeds a point one full cell north of the intended cell.
                x = self.transform.c + (col + 0.5) * self.transform.a
                y = self.transform.f + (row + 0.5) * self.transform.e
                outlet_points = [(x, y)]

        def _acc_at(x, y):
            row, col = xy_to_rc(self.transform, x, y)
            row = max(0, min(acc_array.shape[0] - 1, row))
            col = max(0, min(acc_array.shape[1] - 1, col))
            return int(acc_array[row, col])

        outlet_points_sorted = sorted(outlet_points, key=lambda pt: _acc_at(*pt))
        fdir_array = np.array(self.fdir)
        claimed = np.zeros(acc_array.shape, dtype=bool)
        # One padded copy for every outlet, not one per outlet.
        padded = pad_for_seeded_walk(self.fdir)

        results = []
        for i, (x, y) in enumerate(outlet_points_sorted):
            # snap="center" — "the cell this point falls in", which is what an outlet
            # coordinate means. pysheds defaults to "corner", i.e. the nearest grid
            # intersection, and resolves it with np.around. A cell centre is exactly
            # half a cell off a corner, so every seed lands on a .5 index and
            # banker's rounding decides by parity: row 150.5 floors to 150, row 151.5
            # rounds up to 152 — the wrong cell — and the bottom row, 299.5, rounds to
            # 300 and off the raster, returning no catchment at all. np.floor, which
            # "center" uses, is the only resolution that means what we asked.
            # Every one of these points is an *exit* point, which is to say it
            # sits on the rim of the grid by definition — and pysheds' catchment
            # kernel reads the neighbours of its seed with bounds checking off, so
            # a rim seed reads outside the array and kills the process. That is
            # what `catchment_from_seed` pads against; see its docstring. The
            # check here stays for the seed that is off the raster altogether,
            # which is a point to skip rather than an error to raise.
            seed_row, seed_col = xy_to_rc(self.transform, x, y)
            if not (0 <= seed_row < acc_array.shape[0]
                    and 0 <= seed_col < acc_array.shape[1]):
                continue

            try:
                catch_mask = catchment_from_seed(
                    self.grid, self.fdir, x, y, routing=self.routing,
                    snap="center", padded=padded,
                )
            except Exception:
                continue

            catch_bool = np.array(catch_mask).astype(bool)
            local_catch = catch_bool & ~claimed
            claimed |= catch_bool

            if not local_catch.any():
                continue

            local_uint8 = np.where(local_catch, 1, 0).astype("uint8")
            polys = [
                shapely_shape(geom)
                for geom, val in rasterio_shapes(local_uint8, transform=self.transform)
                if val == 1
            ]
            if not polys:
                continue

            catchment_poly = unary_union(polys)
            area_m2 = catchment_poly.area
            area_ha = area_m2 / 10000

            dominant_bearing = 0.0
            try:
                if self.routing == 'dinf':
                    safe_fdir = np.nan_to_num(fdir_array, nan=0.0)
                    bearings = (90.0 - np.degrees(safe_fdir)) % 360.0
                else:
                    # A lookup table indexed by the D8 code, not a dict read per
                    # cell. The codes are powers of two up to 128, so a 129-entry
                    # array covers them and anything else indexes to 0 — the same
                    # answer the dict's default gave.
                    d8_bearing = np.zeros(129, dtype="float64")
                    for code, deg in ((64, 0), (128, 45), (1, 90), (2, 135),
                                      (4, 180), (8, 225), (16, 270), (32, 315)):
                        d8_bearing[code] = deg
                    codes = np.clip(np.nan_to_num(fdir_array, nan=0.0),
                                    0, 128).astype("int64")
                    bearings = d8_bearing[codes]
                weights = acc_array * local_catch
                total_w = weights.sum()
                if total_w > 0:
                    sin_sum = np.sum(np.sin(np.radians(bearings)) * weights)
                    cos_sum = np.sum(np.cos(np.radians(bearings)) * weights)
                    dominant_bearing = float(np.degrees(np.arctan2(sin_sum, cos_sum)) % 360)
            except Exception:
                pass

            results.append({
                "id": i + 1,
                "geometry": catchment_poly,
                "area_m2": round(area_m2, 1),
                "area_ha": round(area_ha, 2),
                "flow_bearing": round(dominant_bearing, 1),
                "label": f"Catchment {i + 1}: {area_ha:.1f} ha",
                "_outlet_acc": _acc_at(x, y),
            })

        results.sort(key=lambda r: r["_outlet_acc"], reverse=True)
        for idx, r in enumerate(results):
            r["id"] = idx + 1
            r["label"] = f"Catchment {idx + 1}: {r['area_ha']:.1f} ha ({r['area_m2']:,.0f} m²)"

        return results

    def _find_boundary_outlets(self, acc_array, min_acc):
        """Find significant outlet points on grid boundary edge."""
        rows, cols = acc_array.shape
        cell_w = abs(self.transform.a)

        boundary_cells = set()
        for c in range(cols):
            boundary_cells.add((0, c))
            boundary_cells.add((rows - 1, c))
        for r in range(rows):
            boundary_cells.add((r, 0))
            boundary_cells.add((r, cols - 1))

        candidates = [
            (int(acc_array[r, c]), r, c)
            for r, c in boundary_cells if acc_array[r, c] >= min_acc
        ]
        candidates.sort(reverse=True)

        min_dist = cell_w * 20
        kept = []
        for acc, r, c in candidates:
            x = self.transform.c + (c + 0.5) * self.transform.a
            y = self.transform.f + (r + 0.5) * self.transform.e
            too_close = any(
                (x - kx) ** 2 + (y - ky) ** 2 < min_dist ** 2
                for _, kx, ky in kept
            )
            if not too_close:
                kept.append((acc, x, y))

        return [(x, y) for _, x, y in kept]

    def get_fdir_description(self):
        """Return band description for the flow direction raster."""
        if self.routing == 'dinf':
            return "D-infinity flow direction (angle in radians, CCW from east)"
        return "D8 flow direction (ESRI codes: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE)"

    def get_fdir_nodata(self):
        """No-data value for this instance's flow-direction raster.

        See :func:`fdir_nodata` — the rule lives there so the read side can share it.
        """
        return fdir_nodata(self.routing)

    def get_profile(self, dtype="float32", nodata=None):
        """rasterio write profile for result GeoTIFFs.

        ``nodata`` is written into the file rather than left off. An untagged
        GeoTIFF is not neutral: ``pysheds.io.read_raster`` falls back to ``0``
        when the file declares nothing, so a D-infinity direction raster comes
        back with every due-east cell marked no-data and routing through it
        collapses into a self-loop.
        """
        return {
            "driver": "GTiff", "dtype": dtype,
            "crs": self.crs, "transform": self.transform,
            "width": self.grid.shape[1], "height": self.grid.shape[0],
            "count": 1, "compress": "lzw", "nodata": nodata,
        }

    def save_result(self, array, output_path, band_description="", dtype="float32",
                    nodata=None):
        """Save a result array to GeoTIFF.

        float32 is right for every output whose values are metres of water or counts of
        cells. It is **wrong for the conditioned surface**, which carries a synthetic
        gradient across every flat: ``resolve_flats`` inflates a flat by integer multiples
        of ``eps = 1e-5`` m, and float32's spacing already reaches ~6.1e-5 m at 1000 m
        elevation. Above roughly 600 m the whole gradient is quantised away on the way to
        disk, and ``flow_graph.d8_from_dem`` — which needs a *strictly* positive drop —
        then reads a genuine flat and turns every cell of it into a ``LABEL_SINK``. Pass
        ``dtype="float64"`` for that raster; a site near sea level never shows the fault.

        ``nodata`` is likewise the caller's choice, because it differs per raster:
        NaN for the float measures, ``get_fdir_nodata()`` for the direction grid,
        the source DEM's sentinel for the conditioned surface, and None for a mask
        whose 0 is a real value. Leaving it off entirely is the one wrong answer —
        see ``get_profile``.
        """
        profile = self.get_profile(dtype, nodata=nodata)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)
            if band_description:
                dst.update_tags(1, description=band_description)


# ---------------------------------------------------------------------------
# QThread worker — re-exported from qgis/workers/ for backward compatibility
# ---------------------------------------------------------------------------

# AnalysisWorker has moved to terrainflow_assessment.qgis.workers.analysis_worker.
# This re-export keeps existing imports working without changes.
from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker  # noqa: F401, E402
