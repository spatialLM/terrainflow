"""
simulation.py — Time-stepped fill simulation with cascading overflow.

Adapts the base plugin's SimulationWorker and adds:
  - Per-earthwork fill tracking (EarthworkStore)
  - Cascading overflow: when an earthwork overflows, surplus is routed to
    the next downslope earthwork (or exits the site)
  - Infiltration losses per timestep (USCS-derived rate × earthwork area)
  - Output: timestep table, earthwork summary, total site outflow

Key exports
-----------
EarthworkStore        — tracks fill state for one earthwork
SimulationWorker      — QThread worker
run_simulation()      — standalone function (for testing / non-GUI use)
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window

from terrainflow_assessment.modules.flow_analysis import fdir_nodata
from terrainflow_assessment.modules.footprint import xy_to_rc

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Earthwork fill state
# ---------------------------------------------------------------------------

@dataclass
class EarthworkStore:
    """Tracks the fill state of one earthwork over the simulation."""
    name: str
    ew_type: str
    capacity_m3: float
    area_m2: float              # footprint area for infiltration
    infiltration_rate_mm_hr: float = 4.0  # default Loam
    elevation: float = 0.0     # approximate centroid elevation (for cascade ordering)
    # False when the DEM could not supply a real elevation for this centroid — nodata,
    # NaN, a centroid outside the raster, or a read error. Such a store must be kept out
    # of the elevation heuristic entirely: a 0.0 (or −9999) stand-in sorts below every
    # real feature and silently makes that one store the whole site's overflow receiver.
    elevation_known: bool = True

    # Overflow linkage (carried from Earthwork; consumed by the future analytical cascade).
    # id — stable identity; overflow_target_id — user-intended recipient (None = infer).
    id: Optional[str] = None
    overflow_target_id: Optional[str] = None

    # The drawn capacity, carried alongside whichever basis ``capacity_m3`` is on, so the
    # readouts can show both without asking the earthwork again. Equal to ``capacity_m3``
    # whenever no terrain measurement was available.
    drawn_capacity_m3: float = 0.0
    # True when ``capacity_m3`` is a flood measurement rather than the drawn section — the
    # readouts say which, because the two differ by a factor of two on keyed swales and a
    # figure that size cannot be presented without saying where it came from.
    capacity_is_measured: bool = False

    # Direct contributing catchment — the cells whose runoff this feature is the FIRST
    # to intercept (from flow_graph.label_direct_catchments). Mutually exclusive across
    # features, so summing them never double-counts a shared hillside.
    direct_catchment_m2: float = 0.0
    # Flat raster index of the feature's lowest cell: where its overflow leaves from,
    # and the start of the downslope walk that resolves its routing target.
    outlet_flat: Optional[int] = None

    # Per-timestep accumulators (set externally each step)
    inflow_m3: float = 0.0
    stored_m3: float = 0.0
    overflowed: bool = False
    first_overflow_hr: Optional[float] = None
    total_overflow_m3: float = 0.0
    total_inflow_m3: float = 0.0
    total_infiltration_m3: float = 0.0      # actually credited to capture
    total_infiltration_potential_m3: float = 0.0  # what the soil could have taken
    peak_fill_pct: float = 0.0

    # Cut/fill volumes (populated from earthwork geometry, set before simulation)
    cut_vol_m3: float = 0.0
    fill_vol_m3: float = 0.0

    # Raster coordinates of the earthwork centroid (for inflow sampling)
    centroid_row: Optional[int] = None
    centroid_col: Optional[int] = None

    def step_infiltration(self, dt_hr):
        """Infiltration *potential* this timestep (m³) — rate × wetted area × time.

        A single steady-state rate for the whole event: the long-run ("final")
        infiltration capacity of the soil texture, not the much higher initial rate a
        dry profile accepts. Real infiltration decays from that initial rate toward
        the steady one as the profile wets (Horton / Green-Ampt), so using the final
        rate throughout is conservative early and optimistic late.

        There is deliberately **no saturation cut-off**: no soil-moisture store, no
        water table, no mounding. The only limit applied by the caller is the water
        actually present. Over a long event that over-credits soakage, which is why
        capture can be reported with infiltration excluded — see ``cascade_overflow``.
        """
        rate_m_hr = self.infiltration_rate_mm_hr / 1000.0
        return rate_m_hr * self.area_m2 * dt_hr


# ---------------------------------------------------------------------------
# Cascading overflow routing
# ---------------------------------------------------------------------------

def _find_downslope_store(store, all_stores):
    """
    Return the EarthworkStore to route this store's overflow into.

    Honours a user-specified overflow link first: if ``store.overflow_target_id``
    matches another store that is **downslope** (lower elevation — which keeps the
    single top→bottom cascade pass valid), route there. Otherwise fall back to the
    elevation heuristic: the highest store still below this one (the most direct
    downslope receiver). Returns None if no lower store exists (overflow exits site).

    A store whose elevation the DEM could not supply (``elevation_known`` False) takes
    no part in the heuristic — neither as a receiver nor as a router. Letting an
    unknown elevation stand in as 0.0 m put it below every real feature, so it became
    the receiver for the entire site.
    """
    if store.overflow_target_id is not None:
        linked = next(
            (
                s for s in all_stores
                if s is not store
                and s.id == store.overflow_target_id
                # An explicit link is honoured unless both elevations are known and
                # contradict it; an unverifiable link is still the user's stated intent.
                and not (s.elevation_known and store.elevation_known
                         and s.elevation >= store.elevation)
            ),
            None,
        )
        if linked is not None:
            return linked

    if not store.elevation_known:
        return None

    candidates = [
        s for s in all_stores
        if s is not store and s.elevation_known and s.elevation < store.elevation
    ]
    if not candidates:
        return None
    # Nearest in elevation = most direct downslope receiver
    return max(candidates, key=lambda s: s.elevation)


def _top_down_key(store):
    """Sort key for a top→bottom cascade pass: highest first, unknown elevations last.

    Used with ``reverse=True``, so ``elevation_known`` True sorts ahead of False.
    """
    return (store.elevation_known, store.elevation)


@dataclass
class RoutingResult:
    """Where every feature's overflow goes, and in what order to process them."""
    edges: dict          # {from_id: to_id or None}   None = leaves the site
    is_user: dict        # {from_id: bool}            True = honoured user override
    order: list          # feature ids, upstream first — the cascade order
    warnings: list       # human-readable notes about rejected links


def resolve_targets(stores, walker=None):
    """Resolve each feature's overflow target, honouring user links where legal.

    ``walker(store) -> target_id or None`` is injected by the caller (it owns the
    flow-direction raster): it follows the actual flow path from the feature's outlet
    cell until it reaches another earthwork or the site edge. That replaces the old
    elevation-only heuristic, which picked "the highest feature below this one"
    regardless of whether water could ever travel between them — two features on
    opposite sides of a ridge were routinely linked.

    A user's ``overflow_target_id`` wins over the walked target, but only if it does
    not create a cycle: cycles are detected with
    :func:`~terrainflow_assessment.modules.flow_graph.topological_order` and the
    offending override is demoted to its walked target (one at a time, so only the
    actual offender is undone), each demotion recorded in ``warnings``.

    Note the old "the link must be lower in elevation" test is deliberately gone —
    with real flow paths, centroid elevation is the wrong proxy: a large tilted basin's
    centroid can sit above a swale that genuinely drains into it.
    """
    from .flow_graph import topological_order

    by_id = {s.id: s for s in stores if s.id is not None}

    auto = {}
    for store in stores:
        if store.id is None:
            continue
        target = None
        if walker is not None:
            try:
                target = walker(store)
            except Exception:
                target = None
        if target not in by_id or target == store.id:
            target = None
        auto[store.id] = target

    edges, is_user = {}, {}
    for sid, walked in auto.items():
        wanted = by_id[sid].overflow_target_id
        if wanted is not None and wanted in by_id and wanted != sid:
            edges[sid], is_user[sid] = wanted, True
        else:
            edges[sid], is_user[sid] = walked, False

    warnings = []
    order, broken = topological_order(edges)
    for _ in range(len(edges) + 2):
        if not broken:
            break
        victim = next((sid for sid in broken if is_user.get(sid)), None)
        if victim is not None:
            warnings.append(
                f"'{by_id[victim].name}' overflows into a loop — that link would send "
                f"water back into itself. Using the natural downslope path instead."
            )
            edges[victim] = auto.get(victim)
            is_user[victim] = False
        else:
            victim = broken[0]
            warnings.append(
                f"'{by_id[victim].name}' sits in a circular flow path; its overflow is "
                f"treated as leaving the site."
            )
            edges[victim] = None
        order, broken = topological_order(edges)

    return RoutingResult(edges=edges, is_user=is_user, order=order, warnings=warnings)


def overflow_graph(stores, walker=None):
    """Resolve the overflow routing as a graph, for display (matches the cascade).

    Returns ``{from_id: (to_id_or_None, is_user_link)}`` for every store with an
    ``id``. Thin wrapper over :func:`resolve_targets` so the drawn network can never
    disagree with where the cascade actually sends water.
    """
    routing = resolve_targets(stores, walker=walker)
    return {sid: (tgt, routing.is_user.get(sid, False))
            for sid, tgt in routing.edges.items()}


def cascade_overflow(stores: list[EarthworkStore], time_hr: float,
                     dt_hr: float, routing: "RoutingResult" = None,
                     count_infiltration: bool = True) -> float:
    """
    Process one simulation timestep for all earthwork stores.

    Processes stores upstream-first:
    1. Add inflow and subtract infiltration losses.
    2. If stored > capacity: overflow to the downstream store, or off site.
    3. Update peak fill % and first overflow time.

    Parameters
    ----------
    stores : list of EarthworkStore
    time_hr : float — current simulation time (hours from start)
    dt_hr   : float — timestep duration (hours)
    routing : RoutingResult or None — flow-path routing from :func:`resolve_targets`.
        When None, falls back to the legacy elevation sort and
        :func:`_find_downslope_store`, so the time-stepped simulation keeps its
        existing behaviour until it is migrated separately.
    count_infiltration : when False, soakage is measured but **not** credited — every
        feature must hold its water as storage. Sizing then rests only on volume
        actually impounded, and whatever the ground takes is spare capacity rather
        than something the design depends on. Steady-state infiltration rates are
        hard to predict on real ground and the model has no saturation limit, so
        treating soakage as a bonus is the defensible way to size. The potential is
        still accumulated on each store, so the buffer can be reported.

    A zero-capacity feature (berm, diversion drain) is a **redirect** node, not a
    store: everything it intercepts immediately exceeds its zero capacity and is
    re-emitted to its downstream target. That is what those types physically do —
    convey or deflect water somewhere it would not otherwise have gone — and is why
    they must not be filtered out of the store list.

    Returns
    -------
    float — total volume that exited the site this timestep (m³)
    """
    if routing is not None:
        rank = {sid: i for i, sid in enumerate(routing.order)}
        sorted_stores = sorted(
            stores, key=lambda s: rank.get(s.id, len(rank)))
        by_id = {s.id: s for s in stores if s.id is not None}
    else:
        sorted_stores = sorted(stores, key=_top_down_key, reverse=True)
        by_id = None
    site_exit_m3 = 0.0

    for store in sorted_stores:
        potential = store.step_infiltration(dt_hr)
        available = store.stored_m3 + store.inflow_m3
        store.total_infiltration_potential_m3 += min(potential, available)
        infiltration = min(potential, available) if count_infiltration else 0.0

        store.stored_m3 += store.inflow_m3 - infiltration
        store.stored_m3 = max(0.0, store.stored_m3)
        store.total_inflow_m3 += store.inflow_m3
        store.total_infiltration_m3 += infiltration
        store.inflow_m3 = 0.0  # reset for next step

        # Check overflow
        if store.stored_m3 > store.capacity_m3:
            overflow = store.stored_m3 - store.capacity_m3
            store.stored_m3 = store.capacity_m3
            store.total_overflow_m3 += overflow
            if not store.overflowed:
                store.overflowed = True
                store.first_overflow_hr = time_hr

            # Route overflow to the downstream store, or off site.
            if routing is not None:
                target_id = routing.edges.get(store.id)
                downstream = by_id.get(target_id) if target_id is not None else None
            else:
                downstream = _find_downslope_store(store, stores)
            if downstream is not None:
                downstream.inflow_m3 += overflow
            else:
                site_exit_m3 += overflow

        fill_pct = (store.stored_m3 / store.capacity_m3 * 100.0) if store.capacity_m3 > 0 else 0.0
        if fill_pct > store.peak_fill_pct:
            store.peak_fill_pct = fill_pct

    return site_exit_m3



class CatchmentPartition:
    """Splits a runoff array between features by direct catchment.

    Each cell belongs to the feature that is the **first** to intercept its runoff
    (``flow_graph.label_direct_catchments``). Those catchments are mutually exclusive,
    so a share is never counted twice and there is nothing to subtract afterwards —
    which is the whole reason the simulation uses them rather than sampling the
    cumulative accumulation raster. See ``water_balance`` for the double-count this
    replaced.

    One ``np.bincount`` answers for every feature at once, so the cost per timestep is
    one pass over the domain regardless of how many features there are.
    """

    __slots__ = ("lab_flat", "inside", "label_index", "n_labels")

    def __init__(self, lab_flat, inside, label_index):
        self.lab_flat = lab_flat
        self.inside = inside
        self.label_index = label_index
        self.n_labels = len(label_index)

    def shares(self, values):
        """Per-label totals of ``values`` (any array shaped like the labelling)."""
        flat = np.asarray(values).ravel()
        if flat.shape != self.lab_flat.shape:
            raise ValueError(
                f"expected an array of {self.lab_flat.size} cells, got {flat.size}")
        return np.bincount(self.lab_flat[self.inside],
                           weights=flat[self.inside],
                           minlength=max(self.n_labels, 1))

    def credit(self, stores, values):
        """Add each store's share of ``values`` to its pending ``inflow_m3``."""
        if not self.n_labels:
            return
        share = self.shares(values)
        for store in stores:
            idx = self.label_index.get(store.id)
            if idx is not None and idx < share.size:
                store.inflow_m3 += max(0.0, float(share[idx]))
            # else: this feature intercepts nothing — no catchment, no inflow


def catchment_partition(catchment_labels, catchment_label_ids, shape):
    """Build a :class:`CatchmentPartition` over a labelling, checking it fits."""
    labels_arr = np.asarray(catchment_labels)
    if labels_arr.shape != tuple(shape):
        raise ValueError(
            f"catchment_labels is {labels_arr.shape}, but the grid is {tuple(shape)} — "
            "the labelling was built on a different DEM."
        )
    lab_flat = labels_arr.ravel()
    return CatchmentPartition(
        lab_flat=lab_flat,
        inside=lab_flat >= 0,
        label_index={sid: i for i, sid in enumerate(catchment_label_ids or [])},
    )


def layer_nodes(node_ids, edges):
    """Lay a flow network out in ranks — ``{id: (rank, order)}``.

    *rank* is the number of features water passes through before reaching this one:
    0 for anything nothing spills into, 1 for whatever those feed, and so on. *order*
    is the position within a rank, kept in the caller's own sequence so the chart does
    not reshuffle itself between refreshes when nothing has changed.

    Rank is the **longest** path from a source, not the shortest. A feature fed by
    both a first-rank swale and a third-rank dam belongs below the dam — drawing it
    beside the swale would show water flowing upwards.

    Cycle-safe. Nodes in a ring have no well-defined rank, so they are placed after
    everything acyclic rather than being allowed to loop forever; the caller already
    refuses to *create* a cycle, so this only has to avoid hanging on one that
    survives some other route.
    """
    from .flow_graph import topological_order

    ids = list(node_ids)
    known = set(ids)
    order, broken = topological_order({k: edges.get(k) for k in ids})
    cyclic = set(broken)

    rank = {k: 0 for k in ids}
    for node in order:
        if node in cyclic:
            continue
        target = edges.get(node)
        if target in known and target != node and target not in cyclic:
            rank[target] = max(rank[target], rank[node] + 1)

    # Ring members sit below everything that resolves, so an unresolvable link reads
    # as "downstream of the whole system" rather than silently mixing into it.
    if cyclic:
        floor = max((rank[k] for k in ids if k not in cyclic), default=-1) + 1
        for k in cyclic:
            rank[k] = floor

    seen = {}
    layout = {}
    for k in ids:                       # caller order, so the chart is stable
        r = rank[k]
        layout[k] = (r, seen.get(r, 0))
        seen[r] = seen.get(r, 0) + 1
    return layout


# ---------------------------------------------------------------------------
# QThread worker — re-exported from qgis/workers/ for backward compatibility
# ---------------------------------------------------------------------------

# SimulationWorker has moved to terrainflow_assessment.qgis.workers.simulation_worker.
# This re-export keeps existing imports working without changes.
from terrainflow_assessment.qgis.workers.simulation_worker import (  # noqa: E402
    SimulationWorker,  # noqa: F401
)

# ---------------------------------------------------------------------------
# Core simulation logic
# ---------------------------------------------------------------------------

def _fdir_nodata(fdir_path, routing):
    """No-data value to read ``fdir_path`` with — never left to pysheds to guess.

    Prefers what the file itself declares, which since the writer started tagging
    them is the right answer by construction. The routing-derived fallback covers a
    raster written before that, where the alternative is pysheds' own default of 0
    — the value that means "due east" under D-infinity.
    """
    try:
        with rasterio.open(fdir_path) as src:
            if src.nodata is not None:
                return src.nodata
    except Exception:
        pass
    return fdir_nodata(routing)


def _run_simulation(dem_path, fdir_path, output_dir, cn, moisture,
                    rainfall_data, routing='dinf', cn_zones_data=None,
                    earthwork_stores=None, progress_callback=None,
                    catchment_labels=None, catchment_label_ids=None,
                    store_routing=None):
    """
    Run the time-stepped simulation and return results.

    Parameters and return value match SimulationWorker.completed signal.
    Can also be called directly (without QThread) for testing.

    ``routing`` is the *raster* routing scheme (``'d8'`` / ``'dinf'``) — the flow
    grid's own. The **overflow** network between features is ``store_routing``,
    a ``RoutingResult`` from :func:`resolve_targets`; the two are unrelated and the
    similar names are historical.

    Feature inflow: the design tier and the simulation must answer the same question
    with the same partition, or the comparative report shows two networks as one.
    Both therefore split runoff by ``flow_graph.label_direct_catchments`` —
    ``catchment_labels`` here — where each cell is credited to the feature that
    *first* intercepts it. Those catchments are mutually exclusive, so summing them
    never double-counts a shared hillside and no interception correction is needed.

    That correction is the reason for the hard requirement below. The previous model
    sampled the **cumulative** accumulation raster at each feature's centroid, which
    counts every upstream feature's catchment again, and then unpicked it by
    subtracting upstream capture ranked purely on centroid elevation — while
    ``cascade_overflow`` added the upstream overflow back on top, so pass-through
    water was double-counted regardless. ``water_balance.py`` removed exactly that
    correction for exactly that reason. There is no correct fallback without a
    labelling, so a store list arrives with one or the call fails.
    """
    from pysheds.grid import Grid

    from .catchment import SCSRunoff

    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    earthwork_stores = earthwork_stores or []
    scs = SCSRunoff()

    if earthwork_stores and catchment_labels is None:
        raise ValueError(
            "The simulation needs the direct-catchment labelling to split runoff "
            "between features. Run the design analysis first."
        )

    _p(2, "Loading DEM for simulation...")
    with rasterio.open(dem_path) as src:
        cell_w = abs(src.transform.a)
        cell_h = abs(src.transform.e)
        cell_area_m2 = cell_w * cell_h
        shape = (src.height, src.width)
        out_meta = src.meta.copy()
        out_meta.update(dtype="float32", count=1, nodata=-9999.0)

    _p(5, "Loading flow direction...")
    grid = Grid.from_raster(dem_path)
    # Explicit, not inferred. pysheds falls back to nodata=0 for an untagged file,
    # and under D-infinity 0.0 rad is "due east" — so every east-flowing cell would
    # be read as no-data and route to itself, and the simulation would disagree with
    # the flow map it is supposed to be replaying.
    fdir = grid.read_raster(fdir_path, nodata=_fdir_nodata(fdir_path, routing))

    _p(8, "Building CN raster...")
    if cn_zones_data:
        from shapely.wkt import loads as wkt_loads
        with rasterio.open(dem_path) as src:
            transform = src.transform
        zone_geoms_cn = []
        for z in cn_zones_data:
            try:
                zone_geoms_cn.append((wkt_loads(z["wkt"]), z["cn"]))
            except Exception:
                pass
        cn_array = scs.build_cn_raster(
            shape, transform, zone_geoms_cn, cn, moisture
        )
    else:
        eff_cn = scs.adjust_cn(cn, moisture)
        cn_array = np.full(shape, eff_cn, dtype="float32")

    sim_dir = os.path.join(output_dir, "simulation")
    os.makedirs(sim_dir, exist_ok=True)
    for f in os.listdir(sim_dir):
        try:
            os.remove(os.path.join(sim_dir, f))
        except OSError:
            pass

    n_steps = len(rainfall_data) - 1
    if n_steps < 1:
        raise ValueError("rainfall_data must have at least 2 entries.")

    partition = (catchment_partition(catchment_labels, catchment_label_ids, shape)
                 if earthwork_stores else None)

    frames = []
    peak_arr = np.zeros(shape, dtype="float32")
    cum_acc = np.zeros(shape, dtype="float64")
    prev_q_mm = np.zeros(shape, dtype="float64")

    # Timestep table: one row per timestep
    timestep_table = []
    total_site_outflow_m3 = 0.0
    peak_outflow_ls = 0.0
    peak_outflow_time_hr = 0.0

    # Track outflow at the main boundary exit (approximated from raster)
    # We'll estimate it from the incremental accumulation at the highest-acc
    # boundary cell if no earthwork stores are provided.

    for i in range(n_steps):
        time_min = rainfall_data[i + 1][0]
        time_hr = time_min / 60.0
        p_cum = float(rainfall_data[i + 1][1])

        pct = int(10 + 80 * i / n_steps)
        _p(pct, f"Timestep {i + 1}/{n_steps} — T = {time_min} min...")

        # SCS incremental runoff this step
        q_cum_now = scs.runoff_depth_array(p_cum, cn_array)
        dq_mm = np.maximum(0.0, q_cum_now - prev_q_mm)
        prev_q_mm = q_cum_now

        dq_m3 = (dq_mm / 1000.0) * cell_area_m2

        try:
            from pysheds.sview import Raster as _PR
            dq_raster = _PR(dq_m3.astype("float64"), viewfinder=fdir.viewfinder)
        except Exception:
            dq_raster = dq_m3.astype("float64")

        try:
            inc_acc = grid.accumulation(fdir, weights=dq_raster, routing=routing)
        except TypeError:
            inc_acc = grid.accumulation(fdir, weights=dq_raster)
        inc_acc = np.array(inc_acc, dtype="float32")

        cum_acc += inc_acc.astype("float64")
        cum_arr = cum_acc.astype("float32")

        # Estimate site-level outflow this step from the DOMAIN-WIDE maximum
        # accumulation — not, despite an earlier comment here, a boundary cell. On a
        # DEM clipped to the site the two coincide; on a larger DEM the maximum can sit
        # inside the domain, so this is an upper bound on what leaves. A rough proxy
        # either way — the detailed per-earthwork cascade handles the rest.
        inc_max = float(inc_acc.max())
        dt_hr = (time_hr - (rainfall_data[i][0] / 60.0)) if i > 0 else time_hr
        dt_s = max(dt_hr * 3600.0, 1.0)
        outflow_ls_raster = inc_max * 1000.0 / dt_s  # L/s proxy

        # Per-earthwork cascade
        # Inject catchment runoff into stores based on their catchment fraction
        total_runoff_m3_step = float(np.sum(dq_m3))

        if earthwork_stores:
            # Each feature takes the runoff from the cells it is the FIRST to
            # intercept. Mutually exclusive by construction, so there is nothing to
            # subtract afterwards — the upstream feature's share simply is not in the
            # downstream feature's total, and whatever the upstream one cannot hold
            # arrives below as overflow through the cascade, once.
            if partition is not None:
                partition.credit(earthwork_stores, dq_m3)

            step_exit = cascade_overflow(
                stores=earthwork_stores,
                time_hr=time_hr,
                dt_hr=dt_hr,
                routing=store_routing,
            )
            total_site_outflow_m3 += step_exit
            outflow_ls = step_exit * 1000.0 / dt_s
        else:
            outflow_ls = outflow_ls_raster
            total_site_outflow_m3 += inc_max  # approximate

        if outflow_ls > peak_outflow_ls:
            peak_outflow_ls = outflow_ls
            peak_outflow_time_hr = time_hr

        # Baseline (no-storage) outflow: all runoff exits immediately this step
        outflow_ls_baseline = total_runoff_m3_step * 1000.0 / dt_s if dt_s > 0 else 0.0

        # Build timestep row
        row = {
            "time_min": time_min,
            "time_hr": round(time_hr, 2),
            "rainfall_cum_mm": round(p_cum, 1),
            "runoff_m3": round(total_runoff_m3_step, 1),
            "outflow_ls": round(outflow_ls, 1),
            "outflow_ls_baseline": round(outflow_ls_baseline, 1),
        }
        for store in earthwork_stores:
            fill_pct = (store.stored_m3 / store.capacity_m3 * 100.0) if store.capacity_m3 > 0 else 0.0
            row[f"{store.name}_fill_pct"] = round(fill_pct, 1)
            row[f"{store.name}_overflow"] = store.overflowed
        timestep_table.append(row)

        # Save rasters. NaN is masked as well as negatives: `data < 0` is False
        # for NaN, so a NaN went into the frame, was missed again by `== -9999`
        # on the way back in, and np.maximum then propagated it through the whole
        # peak raster.
        inc_path = os.path.join(sim_dir, f"inc_{i:03d}.tif")
        cum_path = os.path.join(sim_dir, f"cum_{i:03d}.tif")
        inc_out = np.where(np.isfinite(inc_acc) & (inc_acc >= 0),
                           inc_acc, out_meta["nodata"]).astype("float32")
        with rasterio.open(inc_path, "w", **out_meta) as dst:
            dst.write(inc_out, 1)
        with rasterio.open(cum_path, "w", **out_meta) as dst:
            dst.write(np.where(np.isfinite(cum_arr) & (cum_arr >= 0),
                               cum_arr, out_meta["nodata"]).astype("float32"), 1)

        # The peak is a running maximum over what we already have in hand. Reading
        # every frame back off disk afterwards was a second full pass over the
        # whole simulation for a number computed here for free.
        np.maximum(peak_arr, np.where(inc_out == out_meta["nodata"], 0.0, inc_out),
                   out=peak_arr)

        # Snapshot fill state for each store this frame.
        #
        # Keyed by `store.id`, not `store.name`. The default earthwork name counts all
        # features, so deleting one and drawing another reproduces a name already in
        # use — and two stores sharing a key means one of them silently overwrites the
        # other's fill state in every frame. The display name travels in the value, so
        # a caller drawing a label never has to key by it.
        frame_fills = {}
        for store in earthwork_stores:
            fill_pct = (store.stored_m3 / store.capacity_m3 * 100.0) if store.capacity_m3 > 0 else 0.0
            frame_fills[store.id or store.name] = {
                "name": store.name,
                "fill_pct": round(min(fill_pct, 100.0), 1),
                "overflowed": store.overflowed,
                "first_overflow_this_step": (
                    store.overflowed and
                    store.first_overflow_hr is not None and
                    abs(store.first_overflow_hr - time_hr) < (dt_hr + 1e-6)
                ),
            }
        frames.append({"time_min": time_min, "inc": inc_path, "cum": cum_path,
                        "fills": frame_fills})

    # Peak flow raster — accumulated in the timestep loop above.
    _p(96, "Writing peak flow raster...")
    peak_path = os.path.join(sim_dir, "peak_flow.tif")
    with rasterio.open(peak_path, "w", **out_meta) as dst:
        peak_arr[peak_arr < 0] = out_meta["nodata"]
        dst.write(peak_arr, 1)

    # Build earthwork summary
    earthwork_summary = []
    for store in earthwork_stores:
        fill_pct = (store.stored_m3 / store.capacity_m3 * 100.0) if store.capacity_m3 > 0 else 0.0
        earthwork_summary.append({
            # Carried alongside the name so consumers can join on identity rather
            # than on a label two features can share.
            "id": store.id,
            "name": store.name,
            "type": store.ew_type,
            "capacity_m3": round(store.capacity_m3, 1),
            "stored_m3": round(store.stored_m3, 1),
            "peak_fill_pct": round(store.peak_fill_pct, 1),
            "final_fill_pct": round(fill_pct, 1),
            "overflowed": store.overflowed,
            "first_overflow_hr": round(store.first_overflow_hr, 2) if store.first_overflow_hr else None,
            "total_overflow_m3": round(store.total_overflow_m3, 1),
            "total_inflow_m3": round(store.total_inflow_m3, 1),
            "total_infiltration_m3": round(store.total_infiltration_m3, 1),
            "cut_vol_m3": round(store.cut_vol_m3, 1),
            "fill_vol_m3": round(store.fill_vol_m3, 1),
        })

    # Time labels for UI slider
    time_labels = [f"{row['time_min']} min" for row in timestep_table]

    _p(100, "Simulation complete.")
    return {
        "frames": frames,
        "peak_path": peak_path,
        "timestep_table": timestep_table,
        "time_labels": time_labels,
        "earthwork_summary": earthwork_summary,
        "total_outflow_m3": round(total_site_outflow_m3, 1),
        "peak_outflow_ls": round(peak_outflow_ls, 1),
        "peak_outflow_time_hr": round(peak_outflow_time_hr, 2),
    }


# ---------------------------------------------------------------------------
# Helper: build EarthworkStore list from plugin earthworks
# ---------------------------------------------------------------------------

def build_stores_from_earthworks(earthworks, soil_name="Loam", dem_path=None,
                                 basis="terrain"):
    """
    Build a list of EarthworkStore objects from the plugin's Earthwork list.

    Parameters
    ----------
    earthworks : list of Earthwork (from earthwork_design module)
    soil_name : str — global soil type for infiltration rates
    dem_path : str or None — used to look up centroid elevation
    basis : ``"terrain"`` (default) sizes each store by what it impounds on the actual
        ground; ``"drawn"`` forces the cross-section figure. The second exists so the
        two can be *compared* — running the same storm both ways is what shows whether
        capacity is the constraint at all, and on the Quail Island design it is not:
        the score is 57% either way because 7,567 m³ of the 17,537 m³ storm never
        reaches a feature.

    Returns
    -------
    list of EarthworkStore
    """
    import json

    from shapely.geometry import shape as shapely_shape

    from .earthwork_design import calculate_cut_volume, calculate_fill_volume
    from .swale_design import get_infiltration_rate

    site_rate = get_infiltration_rate(soil_name)
    stores = []

    for ew in earthworks:
        # Zero-capacity features are kept deliberately: a berm or diversion drain has
        # no storage but is very much part of the routing, and filtering them here was
        # why they always rendered "leaves site" with no downstream link.
        if not ew.enabled:
            continue

        # Get footprint area from shapely geometry. Every shapely geometry HAS an
        # `.area` attribute — a LineString's is simply 0.0 — so testing for the
        # attribute made the line branch unreachable and gave swales, diversions and
        # berms (all polylines) a zero wetted footprint: no infiltration, no drain-down.
        # Test the value, not the attribute.
        try:
            shapely_geom = shapely_shape(json.loads(ew.geometry.asJson()))
            area_m2 = float(getattr(shapely_geom, "area", 0.0) or 0.0)
            if area_m2 <= 0:
                # Linear feature: its wetted footprint is length × top width.
                area_m2 = float(shapely_geom.length) * float(ew.width or 0.0)
        except Exception:
            shapely_geom = None
            area_m2 = 100.0  # fallback
        if area_m2 <= 0:
            area_m2 = 100.0

        # Centroid elevation + raster coordinates from DEM. `elevation_known` stays
        # False unless the DEM actually yields a finite, non-nodata value: a stand-in
        # 0.0 m sorts below every real feature and turns this store into the site's
        # universal overflow receiver (and every other store into its "upstream").
        elevation = 0.0
        elevation_known = False
        centroid_row = None
        centroid_col = None
        if dem_path and shapely_geom is not None:
            try:
                centroid = shapely_geom.centroid
                import rasterio
                with rasterio.open(dem_path) as src:
                    t = src.transform
                    row, col = xy_to_rc(t, centroid.x, centroid.y)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        centroid_row = row
                        centroid_col = col
                        # One cell through a window, not the whole band. This read
                        # the entire DEM to sample a single elevation, once per
                        # feature: about 410 MB of I/O for a design of 36.
                        value = float(src.read(
                            1, window=Window(col, row, 1, 1))[0, 0])
                        is_nodata = (
                            src.nodata is not None
                            and math.isclose(value, float(src.nodata), rel_tol=1e-9,
                                             abs_tol=1e-6)
                        )
                        if math.isfinite(value) and not is_nodata:
                            elevation = value
                            elevation_known = True
            except Exception:
                pass

        cut_vol = calculate_cut_volume(ew.type, ew.geometry, ew.depth, ew.width)
        fill_vol = calculate_fill_volume(ew.type, ew.geometry, ew.depth, ew.width,
                                         ew.companion_berm)

        # A fill-only feature (berm, dam wall) is built ground, not an excavated wetted
        # surface — it infiltrates nothing, so crediting it soakage would invent capture.
        try:
            from terrainflow_assessment.core.registry.earthwork_types import get_type
            wets_soil = bool(get_type(ew.type).has_cut)
        except Exception:
            wets_soil = True

        # A per-feature soil overrides the site default; None inherits it.
        own_soil = getattr(ew, "soil_name", None)
        infil_rate = get_infiltration_rate(own_soil) if own_soil else site_rate

        # **The one place the capacity basis is chosen.** Everything downstream — the
        # balance, the simulation, the flow-network nodes, the scorecard and the report —
        # reads capacity through this store, so switching basis here switches it
        # everywhere and cannot be switched inconsistently anywhere.
        #
        # The terrain measurement wins where there is one. Sizing against the drawn
        # section instead means a keyed swale reports "full at this storm" while most of
        # its pond is still empty — Swale 5 of the Quail Island design reads 100% of
        # 440 m³ against a pond of 1,095 m³ — and the designer enlarges a feature that
        # needed nothing. The drawn figure travels alongside rather than being discarded:
        # it is the one that can be checked by hand and the one a contractor builds to.
        drawn = float(getattr(ew, "capacity_m3", 0.0) or 0.0)
        terrain = getattr(ew, "terrain_capacity_m3", None)
        measured = (basis == "terrain" and terrain is not None and float(terrain) > 0)

        store = EarthworkStore(
            name=ew.name,
            ew_type=ew.type,
            capacity_m3=float(terrain) if measured else drawn,
            drawn_capacity_m3=drawn,
            capacity_is_measured=measured,
            area_m2=area_m2,
            infiltration_rate_mm_hr=infil_rate if wets_soil else 0.0,
            elevation=elevation,
            elevation_known=elevation_known,
            cut_vol_m3=cut_vol,
            fill_vol_m3=fill_vol,
            centroid_row=centroid_row,
            centroid_col=centroid_col,
            id=getattr(ew, "id", None),
            overflow_target_id=getattr(ew, "overflow_target_id", None),
        )
        stores.append(store)

    return stores
