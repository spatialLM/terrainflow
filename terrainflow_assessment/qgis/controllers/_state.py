"""
_state.py — PluginState: single owner of all cross-controller mutable state.

All five controllers share one PluginState instance, so they can read each
other's results without direct controller-to-controller coupling.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginState:
    # ------------------------------------------------------------------ DEM / paths
    dem_path: str | None = None
    dem_info: Any | None = None           # DEMInfo from dem_loader
    boundary_path: str | None = None
    analysis_area_path: str | None = None
    earthworks_area_path: str | None = None
    modified_dem_path: str | None = None
    slope_raster_path: str | None = None
    output_dir: str = field(default_factory=lambda: tempfile.mkdtemp(prefix="tfa_"))
    ponding_raster_path: str | None = None

    # The arrays the verification pass already read off disk, handed forward to the
    # event-pond layers. Both views describe the same pools on the same grid, and
    # re-reading and re-aligning the rasters a second time is how they would come to
    # disagree — the baseline alignment in particular has three failure modes and a
    # fallback to zeros. None until an earthworks re-analysis has run.
    pond_context: dict | None = None

    # ------------------------------------------------------------------ Layer tree
    # Abbreviated run parameters ("120mm·24h·C0.40·1ha") shown on the stage groups.
    # Frozen when Baseline runs, not read live from the panel: Analysis and Design
    # outputs describe the storm that was actually routed, so nudging a spinner
    # afterwards must not retag groups full of layers built under the old numbers.
    run_tag: str = ""

    # ------------------------------------------------------------------ Analysis results
    baseline_result: dict | None = None
    earthworks_result: dict | None = None
    sim_result: dict | None = None

    # ------------------------------------------------------------------ Before/after layer IDs
    baseline_layer_ids: list[str] = field(default_factory=list)
    earthworks_layer_ids: list[str] = field(default_factory=list)
    slope_class_layer_id: str | None = None
    slope_vectors_layer_id: str | None = None

    # ------------------------------------------------------------------ Earthworks
    burner: Any | None = None             # DEMBurner instance
    earthwork_manager: Any | None = None  # EarthworkManager (set in controller __init__)
    # earthwork type → layer id. Named for what it holds: the values are ids
    # resolved through _layers.resolve_layer, never layer objects.
    ew_layer_ids: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ Flow-graph cache
    # The design-tier catchment labelling is storm-independent: it depends only on the
    # conditioned DEM (flow_next) and the earthwork geometry (catchment_labels). So the
    # pointers are rebuilt only when the DEM changes and the labels only when geometry
    # moves, which is what lets a storm-slider change re-score the whole site in ~30 ms
    # while a geometry edit costs ~0.3 s. Arrays, not layers — nothing to resolve.
    flow_next: Any | None = None            # int32 flat next-cell pointers
    flow_sink: Any | None = None            # bool flat sink mask
    flow_dem: Any | None = None             # float32 conditioned DEM (outlet/pour-point lookups)
    flow_domain_mask: Any | None = None     # bool (rows, cols) — the site
    flow_grid_meta: dict | None = None      # {shape, transform, cell_area_m2, cell_size_m}
    catchment_labels: Any | None = None     # int32 (rows, cols) — LabelResult.labels
    catchment_label_ids: list = field(default_factory=list)  # label index → earthwork id
    catchment_counts: dict = field(default_factory=dict)     # earthwork id → cell count
    # earthwork id → (own_peak_m3s, upstream_peak_m3s). Sizes spillways; distinct
    # from the volume-based catchment counts because a peak rate is a different
    # question from an event total.
    peak_flows: dict = field(default_factory=dict)
    # Depth-duration-frequency data the user looked up (HIRDS). None until entered —
    # it cannot be derived from terrain, so its absence is a real state, not a default.
    idf_table: object = None
    catchment_exit_cells: int = 0
    catchment_sink_cells: int = 0
    catchment_outlets: dict = field(default_factory=dict)    # earthwork id → flat outlet cell
    catchment_labels_layer_id: str | None = None
    throughflow_layer_id: str | None = None
    connections_layer_id: str | None = None
    spillway_layer_id: str | None = None
    stress_points_layer_id: str | None = None   # where features overtop locally

    # ------------------------------------------------------------------ Contour analysis
    # Result layers are held by ID (str), not object, so a deleted/swapped layer
    # resolves to None instead of a dead "wrapped C/C++ object" reference. See _layers.py.
    contour_features: list = field(default_factory=list)
    # The subset "Select Top Swales" last produced. The inflow gradient scopes itself
    # to this when set: grading every contour on the site answers a question nobody
    # asked once the user has narrowed to the swales they are actually considering.
    top_contour_features: list = field(default_factory=list)
    # The SwaleSegments "Find Best Swale Segments" last produced — kept so the
    # peak-inflow overlay can be toggled on without re-running the search.
    segment_features: list = field(default_factory=list)
    usable_polygon: Any | None = None
    contour_layer_id: str | None = None
    top5_layer_id: str | None = None
    segment_layer_id: str | None = None
    segment_gradient_layer_id: str | None = None
    simple_contour_layer_id: str | None = None
    inflow_bands_layer_id: str | None = None
    # Natural-breaks boundaries the candidate contours are currently banded on, so
    # the panel legend prints the same numbers the map is drawn with.
    contour_breaks: list = field(default_factory=list)

    # ------------------------------------------------------------------ Keypoint analysis
    found_keypoints: list | None = None
    keyline_analysis: Any | None = None
    keyline_layer_id: str | None = None
    drawn_keyline_layer_id: str | None = None
    # The current master keyline (generated or drawn) available for "convert to swale".
    keyline_master_geom: Any | None = None
    keyline_master_coords: list | None = None

    # ------------------------------------------------------------------ Simulation display
    sim_global_max_inc: float = 1.0
    sim_global_max_cum: float = 1.0
    sim_fill_layer_id: str | None = None
    sim_ew_centroids: dict = field(default_factory=dict)
    sim_ponding_capacity: Any | None = None
    sim_ponding_masks: dict = field(default_factory=dict)
    # The bed under the playback's water. Read once with the capacity raster, because
    # each frame solves for the level that holds the volume delivered so far and
    # re-reading the burned DEM twice a second is not a thing to do during playback.
    sim_ponding_ground: Any | None = None
    sim_ponding_meta: dict | None = None
    sim_ponding_frame_layer_id: str | None = None
    # The playback frame raster. Held by id like everything else, so
    # swapping it never depends on matching a layer by its name.
    sim_frame_layer_id: str | None = None
    sim_ponding_outline_layer_id: str | None = None

    # ------------------------------------------------------------------ Reporting
    baseline_report: Any | None = None
    post_report: Any | None = None
    comparison: Any | None = None
    verification: Any | None = None       # VerificationResult (terrain vs analytic, §4)
    # Design-tier results, retained for the report. The live assessment recomputes
    # these on every edit and used to drop them on the floor, which is the only
    # reason the report needed a simulation to exist: BalanceResult already carries
    # capture %, per-feature water, cut/fill and the mass-balance flag.
    # balance_stores is kept alongside because the stores hold per-feature
    # cut_vol_m3/fill_vol_m3 that BalanceResult.per_feature does not — recomputing
    # them invites the diversion width/bed-width trap in calculate_cut_volume.
    balance: Any | None = None            # BalanceResult (design tier, live)
    balance_stores: list | None = None    # EarthworkStore list it was built from
    # The overflow network the balance was routed through (RoutingResult from
    # resolve_targets, walked along real flow paths). Retained because the fill
    # simulation must cascade along the same links — routed separately, it produced a
    # second network that the comparative report then presented as one.
    balance_routing: Any | None = None
    # {cut_m3, fill_m3} the burn actually moved, measured off the two elevation
    # surfaces. Distinct from the analytic cut/fill above, which assumes flat ground —
    # see earthwork_design.burn_quantities.
    burn_quantities: dict | None = None
    # Spillway review rows are built for the panel table; the report needs the same
    # figures and cannot reach the earthworks controller to ask for them.
    spillway_rows: list | None = None
    spillway_context: dict | None = None
    report_last_path: str | None = None
    # Verified-vs-design tracking (Workbench scorecard chip). edits_since_verify is
    # None until the first burn; 0 right after a burn; incremented per design edit.
    verified_delta_pct: float | None = None
    edits_since_verify: int | None = None

    # ------------------------------------------------------------------ Workers (prevent GC)
    analysis_worker: Any | None = None
    sim_worker: Any | None = None
    # The heavy analyses that are not the baseline or the simulation, one slot per
    # controller so a contour run and a design burn can proceed at once but two of
    # either cannot. `keypoint_worker` was declared for this and never used; it is
    # renamed because the slot now carries contours, segments, keypoints and the
    # keyline, not keypoints alone.
    contour_worker: Any | None = None
    design_worker: Any | None = None

    # ------------------------------------------------------------------ Cache invalidation
    # Lives on the state rather than a controller so any controller can invalidate
    # without reaching across to another one (baseline invalidates what earthworks owns).

    def invalidate_catchment_cache(self):
        """Forget which cells drain to which earthwork (geometry changed)."""
        self.catchment_labels = None
        self.catchment_label_ids = []
        self.catchment_counts = {}
        self.peak_flows = {}
        self.catchment_exit_cells = 0
        self.catchment_sink_cells = 0
        self.catchment_outlets = {}

    def invalidate_flow_cache(self):
        """Forget the flow pointers too (DEM or baseline changed)."""
        self.flow_next = None
        self.flow_sink = None
        self.flow_dem = None
        self.flow_domain_mask = None
        self.flow_grid_meta = None
        self.invalidate_catchment_cache()

    def invalidate_results(self):
        """Drop every result derived from terrain. Call when the terrain changes.

        A design file's Open path has always done this — nothing computed against the
        previous DEM describes the new one, and a stale verification chip reading green
        for a design that has not been analysed once is the failure the format exists to
        avoid. Swapping the DEM in the picker is the same event and was not doing it, so
        a baseline computed on one grid survived alongside a burner built on another.
        That is not merely stale: the two rasters then have different extents, the
        subtraction that isolates earthwork storage is skipped, and every measured figure
        silently carries whatever ponded there naturally.

        State only — the panel half (buttons, summary labels) stays with the controller
        that owns the panel, so this can be called from any of them.
        """
        self.baseline_result = None
        self.earthworks_result = None
        self.sim_result = None
        self.baseline_report = None
        self.post_report = None
        self.comparison = None
        self.verification = None
        self.verified_delta_pct = None
        self.edits_since_verify = None
        self.balance = None
        self.balance_stores = None
        self.balance_routing = None
        self.burn_quantities = None
        self.spillway_rows = None
        self.spillway_context = None
        self.modified_dem_path = None
        self.pond_context = None
