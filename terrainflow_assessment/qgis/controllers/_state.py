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

    # ------------------------------------------------------------------ Analysis results
    baseline_result: dict | None = None
    earthworks_result: dict | None = None
    sim_result: dict | None = None

    # ------------------------------------------------------------------ Before/after layer IDs
    baseline_layer_ids: list[str] = field(default_factory=list)
    earthworks_layer_ids: list[str] = field(default_factory=list)
    accumulation_layer_id: str | None = None
    slope_class_layer_id: str | None = None
    slope_arrows_layer_id: str | None = None
    slope_vectors_layer_id: str | None = None

    # ------------------------------------------------------------------ Earthworks
    burner: Any | None = None             # DEMBurner instance
    earthwork_manager: Any | None = None  # EarthworkManager (set in controller __init__)
    ew_layers: dict = field(default_factory=dict)
    ew_group: Any | None = None

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
    stress_points_layer_id: str | None = None
    spillway_layer_id: str | None = None
    stress_points_layer_id: str | None = None   # where features overtop locally

    # ------------------------------------------------------------------ Contour analysis
    # Result layers are held by ID (str), not object, so a deleted/swapped layer
    # resolves to None instead of a dead "wrapped C/C++ object" reference. See _layers.py.
    contour_features: list = field(default_factory=list)
    usable_polygon: Any | None = None
    contour_layer_id: str | None = None
    top5_layer_id: str | None = None
    segment_layer_id: str | None = None
    simple_contour_layer_id: str | None = None
    inflow_bands_layer_id: str | None = None

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
    sim_ponding_meta: dict | None = None
    sim_ponding_frame_layer_id: str | None = None
    sim_ponding_outline_layer_id: str | None = None

    # ------------------------------------------------------------------ Reporting
    baseline_report: Any | None = None
    post_report: Any | None = None
    comparison: Any | None = None
    verification: Any | None = None       # VerificationResult (terrain vs analytic, §4)
    # Verified-vs-design tracking (Workbench scorecard chip). edits_since_verify is
    # None until the first burn; 0 right after a burn; incremented per design edit.
    verified_delta_pct: float | None = None
    edits_since_verify: int | None = None

    # ------------------------------------------------------------------ Workers (prevent GC)
    analysis_worker: Any | None = None
    sim_worker: Any | None = None
    keypoint_worker: Any | None = None

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
