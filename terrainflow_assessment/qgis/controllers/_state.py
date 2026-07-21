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

    # ------------------------------------------------------------------ Earthworks
    burner: Any | None = None             # DEMBurner instance
    earthwork_manager: Any | None = None  # EarthworkManager (set in controller __init__)
    ew_layers: dict = field(default_factory=dict)
    ew_group: Any | None = None

    # ------------------------------------------------------------------ Contour analysis
    contour_features: list = field(default_factory=list)
    usable_polygon: Any | None = None
    contour_layer: Any | None = None
    top5_layer: Any | None = None
    segment_layer: Any | None = None
    simple_contour_layer: Any | None = None

    # ------------------------------------------------------------------ Keypoint analysis
    found_keypoints: list | None = None
    keyline_analysis: Any | None = None

    # ------------------------------------------------------------------ Simulation display
    sim_global_max_inc: float = 1.0
    sim_global_max_cum: float = 1.0
    sim_fill_layer: Any | None = None
    sim_ew_centroids: dict = field(default_factory=dict)
    sim_ponding_capacity: Any | None = None
    sim_ponding_masks: dict = field(default_factory=dict)
    sim_ponding_meta: dict | None = None
    sim_ponding_frame_layer: Any | None = None
    sim_ponding_outline_layer: Any | None = None

    # ------------------------------------------------------------------ Reporting
    baseline_report: Any | None = None
    post_report: Any | None = None
    comparison: Any | None = None

    # ------------------------------------------------------------------ Workers (prevent GC)
    analysis_worker: Any | None = None
    sim_worker: Any | None = None
    keypoint_worker: Any | None = None
