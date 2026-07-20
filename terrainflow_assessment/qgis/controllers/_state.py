"""
_state.py — PluginState: single owner of all cross-controller mutable state.

All five controllers share one PluginState instance, so they can read each
other's results without direct controller-to-controller coupling.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PluginState:
    # ------------------------------------------------------------------ DEM / paths
    dem_path: Optional[str] = None
    dem_info: Optional[Any] = None           # DEMInfo from dem_loader
    boundary_path: Optional[str] = None
    analysis_area_path: Optional[str] = None
    earthworks_area_path: Optional[str] = None
    modified_dem_path: Optional[str] = None
    slope_raster_path: Optional[str] = None
    output_dir: str = field(default_factory=lambda: tempfile.mkdtemp(prefix="tfa_"))
    ponding_raster_path: Optional[str] = None

    # ------------------------------------------------------------------ Analysis results
    baseline_result: Optional[Dict] = None
    earthworks_result: Optional[Dict] = None
    sim_result: Optional[Dict] = None

    # ------------------------------------------------------------------ Before/after layer IDs
    baseline_layer_ids: List[str] = field(default_factory=list)
    earthworks_layer_ids: List[str] = field(default_factory=list)
    accumulation_layer_id: Optional[str] = None
    slope_class_layer_id: Optional[str] = None
    slope_arrows_layer_id: Optional[str] = None

    # ------------------------------------------------------------------ Earthworks
    burner: Optional[Any] = None             # DEMBurner instance
    earthwork_manager: Optional[Any] = None  # EarthworkManager (set in controller __init__)
    ew_layers: Dict = field(default_factory=dict)
    ew_group: Optional[Any] = None

    # ------------------------------------------------------------------ Contour analysis
    contour_features: List = field(default_factory=list)
    usable_polygon: Optional[Any] = None
    contour_layer: Optional[Any] = None
    top5_layer: Optional[Any] = None
    segment_layer: Optional[Any] = None
    simple_contour_layer: Optional[Any] = None

    # ------------------------------------------------------------------ Keypoint analysis
    found_keypoints: Optional[List] = None
    keyline_analysis: Optional[Any] = None

    # ------------------------------------------------------------------ Simulation display
    sim_global_max_inc: float = 1.0
    sim_global_max_cum: float = 1.0
    sim_fill_layer: Optional[Any] = None
    sim_ew_centroids: Dict = field(default_factory=dict)
    sim_ponding_capacity: Optional[Any] = None
    sim_ponding_masks: Dict = field(default_factory=dict)
    sim_ponding_meta: Optional[Dict] = None
    sim_ponding_frame_layer: Optional[Any] = None
    sim_ponding_outline_layer: Optional[Any] = None

    # ------------------------------------------------------------------ Reporting
    baseline_report: Optional[Any] = None
    post_report: Optional[Any] = None
    comparison: Optional[Any] = None

    # ------------------------------------------------------------------ Workers (prevent GC)
    analysis_worker: Optional[Any] = None
    sim_worker: Optional[Any] = None
    keypoint_worker: Optional[Any] = None
