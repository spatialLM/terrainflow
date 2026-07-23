"""
reporting.py — Before/after comparison and HTML report export.

Provides:
  BaselineReport       — data from a baseline (no earthworks) analysis
  PostInterventionReport — data from a post-earthworks analysis + simulation
  ComparisonResult     — computed before/after metrics
  compare()            — compute comparison metrics
  export_html()        — generate a self-contained HTML report with embedded charts
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report data containers
# ---------------------------------------------------------------------------

@dataclass
class BaselineReport:
    """Results from the baseline (no earthworks) analysis run."""
    site_name: str = "Unnamed Site"
    dem_path: str = ""
    crs: str = ""
    cell_size_m: float = 1.0
    catchment_area_ha: float = 0.0
    rainfall_mm: float = 0.0
    duration_hr: float = 1.0
    cn: float = 70.0
    runoff_mm: float = 0.0
    total_runoff_m3: float = 0.0         # total runoff generated
    exit_volume_m3: float = 0.0          # total volume exiting the site
    peak_outflow_ls: float = 0.0         # peak flow rate at exit (L/s)
    peak_outflow_time_hr: float = 0.0    # time of peak (hr from storm start)
    exit_points: list[dict] = field(default_factory=list)
    # Timestep series for hydrograph
    timestep_table: list[dict] = field(default_factory=list)


@dataclass
class PostInterventionReport:
    """Results from the post-earthworks analysis + fill simulation."""
    exit_volume_m3: float = 0.0
    peak_outflow_ls: float = 0.0
    peak_outflow_time_hr: float = 0.0
    total_infiltrated_m3: float = 0.0
    earthwork_summary: list[dict] = field(default_factory=list)
    timestep_table: list[dict] = field(default_factory=list)
    exit_points: list[dict] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Non-circular check: terrain-derived ponding vs analytic capacity (spec §4)."""
    analytic_total_m3: float = 0.0       # Σ analytic capacity over storage features
    terrain_total_m3: float = 0.0        # Σ(earthworks ponding − baseline ponding), floored ≥0
    delta_m3: float = 0.0                # terrain − analytic
    delta_pct: float = 0.0               # delta as % of analytic (0 when analytic == 0)
    unattributed_m3: float = 0.0         # terrain ponding not tied to any feature footprint
    per_feature: list[dict] = field(default_factory=list)  # {name, analytic_m3, terrain_m3|None, delta_pct|None, routing_only}
    caveats: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Computed before/after metrics."""
    captured_pct: float = 0.0           # % of runoff now retained on-site
    exit_reduction_pct: float = 0.0     # % reduction in exit volume
    peak_reduction_pct: float = 0.0     # % reduction in peak outflow rate
    peak_delay_hr: float = 0.0          # hours by which peak is delayed
    baseline: Optional[BaselineReport] = None
    post: Optional[PostInterventionReport] = None
    net_cut_m3: float = 0.0             # total soil excavated
    net_fill_m3: float = 0.0            # total material placed
    net_cut_fill_m3: float = 0.0        # cut - fill (positive = net cut)
    verification: Optional[VerificationResult] = None  # terrain-vs-analytic (§4)


# ---------------------------------------------------------------------------
# Non-circular verification — terrain-derived ponding vs analytic capacity (§4)
# ---------------------------------------------------------------------------

def raster_ponding_volume(ponding, cell_area_m2, min_depth=0.001):
    """Total ponded volume (m³) from a depth raster = Σ depth·cell_area over ponded cells."""
    import numpy as np
    arr = np.asarray(ponding, dtype="float64")
    ponded = np.where(arr >= min_depth, arr, 0.0)
    return float(ponded.sum() * cell_area_m2)


def impounded_volume(baseline_ponding, dammed_ponding, cell_area_m2):
    """Volume (m³) a dam impounds = the *new* ponding it creates on the DEM.

    ``Σ max(dammed − baseline, 0) × cell_area`` over the two ponding-depth rasters
    (before/after burning the dam to its crest). The positive clip counts only cells
    the dam newly floods; a wall too short to hold water yields ~0. Pure and testable.
    """
    import numpy as np
    dammed = np.asarray(dammed_ponding, dtype="float64")
    baseline = np.asarray(baseline_ponding, dtype="float64")
    new_ponding = np.clip(dammed - baseline, 0.0, None)
    return float(new_ponding.sum() * cell_area_m2)


# ---------------------------------------------------------------------------
# Live Assessment panel readout (design tier)
# ---------------------------------------------------------------------------

# Capture-% traffic light: green ≥ 80, amber ≥ 40, red below.
_LIVE_GOOD, _LIVE_MID, _LIVE_BAD = "#1e8449", "#b9770e", "#c0392b"
_LIVE_MUTED = "#566573"
_LIVE_FILL = "#2e86c1"


def _capture_colour(pct):
    if pct >= 80:
        return _LIVE_GOOD
    if pct >= 40:
        return _LIVE_MID
    return _LIVE_BAD


def _mini_bar(pct, colour, back="#d6dbdf"):
    """A thin horizontal bar as a Qt-rich-text table (QLabel has no CSS widths)."""
    p = int(max(0.0, min(100.0, pct)))
    cells = []
    if p > 0:
        cells.append(
            f"<td bgcolor='{colour}' width='{p}%'>"
            "<span style='font-size:3px;'>&nbsp;</span></td>"
        )
    if p < 100:
        cells.append(
            f"<td bgcolor='{back}' width='{100 - p}%'>"
            "<span style='font-size:3px;'>&nbsp;</span></td>"
        )
    return (
        "<table width='100%' cellspacing='0' cellpadding='0'><tr>"
        + "".join(cells) + "</tr></table>"
    )


def format_live_assessment(result, have_flow):
    """Qt-rich-text HTML for the Live Assessment panel readout.

    ``result`` is a :class:`~terrainflow_assessment.modules.water_balance.BalanceResult`.
    With flow data: a colour-coded capture-% headline + bar, the held/leaves split,
    and a per-feature table (inflow → stored, fill %, ⚠ when overflowing). Without:
    per-feature capacities and a run-baseline hint. Pure string building — testable.
    """
    r = result
    parts = []

    if have_flow:
        colour = _capture_colour(r.capture_pct)
        parts.append(
            f"<span style='font-size:20px;font-weight:bold;color:{colour};'>"
            f"{r.capture_pct:.0f}%</span> "
            f"<span style='font-size:11px;color:{_LIVE_MUTED};'>of storm runoff "
            f"captured</span>"
        )
        parts.append(_mini_bar(r.capture_pct, colour))
        stored = max(0.0, r.total_captured_m3 - r.total_infiltration_m3)
        parts.append(
            f"<span style='font-size:11px;color:#2c3e50;'>"
            f"{r.total_captured_m3:,.0f} m³ held ({stored:,.0f} stored + "
            f"{r.total_infiltration_m3:,.0f} soaked in) · "
            f"{r.site_exit_m3:,.0f} m³ leaves site</span>"
        )
    else:
        parts.append(
            f"<span style='font-size:11px;color:{_LIVE_MUTED};'><i>Run baseline "
            "analysis to see storm capture %.</i></span>"
        )

    if r.per_feature:
        rows = []
        for f in r.per_feature:
            if have_flow:
                detail = f"{f['inflow_m3']:,.0f} → {f['stored_m3']:,.0f} m³"
                if f["overflowed"] or f["fill_pct"] >= 100:
                    status = (
                        f"<span style='color:{_LIVE_MID};font-weight:bold;'>⚠ full</span>"
                    )
                else:
                    status = (
                        f"<span style='color:{_LIVE_FILL};'>{f['fill_pct']:.0f}%</span>"
                    )
            else:
                detail = f"{f.get('capacity_m3', 0.0):,.0f} m³"
                status = ""
            rows.append(
                "<tr>"
                f"<td>{f['name']}</td>"
                f"<td align='right'>{detail}</td>"
                f"<td align='right' width='40'>{status}</td>"
                "</tr>"
            )
        parts.append(
            "<table width='100%' cellspacing='0' cellpadding='1' "
            "style='font-size:11px;color:#2c3e50;'>" + "".join(rows) + "</table>"
        )

    parts.append(
        f"<span style='font-size:11px;color:{_LIVE_MUTED};'>"
        f"Capacity {r.total_capacity_m3:,.0f} m³ · Cut {r.total_cut_m3:,.0f} · "
        f"Fill {r.total_fill_m3:,.0f} m³</span>"
    )
    parts.append(
        "<span style='font-size:10px;color:#95a5a6;'>Analytical estimate — verify "
        "with Re-analyse with Earthworks.</span>"
    )
    return "<br>".join(parts)


def attribute_ponding_volume(ponding_diff, cell_area_m2, footprints, min_depth=0.001):
    """Attribute a ponding-difference raster to earthwork footprints, one region each.

    Connected ponded regions (``>= min_depth``) are labelled; each region's volume is
    attributed **once** to the footprint it overlaps most (so a pool upstream of a dam
    is captured via the dam's footprint, and adjacent features don't double-count).
    Regions overlapping no footprint accrue to ``unattributed_m3``.

    Parameters
    ----------
    ponding_diff : 2-D array — terrain ponding depth (typically earthworks − baseline)
    cell_area_m2 : float
    footprints   : list of (name, bool_mask) — one boolean footprint per feature

    Returns
    -------
    (per_name, unattributed_m3) — dict{name: m³}, float
    """
    import numpy as np
    from scipy.ndimage import label

    arr = np.asarray(ponding_diff, dtype="float64")
    ponded = arr >= min_depth
    per_name = {name: 0.0 for name, _ in footprints}
    unattributed = 0.0

    labels, n_regions = label(ponded)
    for region_id in range(1, n_regions + 1):
        region = labels == region_id
        volume = float(arr[region].sum() * cell_area_m2)

        best_name = None
        best_overlap = 0
        for name, mask in footprints:
            overlap = int(np.logical_and(region, mask).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = name

        if best_name is None:
            unattributed += volume
        else:
            per_name[best_name] += volume

    return per_name, unattributed


def build_verification(analytic_by_name, terrain_by_name, baseline_total_m3,
                       earthworks_total_m3, min_dims, cell_size):
    """Assemble the terrain-vs-analytic verification (site headline + per-feature).

    Site terrain-derived storage = ``earthworks_total − baseline_total`` (floored ≥0) —
    isolates the earthwork effect (step-5 before/after integrity). Per feature, a
    ``min_dimension`` below ``cell_size`` is flagged ``routing_only`` and gets no
    independent volume claim (spec §4): sub-cell features validate placement/routing
    only, not storage.
    """
    analytic_total = float(sum(analytic_by_name.values()))
    terrain_total = max(0.0, earthworks_total_m3 - baseline_total_m3)
    delta_m3 = terrain_total - analytic_total
    delta_pct = (delta_m3 / analytic_total * 100.0) if analytic_total > 0 else 0.0

    per_feature = []
    for name, analytic_m3 in analytic_by_name.items():
        min_dim = min_dims.get(name)
        routing_only = min_dim is not None and cell_size > 0 and min_dim < cell_size
        if routing_only:
            terrain_m3 = None
            feat_delta_pct = None
        else:
            terrain_m3 = float(terrain_by_name.get(name, 0.0))
            feat_delta_pct = (
                (terrain_m3 - analytic_m3) / analytic_m3 * 100.0 if analytic_m3 > 0 else None
            )
        per_feature.append({
            "name": name,
            "analytic_m3": float(analytic_m3),
            "terrain_m3": terrain_m3,
            "delta_pct": feat_delta_pct,
            "routing_only": routing_only,
        })

    caveats = [
        "Terrain volume is attributed by connected depression ∩ footprint — the site "
        "total is robust; per-feature figures are indicative for adjacent features.",
        "Terrain total includes barrier-impounded storage (e.g. dams) that has no "
        "analytic counterpart.",
        "Sub-cell features (narrower than one DEM cell) are validated for placement and "
        "routing only, not independent storage volume.",
    ]

    return VerificationResult(
        analytic_total_m3=analytic_total,
        terrain_total_m3=terrain_total,
        delta_m3=delta_m3,
        delta_pct=delta_pct,
        unattributed_m3=0.0,
        per_feature=per_feature,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare(baseline: BaselineReport,
            post: PostInterventionReport) -> ComparisonResult:
    """
    Compute before/after comparison metrics.

    Parameters
    ----------
    baseline : BaselineReport
    post : PostInterventionReport

    Returns
    -------
    ComparisonResult
    """
    result = ComparisonResult(baseline=baseline, post=post)

    total = baseline.total_runoff_m3
    if total > 0:
        captured = total - post.exit_volume_m3
        result.captured_pct = max(0.0, min(100.0, captured / total * 100.0))

    if baseline.exit_volume_m3 > 0:
        result.exit_reduction_pct = max(
            0.0,
            (baseline.exit_volume_m3 - post.exit_volume_m3) / baseline.exit_volume_m3 * 100.0
        )

    if baseline.peak_outflow_ls > 0:
        result.peak_reduction_pct = max(
            0.0,
            (baseline.peak_outflow_ls - post.peak_outflow_ls) / baseline.peak_outflow_ls * 100.0
        )

    result.peak_delay_hr = max(0.0, post.peak_outflow_time_hr - baseline.peak_outflow_time_hr)

    result.net_cut_m3 = sum(
        s.get("cut_vol_m3", 0.0) for s in post.earthwork_summary
    )
    result.net_fill_m3 = sum(
        s.get("fill_vol_m3", 0.0) for s in post.earthwork_summary
    )
    result.net_cut_fill_m3 = result.net_cut_m3 - result.net_fill_m3

    return result


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig):
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _build_hydrograph_chart(baseline: BaselineReport,
                             post: PostInterventionReport):
    """
    Build a before/after outflow hydrograph chart.
    Returns base64 PNG string or None if matplotlib unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    if baseline.timestep_table:
        times_b = [r["time_hr"] for r in baseline.timestep_table]
        flows_b = [r.get("outflow_ls", 0.0) for r in baseline.timestep_table]
        ax.plot(times_b, flows_b, color="#c0392b", linewidth=2,
                label="Baseline (no earthworks)", zorder=3)

    if post.timestep_table:
        times_p = [r["time_hr"] for r in post.timestep_table]
        flows_p = [r.get("outflow_ls", 0.0) for r in post.timestep_table]
        ax.plot(times_p, flows_p, color="#2980b9", linewidth=2,
                label="With earthworks", zorder=3)

    ax.set_xlabel("Time (hr)", fontsize=11)
    ax.set_ylabel("Site exit flow (L/s)", fontsize=11)
    ax.set_title("Outflow Hydrograph — Before vs After", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _build_fill_timeline_chart(post: PostInterventionReport):
    """
    Build a stacked fill-% chart showing each earthwork filling over time.
    Returns base64 PNG string or None.
    """
    if not post.timestep_table or not post.earthwork_summary:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    ew_names = [s["name"] for s in post.earthwork_summary]
    times = [r["time_hr"] for r in post.timestep_table]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    colours = cm.Blues(
        [0.4 + 0.5 * i / max(len(ew_names) - 1, 1) for i in range(len(ew_names))]
    )

    for i, name in enumerate(ew_names):
        col_key = f"{name}_fill_pct"
        if col_key in post.timestep_table[0]:
            fill_series = [r.get(col_key, 0.0) for r in post.timestep_table]
            ax.plot(times, fill_series, linewidth=1.8, color=colours[i], label=name)
            # Mark overflow event
            s = post.earthwork_summary[i]
            if s.get("overflowed") and s.get("first_overflow_hr") is not None:
                ax.axvline(s["first_overflow_hr"], color=colours[i],
                           linestyle="--", alpha=0.6, linewidth=1.2)

    ax.axhline(100, color="#e74c3c", linewidth=1.5, linestyle=":", alpha=0.8,
               label="100% full (overflow)")
    ax.set_xlabel("Time (hr)", fontsize=11)
    ax.set_ylabel("Fill (%)", fontsize=11)
    ax.set_title("Earthwork Fill Timeline", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def export_html(comparison: ComparisonResult, output_path: str,
                methodology_text: str = "") -> str:
    """
    Generate a self-contained HTML report and save it to output_path.

    All matplotlib charts are embedded as base64 PNGs — no external files needed.

    Parameters
    ----------
    comparison : ComparisonResult
    output_path : str — path to write the .html file
    methodology_text : str — optional extra methodology notes

    Returns
    -------
    str — output_path
    """
    baseline = comparison.baseline
    post = comparison.post

    hydrograph_b64 = _build_hydrograph_chart(baseline, post) if (baseline and post) else None
    fill_chart_b64 = _build_fill_timeline_chart(post) if post else None

    def _img_tag(b64, alt=""):
        if b64 is None:
            return "<p><em>(Chart unavailable — matplotlib required)</em></p>"
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;border-radius:6px;">'

    # Build earthwork table rows
    ew_rows = ""
    if post and post.earthwork_summary:
        for s in post.earthwork_summary:
            overflow_cell = (
                f'<span style="color:#e74c3c;">Yes — {s["first_overflow_hr"]} hr</span>'
                if s.get("overflowed") else '<span style="color:#27ae60;">No</span>'
            )
            # Terrain-vs-analytic cells (§4). Sub-cell features validate routing only.
            if s.get("routing_only"):
                terrain_cell = '<span style="color:#7f8c8d;">routing-only</span>'
                delta_cell = "—"
            elif s.get("terrain_ponding_m3") is not None:
                terrain_cell = f"{s['terrain_ponding_m3']:,.1f}"
                dp = s.get("capacity_delta_pct")
                delta_cell = f"{dp:+.0f}%" if dp is not None else "—"
            else:
                terrain_cell = "—"
                delta_cell = "—"
            ew_rows += f"""
            <tr>
              <td>{s['name']}</td>
              <td>{s['type'].capitalize()}</td>
              <td>{s['capacity_m3']:,.1f}</td>
              <td>{terrain_cell}</td>
              <td>{delta_cell}</td>
              <td>{s.get('total_inflow_m3', 0):,.1f}</td>
              <td>{s['peak_fill_pct']:.0f}%</td>
              <td>{overflow_cell}</td>
              <td>{s.get('total_overflow_m3', 0):,.1f}</td>
              <td>{s.get('total_infiltration_m3', 0):,.1f}</td>
              <td>{s.get('cut_vol_m3', 0):,.1f}</td>
              <td>{s.get('fill_vol_m3', 0):,.1f}</td>
            </tr>"""

    # Build the non-circular verification block (§4)
    verification_html = ""
    v = comparison.verification
    if v is not None:
        delta_colour = "#1a7a1a" if abs(v.delta_pct) <= 25 else "#cc6600"
        caveat_items = "".join(f"<li>{c}</li>" for c in v.caveats)
        unattr_row = (
            f'<tr><td>Unattributed terrain ponding</td>'
            f'<td>{v.unattributed_m3:,.1f} m³</td></tr>'
            if v.unattributed_m3 > 0.05 else ""
        )
        verification_html = f"""
  <h2>4b. Non-circular Verification (terrain vs analytic)</h2>
  <div class="card">
    <p>Independent check: storage measured on the <strong>burned</strong> terrain
    (ponding difference vs baseline) against the analytic sizing. Non-circular because
    the terrain figure never sees the analytic capacity.</p>
    <table>
      <tbody>
        <tr><td>Analytic capacity (Σ storage features)</td>
            <td>{v.analytic_total_m3:,.1f} m³</td></tr>
        <tr><td>Terrain-derived storage (earthworks − baseline)</td>
            <td>{v.terrain_total_m3:,.1f} m³</td></tr>
        <tr><td>Delta</td><td><span style="color:{delta_colour};font-weight:bold;">
            {v.delta_m3:+,.1f} m³ ({v.delta_pct:+.0f}%)</span></td></tr>
        {unattr_row}
      </tbody>
    </table>
    <p style="margin-top:8px;font-size:0.85rem;color:#7f8c8d;">The delta is diagnostic,
    not pass/fail — a graded channel legitimately ponds less than its static capacity.</p>
    <div class="caveats"><strong>Attribution caveats</strong><ul>{caveat_items}</ul></div>
  </div>"""

    # Build exit points tables
    def _exit_table(exit_points, title):
        if not exit_points:
            return ""
        rows = "".join(
            f"<tr><td>{ep.get('label', f'Exit {i+1}')}</td>"
            f"<td>{ep.get('volume_m3', 0):,.0f}</td></tr>"
            for i, ep in enumerate(exit_points)
        )
        return f"""
        <h3>{title}</h3>
        <table><thead><tr><th>Exit Point</th><th>Volume (m³)</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    baseline_exits = _exit_table(
        baseline.exit_points if baseline else [], "Baseline Exit Points"
    )
    post_exits = _exit_table(
        post.exit_points if post else [], "Post-Intervention Exit Points"
    )

    # Headline stats
    def _stat(label, value, unit="", highlight=False):
        color = "#2980b9" if highlight else "#2c3e50"
        return f"""
        <div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-value" style="color:{color};">{value}<span class="stat-unit"> {unit}</span></div>
        </div>"""

    stats_html = ""
    if baseline:
        stats_html += _stat("Catchment Area", f"{baseline.catchment_area_ha:,.1f}", "ha")
        stats_html += _stat("Rainfall Event", f"{baseline.rainfall_mm:.0f} mm / {baseline.duration_hr:.0f} hr")
        stats_html += _stat("SCS Curve Number", f"{baseline.cn:.0f}")
        stats_html += _stat("Total Runoff Generated", f"{baseline.total_runoff_m3:,.0f}", "m³")

    summary_rows = ""
    if baseline and post:
        summary_rows = f"""
        <tr><td>Total exit volume</td>
            <td>{baseline.exit_volume_m3:,.0f} m³</td>
            <td>{post.exit_volume_m3:,.0f} m³</td>
            <td class="highlight">−{comparison.exit_reduction_pct:.0f}%</td></tr>
        <tr><td>Peak exit flow</td>
            <td>{baseline.peak_outflow_ls:,.0f} L/s</td>
            <td>{post.peak_outflow_ls:,.0f} L/s</td>
            <td class="highlight">−{comparison.peak_reduction_pct:.0f}%</td></tr>
        <tr><td>Peak flow timing</td>
            <td>{baseline.peak_outflow_time_hr:.1f} hr</td>
            <td>{post.peak_outflow_time_hr:.1f} hr</td>
            <td class="highlight">+{comparison.peak_delay_hr:.1f} hr delay</td></tr>
        <tr><td>Water captured on-site</td>
            <td>—</td>
            <td>{comparison.captured_pct:.0f}% of runoff</td>
            <td class="highlight">{comparison.captured_pct:.0f}%</td></tr>
        """

    net_cut_fill = (
        f"Net cut: {comparison.net_cut_m3:,.0f} m³ | "
        f"Net fill: {comparison.net_fill_m3:,.0f} m³ | "
        f"Balance: {'+' if comparison.net_cut_fill_m3 >= 0 else ''}"
        f"{comparison.net_cut_fill_m3:,.0f} m³ "
        f"({'net cut' if comparison.net_cut_fill_m3 >= 0 else 'net fill'})"
    )

    site_name = baseline.site_name if baseline else "TerrainFlow Assessment"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{site_name} — Hydrological Assessment Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f0f2f5; color: #2c3e50; line-height: 1.6; }}
    .page {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
    h1 {{ font-size: 2rem; color: #1a252f; border-bottom: 3px solid #2980b9;
          padding-bottom: 12px; margin-bottom: 8px; }}
    .subtitle {{ color: #7f8c8d; font-size: 1rem; margin-bottom: 32px; }}
    h2 {{ font-size: 1.4rem; color: #1a252f; margin: 32px 0 12px;
          border-left: 4px solid #2980b9; padding-left: 12px; }}
    h3 {{ font-size: 1.1rem; color: #34495e; margin: 20px 0 8px; }}
    .card {{ background: #fff; border-radius: 10px; padding: 24px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
    .stats-grid {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
    .stat-card {{ background: #fff; border-radius: 8px; padding: 16px 20px;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.08); min-width: 180px; flex: 1; }}
    .stat-label {{ font-size: 0.8rem; color: #7f8c8d; text-transform: uppercase;
                   letter-spacing: 0.5px; margin-bottom: 4px; }}
    .stat-value {{ font-size: 1.5rem; font-weight: 700; color: #2c3e50; }}
    .stat-unit {{ font-size: 0.9rem; font-weight: 400; color: #7f8c8d; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    thead {{ background: #2980b9; color: white; }}
    th {{ padding: 10px 12px; text-align: left; font-weight: 600; }}
    td {{ padding: 9px 12px; border-bottom: 1px solid #ecf0f1; }}
    tr:nth-child(even) td {{ background: #f8f9fa; }}
    .highlight {{ font-weight: 700; color: #27ae60; }}
    .chart-wrap {{ background: #fff; border-radius: 10px; padding: 20px;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
    .caveats {{ background: #fff3cd; border-left: 4px solid #f39c12;
                padding: 16px 20px; border-radius: 6px; margin-top: 12px; }}
    .caveats ul {{ padding-left: 20px; margin-top: 8px; }}
    .caveats li {{ margin-bottom: 6px; font-size: 0.9rem; }}
    footer {{ text-align: center; color: #bdc3c7; font-size: 0.8rem;
              margin-top: 40px; padding-top: 16px; border-top: 1px solid #ecf0f1; }}
  </style>
</head>
<body>
<div class="page">

  <h1>{site_name}</h1>
  <p class="subtitle">Hydrological Assessment Report — generated by TerrainFlow Assessment</p>

  <!-- Section 1: Site summary -->
  <h2>1. Site Summary</h2>
  <div class="stats-grid">
    {stats_html}
  </div>

  <!-- Section 2: Before/after comparison -->
  <h2>2. Before / After Comparison</h2>
  <div class="card">
    <table>
      <thead><tr><th>Metric</th><th>Baseline</th><th>With Earthworks</th><th>Change</th></tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
  </div>

  <!-- Section 3: Hydrograph -->
  <h2>3. Outflow Hydrograph</h2>
  <div class="chart-wrap">
    {_img_tag(hydrograph_b64, "Outflow hydrograph before and after earthworks")}
  </div>

  <!-- Section 4: Earthwork summary -->
  <h2>4. Earthwork Summary</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Name</th><th>Type</th><th>Capacity (m³)</th>
          <th>Terrain Ponding (m³)</th><th>Δ vs analytic</th><th>Total Inflow (m³)</th>
          <th>Peak Fill</th><th>Overflowed?</th><th>Overflow Vol (m³)</th>
          <th>Infiltration (m³)</th><th>Cut (m³)</th><th>Fill (m³)</th>
        </tr>
      </thead>
      <tbody>{ew_rows}</tbody>
    </table>
    <p style="margin-top:12px;font-size:0.85rem;color:#7f8c8d;">{net_cut_fill}</p>
  </div>
  {verification_html}

  <!-- Section 5: Fill timeline -->
  <h2>5. Fill Timeline</h2>
  <div class="chart-wrap">
    {_img_tag(fill_chart_b64, "Earthwork fill percentage over time")}
  </div>

  <!-- Section 6: Exit points -->
  <h2>6. Exit Points</h2>
  <div class="card">
    {baseline_exits}
    {post_exits}
  </div>

  <!-- Section 7: Methodology -->
  <h2>7. Methodology & Caveats</h2>
  <div class="card">
    <h3>Approach</h3>
    <p>Flow routing uses the pysheds D-infinity or D8 algorithm applied to a
    digital elevation model (DEM). Runoff is estimated using the USDA-NRCS SCS
    Curve Number method. Earthwork storage capacity is calculated using trapezoidal
    cross-sections (swales) and polygon depth (basins) with a 0.8 freeboard factor.
    Infiltration losses are estimated using a steady-state rate per soil texture.</p>

    {f'<p style="margin-top:10px;">{methodology_text}</p>' if methodology_text else ""}

    <div class="caveats">
      <strong>Current Version Limitations</strong>
      <ul>
        <li>DEM burning uses a rectangular approximation at the native DEM resolution.
            Accurate swale routing requires DEM resolution ≤ swale width.</li>
        <li>Earthwork capacities assume uniform rectangular cross-sections
            (conservative). Actual trapezoidal capacity is larger.</li>
        <li>Infiltration is modelled as a constant rate (steady-state).
            Initial high infiltration rates (Green-Ampt) are not included.</li>
        <li>Cascading overflow routes to the nearest lower-elevation earthwork
            by elevation centroid — not by actual flow path connectivity.</li>
        <li>Hydrograph timing is approximate: the SCS model distributes runoff
            proportionally to cumulative rainfall and does not model travel time
            through the catchment.</li>
        <li>Future versions will incorporate surveyed cross-sections, higher-resolution
            terrain data, and Green-Ampt infiltration modelling.</li>
      </ul>
    </div>
  </div>

  <footer>TerrainFlow Assessment — Report generated automatically.
  For design decisions, verify results with a qualified engineer.</footer>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    _log.info(f"Report saved to {output_path}")
    return output_path
