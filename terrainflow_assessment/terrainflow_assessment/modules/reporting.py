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
from typing import List, Optional, Dict, Any

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
    exit_points: List[Dict] = field(default_factory=list)
    # Timestep series for hydrograph
    timestep_table: List[Dict] = field(default_factory=list)


@dataclass
class PostInterventionReport:
    """Results from the post-earthworks analysis + fill simulation."""
    exit_volume_m3: float = 0.0
    peak_outflow_ls: float = 0.0
    peak_outflow_time_hr: float = 0.0
    total_infiltrated_m3: float = 0.0
    earthwork_summary: List[Dict] = field(default_factory=list)
    timestep_table: List[Dict] = field(default_factory=list)
    exit_points: List[Dict] = field(default_factory=list)


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
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
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
            ew_rows += f"""
            <tr>
              <td>{s['name']}</td>
              <td>{s['type'].capitalize()}</td>
              <td>{s['capacity_m3']:,.1f}</td>
              <td>{s.get('total_inflow_m3', 0):,.1f}</td>
              <td>{s['peak_fill_pct']:.0f}%</td>
              <td>{overflow_cell}</td>
              <td>{s.get('total_overflow_m3', 0):,.1f}</td>
              <td>{s.get('total_infiltration_m3', 0):,.1f}</td>
              <td>{s.get('cut_vol_m3', 0):,.1f}</td>
              <td>{s.get('fill_vol_m3', 0):,.1f}</td>
            </tr>"""

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
          <th>Name</th><th>Type</th><th>Capacity (m³)</th><th>Total Inflow (m³)</th>
          <th>Peak Fill</th><th>Overflowed?</th><th>Overflow Vol (m³)</th>
          <th>Infiltration (m³)</th><th>Cut (m³)</th><th>Fill (m³)</th>
        </tr>
      </thead>
      <tbody>{ew_rows}</tbody>
    </table>
    <p style="margin-top:12px;font-size:0.85rem;color:#7f8c8d;">{net_cut_fill}</p>
  </div>

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
