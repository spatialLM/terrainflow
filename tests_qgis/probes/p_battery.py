"""p_battery — signed witnesses, dimensional identities, and conservation.

Step D of the defect-documentation campaign. The original battery — finite fraction, NaN,
constant, nodata, dense ranks — catches crashes and blanks, and a **flipped sign has
identical percentile magnitudes**, so it would pass one. These three families are the ones
that earn their place on a real DEM.

**1. Signed witnesses.** The sign of a signed raster's mean over a reference set
identified *independently of that raster*. Plan curvature must be negative over the
top-5 % accumulation network (hollows converge) and positive over the top-5 % TPI (noses
diverge); `aspect` in (90, 270) must coincide with `dz_dy > 0` on real slopes, because
`horn_gradient`'s `dz_dy` rises toward increasing row, which is southward.

The curvature-versus-TPI pass criterion is deliberately not ">60 % agreement", which
correct code fails: plan curvature is a 3x3 second derivative and TPI a 15 m neighbourhood
mean, and they are not measuring the same thing at the same scale. The honest test is the
one that only wrong code fails — `mean(plan[ridge]) > 0 > mean(plan[valley])`, and a
point-biserial correlation that is positive and not trivially so. A **negative**
correlation is the unambiguous failure.

**2. Dimensional identities.** Relations that must hold whatever the terrain, so any
tolerance argument is about float, not about ground:

* `specific_catchment_area` at cell 2 m over cell 1 m is exactly 2.0 — it is
  `(acc+1)·cell_area/cell_w`, linear in cell size.
* `catchment_ha · 10_000 == (acc + 1) · cell_area` — the `+1` is the question. It is
  *documented and measured* in `specific_catchment_area`; whether the keyline tier's own
  `catchment_ha` agrees is a separate matter and is measured here.
* `erosion_spacing_m == terrace_vertical_interval(p50) / (p50 grade)` — the unit chain
  that would expose a feet-to-metres slip in the terrace rule, which is published in feet
  and consumed in metres.

**3. Conservation.** `|accepted| + |refused| == |input|` for every screening tool, each
refusal reproducible by re-running its own guard. This is KPA-39, KPA-43 and IMP-03
expressed as arithmetic rather than as three separate readings, and it generalises to
`clip_to_usable_area`'s `dropped` count.

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_battery.py
"""

import _probe

import numpy as np

TOP_FRACTION = 0.05
SMOOTH_WINDOW_M = 15.0
MIN_SLOPE_DEG_FOR_ASPECT = 2.0


# ------------------------------------------------------------------ witnesses


def point_biserial(values, group):
    """Correlation between a continuous *values* and a boolean *group*.

    The ordinary Pearson r of `values` against `group.astype(float)`, which is what
    "point-biserial" means; written out so the pass criterion in the docstring above is
    checkable against the number without knowing which library computed it.
    """
    v = np.asarray(values, dtype="float64")
    g = np.asarray(group, dtype="float64")
    ok = np.isfinite(v) & np.isfinite(g)
    v, g = v[ok], g[ok]
    if v.size < 2 or v.std() == 0 or g.std() == 0:
        return None
    return float(np.corrcoef(v, g)[0, 1])


def stage_signed_witnesses(ev, z, cell_w, cell_h):
    from terrainflow_assessment.modules.dem_loader import (
        aspect_degrees,
        horn_gradient,
        slope_degrees,
    )
    from terrainflow_assessment.modules.terrain_indices import (
        curvature,
        landform_classes,
        landform_tpi,
    )

    with ev.stage("signed_witness_plan_curvature") as rec:
        from terrainflow_assessment.modules.flow_graph import d8_from_dem

        plan, profile = curvature(z, cell_w, cell_h)
        tpi = landform_tpi(z, cell_w, cell_h, window_m=SMOOTH_WINDOW_M)

        # The reference sets come from somewhere other than curvature, or the witness
        # would be testing a raster against itself.
        next_flat, _sink = d8_from_dem(z, cell_w, cell_h)
        finite = np.isfinite(z).ravel()
        acc = np.ones(z.size, dtype=np.int64)
        acc[~finite] = 0
        flat_z = z.ravel()
        for i in np.argsort(np.where(finite, -flat_z, np.inf), kind="stable"):
            if not finite[i]:
                continue
            j = int(next_flat[i])
            if j != i:
                acc[j] += acc[i]
        acc = acc.reshape(z.shape)

        valid = np.isfinite(plan) & np.isfinite(z)
        acc_cut = np.quantile(acc[valid], 1.0 - TOP_FRACTION)
        tpi_cut = np.quantile(tpi[valid & np.isfinite(tpi)], 1.0 - TOP_FRACTION)
        network = valid & (acc >= acc_cut)
        noses = valid & np.isfinite(tpi) & (tpi >= tpi_cut)

        rec["top_accumulation_cut_cells"] = float(acc_cut)
        rec["top_tpi_cut_m"] = float(tpi_cut)
        rec["mean_plan_over_accumulation_network"] = float(plan[network].mean())
        rec["mean_plan_over_top_tpi"] = float(plan[noses].mean())
        rec["network_cells"] = int(network.sum())
        rec["nose_cells"] = int(noses.sum())
        rec["witness_holds"] = bool(
            rec["mean_plan_over_accumulation_network"] < 0
            < rec["mean_plan_over_top_tpi"])
        ev.note(
            f"Signed witness, plan curvature: mean over the top-{TOP_FRACTION:.0%} "
            f"accumulation network {rec['mean_plan_over_accumulation_network']:+.3e} /m "
            f"({rec['network_cells']:,} cells, must be negative); over the top-"
            f"{TOP_FRACTION:.0%} TPI {rec['mean_plan_over_top_tpi']:+.3e} /m "
            f"({rec['nose_cells']:,} cells, must be positive). Holds: "
            f"{rec['witness_holds']}.")

    with ev.stage("signed_witness_curvature_vs_tpi") as rec:
        classes = landform_classes(tpi)
        ridge = classes == 1
        valley = classes == -1
        rec["ridge_cells"] = int(ridge.sum())
        rec["valley_cells"] = int(valley.sum())
        rec["mean_plan_on_ridge"] = float(np.nanmean(plan[ridge])) if ridge.any() else None
        rec["mean_plan_in_valley"] = (float(np.nanmean(plan[valley]))
                                      if valley.any() else None)
        rec["sign_test_holds"] = bool(
            rec["mean_plan_in_valley"] is not None
            and rec["mean_plan_on_ridge"] is not None
            and rec["mean_plan_in_valley"] < 0 < rec["mean_plan_on_ridge"])

        classed = ridge | valley
        rec["point_biserial_raw"] = point_biserial(plan[classed], ridge[classed])

        # The same correlation after a pre-smooth at the TPI's own scale, which is the
        # apples-to-apples comparison: a 3x3 second derivative and a 15 m mean disagree
        # about high-frequency detail by construction, not by error.
        from scipy.ndimage import uniform_filter

        size = max(3, int(round(SMOOTH_WINDOW_M / ((cell_w + cell_h) / 2.0))))
        if size % 2 == 0:
            size += 1
        finite_z = np.isfinite(z)
        filled = np.where(finite_z, z, 0.0)
        total = uniform_filter(filled, size=size)
        count = uniform_filter(finite_z.astype("float64"), size=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            smooth_z = np.where(count > 0, total / count, np.nan)
        smooth_plan, _sp = curvature(smooth_z, cell_w, cell_h)
        rec["point_biserial_presmoothed"] = point_biserial(
            smooth_plan[classed], ridge[classed])
        rec["presmooth_window_cells"] = size

        # Agreement as a percentage, recorded because the plan quotes 64.5% and a reader
        # will look for it — but NOT used as the pass criterion, for the reason above.
        agree = ((plan > 0) & ridge) | ((plan < 0) & valley)
        rec["agreement_fraction"] = float(agree[classed].sum() / max(classed.sum(), 1))
        rec["criterion"] = (
            "sign test must hold; point-biserial >= +0.25 raw and > +0.5 pre-smoothed; "
            "a negative correlation is the unambiguous failure")
        rec["passes"] = bool(
            rec["sign_test_holds"]
            and (rec["point_biserial_raw"] or -1) >= 0.25)
        ev.note(
            f"Curvature vs TPI: mean plan on ridge "
            f"{rec['mean_plan_on_ridge']:+.3e}, in valley "
            f"{rec['mean_plan_in_valley']:+.3e} (sign test {rec['sign_test_holds']}); "
            f"point-biserial {rec['point_biserial_raw']:+.3f} raw, "
            f"{rec['point_biserial_presmoothed']:+.3f} after a {size}-cell pre-smooth; "
            f"cell-wise agreement {rec['agreement_fraction']:.1%}.")

    with ev.stage("signed_witness_aspect") as rec:
        dz_dx, dz_dy, _invalid = horn_gradient(z, cell_w, cell_h)
        aspect = aspect_degrees(z, cell_w, cell_h)
        slope = slope_degrees(z, cell_w, cell_h)

        real = (np.isfinite(aspect) & (aspect >= 0)
                & np.isfinite(slope) & (slope > MIN_SLOPE_DEG_FOR_ASPECT))
        southward = (aspect > 90.0) & (aspect < 270.0)

        # `horn_gradient`'s dz_dy rises toward increasing row, and row increases
        # southward — so ground that *falls* to the south has dz_dy < 0, and aspect, which
        # points downslope, lands in (90, 270) exactly there. Algebraically:
        # `aspect = atan2(-dz_dx, dz_dy)`, whose northward component is dz_dy, so
        # `cos(aspect) < 0` iff `dz_dy < 0`.
        #
        # The campaign plan and §6 of the register both state this witness as
        # "coincides with dz_dy > 0", carrying the docstring's "increasing row is
        # southward" through without its negation. Both forms are measured below,
        # because a witness stated backwards fails loudly on correct code and the next
        # reader deserves to see which way round it came out.
        falls_south = dz_dy < 0

        agree = southward == falls_south
        as_planned = southward == (dz_dy > 0)
        n = max(int(real.sum()), 1)
        rec["cells_tested"] = int(real.sum())
        rec["min_slope_deg"] = MIN_SLOPE_DEG_FOR_ASPECT
        rec["agreement_fraction"] = float(agree[real].sum() / n)
        rec["disagreeing_cells"] = int((~agree & real).sum())
        rec["agreement_as_the_plan_stated_it"] = float(as_planned[real].sum() / n)
        rec["criterion"] = ">= 0.90 against dz_dy < 0"
        rec["passes"] = bool(rec["agreement_fraction"] >= 0.90)
        ev.note(
            f"Signed witness, aspect: on {rec['cells_tested']:,} cells steeper than "
            f"{MIN_SLOPE_DEG_FOR_ASPECT:.0f} deg, aspect in (90, 270) coincides with "
            f"dz_dy < 0 for {rec['agreement_fraction']:.2%} "
            f"({rec['disagreeing_cells']:,} disagree). Passes: {rec['passes']}. "
            f"Stated as the plan has it — dz_dy > 0 — the same cells agree only "
            f"{rec['agreement_as_the_plan_stated_it']:.2%} of the time, which is the "
            f"signature of a witness written with the sign inverted, not of a broken "
            f"aspect: `aspect = atan2(-dz_dx, dz_dy)` has dz_dy as its northward term.")


# ------------------------------------------------------------------ identities


def stage_dimensional_identities(ev, dem_path, z, cell_w, cell_h):
    from terrainflow_assessment.modules.terrain_indices import specific_catchment_area

    with ev.stage("identity_specific_catchment_area_scales") as rec:
        acc = np.array([[0, 1, 2], [5, 10, 100]], dtype="float64")
        one = specific_catchment_area(acc, 1.0, 1.0)
        two = specific_catchment_area(acc, 2.0, 2.0)
        ratio = two / one
        rec["ratio_min"] = float(ratio.min())
        rec["ratio_max"] = float(ratio.max())
        rec["exactly_two"] = bool(np.all(ratio == 2.0))
        rec["max_abs_error"] = float(np.abs(ratio - 2.0).max())
        ev.note(
            f"Identity, specific catchment area: a(2 m)/a(1 m) = "
            f"[{rec['ratio_min']:.15g}, {rec['ratio_max']:.15g}]; exactly 2.0: "
            f"{rec['exactly_two']}.")

    with ev.stage("identity_catchment_ha_plus_one") as rec:
        # `specific_catchment_area` documents and *measured* a `+1`: pysheds' acc excludes
        # the cell itself, so a ridge cell reads 0 and ln(a) is -inf without it. The
        # keyline tier computes its own `catchment_ha` from the same raster. Whether the
        # two agree about the cell's own footprint is the question, and it is worth a
        # number rather than a reading of two expressions.
        import inspect

        from terrainflow_assessment.modules.keypoint_analysis import (
            YeomansKeylineAnalysis,
        )

        src = inspect.getsource(YeomansKeylineAnalysis.find_keypoints)
        rec["keyline_expression"] = next(
            (line.strip() for line in src.splitlines() if "catchment_ha" in line
             and "=" in line and "kp[" in line), None)
        rec["sca_expression"] = "(acc + 1.0) * cell_area / cell_w"

        from terrainflow_assessment.modules.flow_graph import (
            d8_from_dem,
            strahler_order,
            stream_links,
        )

        ya = YeomansKeylineAnalysis(str(dem_path))
        keypoints, _skipped = ya.find_keypoints(max_valleys=8)
        cell_area = ya.cell_w * ya.cell_h

        # `_catchment(link)` reads the accumulation at the link's **last** cell, not at
        # the keypoint. Rebuild the link list the same way `find_keypoints` does so the
        # identity compares like with like — and so the gap between the two cells can be
        # reported rather than mistaken for the `+1`.
        _fdir, acc_arr = ya._ensure_flow_data()
        threshold = max(20, int(round(2_000.0 / cell_area)))
        stream = (acc_arr >= threshold) & np.isfinite(ya.dem)
        next_flat, _sink = d8_from_dem(ya.dem, ya.cell_w, ya.cell_h)
        order = strahler_order(next_flat, stream.ravel())
        links = stream_links(next_flat, stream.ravel(), order, ya.dem.shape[1],
                             max_order=1)
        links.sort(key=lambda lk: float(acc_arr[lk[-1][0], lk[-1][1]]), reverse=True)

        # Pair each keypoint with its own link by membership, not by position: 126 of the
        # 127 links refuse, so `keypoints[0]` is not `links[0]`. `valley_cells` is carried
        # on the keypoint and is checked against the link found, so a wrong pairing shows
        # up here instead of quietly skewing the identity.
        by_cell = {}
        for link in links:
            for cell in link:
                by_cell.setdefault((int(cell[0]), int(cell[1])), link)

        rows = []
        for kp in keypoints:
            link = by_cell.get((int(kp["_row"]), int(kp["_col"])))
            if link is None:
                rec.setdefault("keypoints_not_on_any_link", []).append(
                    [int(kp["_row"]), int(kp["_col"])])
                continue
            outlet_acc = float(acc_arr[link[-1][0], link[-1][1]])
            keypoint_acc = float(acc_arr[kp["_row"], kp["_col"]])
            rows.append({
                "keypoint_rc": [int(kp["_row"]), int(kp["_col"])],
                "link_outlet_rc": [int(link[-1][0]), int(link[-1][1])],
                "acc_at_link_outlet_cells": outlet_acc,
                "acc_at_keypoint_cells": keypoint_acc,
                "catchment_ha_reported": float(kp["catchment_ha"]),
                "valley_cells_reported": int(kp["valley_cells"]),
                "link_length_cells": len(link),
                "pairing_confirmed": int(kp["valley_cells"]) == len(link),
                "ha_from_outlet_without_plus_one": outlet_acc * cell_area / 10_000.0,
                "ha_from_outlet_with_plus_one": (outlet_acc + 1.0) * cell_area / 10_000.0,
                "ha_above_the_keypoint_itself": keypoint_acc * cell_area / 10_000.0,
            })
        rec["keypoints"] = rows
        if rows:
            r = rows[0]
            rec["matches_without_plus_one"] = bool(
                abs(r["catchment_ha_reported"]
                    - r["ha_from_outlet_without_plus_one"]) < 1e-9)
            rec["matches_with_plus_one"] = bool(
                abs(r["catchment_ha_reported"]
                    - r["ha_from_outlet_with_plus_one"]) < 1e-9)
            rec["plus_one_discrepancy_m2"] = cell_area
            rec["label_overstates_by_ha"] = (
                r["catchment_ha_reported"] - r["ha_above_the_keypoint_itself"])
            ev.note(
                f"Identity, catchment_ha: the rank-1 keypoint's label quotes "
                f"{r['catchment_ha_reported']:.6f} ha. That is the accumulation at the "
                f"link's outlet cell without a +1 "
                f"(match: {rec['matches_without_plus_one']}); with the +1 that "
                f"`specific_catchment_area` documents and measured it would be "
                f"{r['ha_from_outlet_with_plus_one']:.6f} ha "
                f"(match: {rec['matches_with_plus_one']}) — the two conventions differ "
                f"by one cell, {cell_area:.1f} m2, and the tree carries both.")
            ev.note(
                f"Separately: the ground actually above the keypoint is "
                f"{r['ha_above_the_keypoint_itself']:.6f} ha, so the label "
                f"\"{r['catchment_ha_reported']:.1f} ha above\" attached to the keypoint "
                f"overstates it by {rec['label_overstates_by_ha']:.3f} ha "
                f"({rec['label_overstates_by_ha'] / max(r['ha_above_the_keypoint_itself'], 1e-9):.1%}). "
                f"It is the whole valley link's catchment, measured at the link's "
                f"bottom, presented against a point partway up it.")
        else:
            ev.note("Identity, catchment_ha: no keypoints found, nothing to compare.")

    with ev.stage("identity_terrace_unit_chain") as rec:
        from terrainflow_assessment.core.sizing.advisories import (
            spacing_advisory,
            terrace_vertical_interval,
        )
        from terrainflow_assessment.modules.dem_loader import slope_degrees
        from terrainflow_assessment.modules.terrain_indices import slope_statistics

        stats = slope_statistics(slope_degrees(z, cell_w, cell_h))
        rec["slope_statistics"] = stats
        p50_deg = float(stats["p50"])
        p50_pct = float(np.tan(np.radians(p50_deg)) * 100.0)
        rec["p50_deg"] = p50_deg
        rec["p50_pct"] = p50_pct

        vi_m = terrace_vertical_interval(p50_pct, "Loam")
        advice = spacing_advisory(p50_pct, "Loam")
        expected = vi_m / (p50_pct / 100.0)
        rec["vertical_interval_m"] = vi_m
        rec["erosion_spacing_m_reported"] = advice["erosion_spacing_m"]
        rec["erosion_spacing_m_expected"] = expected
        rec["relative_error"] = abs(
            advice["erosion_spacing_m"] - expected) / max(expected, 1e-12)
        rec["identity_holds"] = bool(rec["relative_error"] < 1e-12)

        # The feet-to-metres half of the chain, stated as a number rather than trusted:
        # VI(ft) = X*S + Y, so at S = 0 the interval must be exactly Y feet in metres.
        rec["vi_at_zero_slope_m"] = terrace_vertical_interval(0.0, "Loam")
        rec["expected_y_feet_in_m"] = 2.0 * 0.3048
        rec["foot_conversion_exact"] = bool(
            abs(rec["vi_at_zero_slope_m"] - rec["expected_y_feet_in_m"]) < 1e-12)
        ev.note(
            f"Identity, terrace unit chain: site p50 slope {p50_deg:.3f} deg "
            f"({p50_pct:.3f}%); VI {vi_m:.4f} m; erosion spacing reported "
            f"{advice['erosion_spacing_m']:.6f} m against VI/grade "
            f"{expected:.6f} m (relative error {rec['relative_error']:.2e}, holds "
            f"{rec['identity_holds']}). VI at zero slope {rec['vi_at_zero_slope_m']:.6f} "
            f"m against Y=2.0 ft = {rec['expected_y_feet_in_m']:.6f} m: "
            f"{rec['foot_conversion_exact']}.")


# ---------------------------------------------------------------- conservation


def stage_conservation(ev, dem_path):
    with ev.stage("conservation_find_keypoints") as rec:
        from terrainflow_assessment.modules.flow_graph import (
            d8_from_dem,
            strahler_order,
            stream_links,
        )
        from terrainflow_assessment.modules.keypoint_analysis import (
            YeomansKeylineAnalysis,
        )

        ya = YeomansKeylineAnalysis(str(dem_path))
        _fdir, acc_arr = ya._ensure_flow_data()
        threshold = max(20, int(round(2_000.0 / (ya.cell_w * ya.cell_h))))
        stream = (acc_arr >= threshold) & np.isfinite(ya.dem)
        next_flat, _sink = d8_from_dem(ya.dem, ya.cell_w, ya.cell_h)
        order = strahler_order(next_flat, stream.ravel())
        links = stream_links(next_flat, stream.ravel(), order, ya.dem.shape[1],
                             max_order=1)

        rec["input_links"] = len(links)
        rec["by_max_valleys"] = {}
        for max_valleys in (1, 8, len(links)):
            keypoints, skipped = ya.find_keypoints(max_valleys=max_valleys)
            accounted = len(keypoints) + len(skipped)
            rec["by_max_valleys"][str(max_valleys)] = {
                "accepted": len(keypoints),
                "refused": len(skipped),
                "accounted_for": accounted,
                "unexamined": len(links) - accounted,
                "conserves": accounted == len(links),
            }
        ev.note(
            "Conservation, find_keypoints: " + "; ".join(
                f"max_valleys={k}: {v['accepted']} accepted + {v['refused']} refused = "
                f"{v['accounted_for']} of {rec['input_links']} links "
                f"({v['unexamined']} never examined, conserves {v['conserves']})"
                for k, v in rec["by_max_valleys"].items()))

    with ev.stage("conservation_clip_to_usable_area") as rec:
        # `clip_to_usable_area` is the tool that already returns its refusal count, so it
        # is the one that should conserve exactly. A half-overlapping polygon is the case
        # that exercises trim as well as drop.
        from shapely.geometry import box

        from terrainflow_assessment.modules.contour_analysis import clip_to_usable_area

        import rasterio

        with rasterio.open(str(dem_path)) as src:
            bounds = src.bounds

        contours = _fake_contours(bounds)
        rec["input_contours"] = len(contours)
        half = box(bounds.left, bounds.bottom,
                   (bounds.left + bounds.right) / 2.0, bounds.top)
        kept, dropped = clip_to_usable_area(contours, half)
        rec["kept"] = len(kept)
        rec["dropped"] = dropped
        rec["accounted_for"] = len(kept) + dropped
        rec["conserves"] = rec["accounted_for"] == len(contours)
        ev.note(
            f"Conservation, clip_to_usable_area: {len(contours)} in -> {len(kept)} kept "
            f"+ {dropped} dropped = {rec['accounted_for']} "
            f"(conserves: {rec['conserves']}).")


def _fake_contours(bounds):
    """Straight east-west lines across the DEM's extent, as `ContourFeature`s.

    Synthetic rather than measured on purpose: conservation is arithmetic about counts,
    and using real contours would make a failure ambiguous between the tool and the
    terrain. Half of these lie wholly inside the western half-box, half cross it.
    """
    from shapely.geometry import LineString

    from terrainflow_assessment.modules.contour_analysis import ContourFeature

    mid_x = (bounds.left + bounds.right) / 2.0
    out = []
    for i in range(10):
        y = bounds.bottom + (i + 1) * (bounds.top - bounds.bottom) / 12.0
        if i % 2 == 0:
            geom = LineString([(bounds.left + 1.0, y), (mid_x - 1.0, y)])   # inside
        else:
            geom = LineString([(mid_x + 1.0, y), (bounds.right - 1.0, y)])  # outside
        out.append(ContourFeature(geometry=geom, elevation=float(i), rank=i))
    return out


# --------------------------------------------------------------------- driver


def main():
    _probe.start_qgis()
    dem = _probe.fixture_path()
    _probe.banner("p_battery — signed witnesses, identities, conservation", dem)

    import rasterio

    with rasterio.open(str(dem)) as src:
        z = src.read(1).astype("float64")
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
        cell_w, cell_h = abs(src.transform.a), abs(src.transform.e)

    ev = _probe.Evidence("p_battery", ["STEP-D"], dem)
    ev["dem_stats"] = _probe.dem_stats(dem)

    stage_signed_witnesses(ev, z, cell_w, cell_h)
    stage_dimensional_identities(ev, dem, z, cell_w, cell_h)
    stage_conservation(ev, dem)

    ev.write()


if __name__ == "__main__":
    main()
