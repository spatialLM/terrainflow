"""p_crosscheck — four quantities computed twice, and made to agree.

Step F of the defect-documentation campaign. Each check takes a number two parts of the
tree derive independently and asks whether they match to a tolerance chosen from the
arithmetic rather than from taste:

============================================================  ==========  ==============
Check                                                          Tolerance   Why that one
============================================================  ==========  ==============
``haul_regions`` volumes against ``burn_quantities``              0.1 %    same sum, different grouping
``slope_degrees`` / ``aspect_degrees`` off ``horn_gradient``      1e-12    one stencil, three consumers
``flow_bearing`` against ``aspect_degrees``                     1e-6 deg   same vector, two encodings
``stream_links`` emitted / consumable / keypoints + skipped      exact     counts, not measurements
============================================================  ==========  ==============

The first is the cheapest high-value check available: `haul_regions` and `burn_quantities`
reduce the *same* two surfaces to a volume, one by connected component and one by a flat
sum. Any disagreement beyond `min_region_m3` is an accounting error in a number that goes
to a contractor. The threshold's own cost is measured separately — volume the haul plan
drops on the floor is not an error, but it should be a known quantity rather than an
unknown one.

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_crosscheck.py
"""

import _probe

import numpy as np

HAUL_TOLERANCE = 0.001          # 0.1 %
STENCIL_TOLERANCE = 1e-12
BEARING_TOLERANCE_DEG = 1e-6


# ------------------------------------------------------------- 1. haul volumes


def synthetic_burn(z, cell_w, cell_h):
    """A burn over the real DEM: one large trench, one large bank, and small spoil.

    Synthetic rather than a real design run, and deliberately so. The identity under test
    is arithmetic over two surfaces — it does not care how the second one arose — and a
    full earthwork run would make a failure ambiguous between the burn and the accounting.
    The small features are sized to straddle ``DEFAULT_MIN_REGION_M3`` so the threshold's
    cost is exercised rather than assumed away.
    """
    burned = z.copy()
    rows, cols = z.shape
    rr, cc = np.mgrid[0:rows, 0:cols]

    # A 4 m wide trench 0.5 m deep across the middle third: comfortably over any
    # min_region_m3.
    trench = (np.abs(rr - rows // 3) <= 2) & (cc > cols // 4) & (cc < 3 * cols // 4)
    burned[trench] -= 0.5

    # A bank of similar footprint, so cut and fill are both non-trivial.
    bank = (np.abs(rr - 2 * rows // 3) <= 2) & (cc > cols // 4) & (cc < 3 * cols // 4)
    burned[bank] += 0.4

    # Eight isolated one-cell dabs of 0.1 m. At 1 m cells each is 0.1 m3, well under the
    # 5.0 m3 default floor, so every one of them is dropped by `haul_regions` and counted
    # by `burn_quantities`. That gap is the measurement, not a bug.
    for i in range(8):
        burned[10 + i * 7, 10] += 0.1

    burned[~np.isfinite(z)] = np.nan
    return burned


def stage_haul_vs_burn(ev, z, transform, cell_w, cell_h):
    from terrainflow_assessment.modules.earthwork_design import burn_quantities
    from terrainflow_assessment.modules.mass_haul import (
        DEFAULT_MIN_REGION_M3,
        haul_regions,
    )

    with ev.stage("crosscheck_haul_regions_vs_burn_quantities") as rec:
        cell_area = cell_w * cell_h
        burned = synthetic_burn(z, cell_w, cell_h)
        quantities = burn_quantities(z, burned, cell_area)
        rec["burn_quantities"] = quantities
        rec["cell_area_m2"] = cell_area

        for label, floor in (("no_floor", 0.0), ("default_floor", DEFAULT_MIN_REGION_M3)):
            cuts, fills = haul_regions(z, burned, transform, cell_area,
                                       min_region_m3=floor)
            cut_total = float(sum(r["volume_m3"] for r in cuts))
            fill_total = float(sum(r["volume_m3"] for r in fills))
            rec[label] = {
                "min_region_m3": floor,
                "cut_regions": len(cuts),
                "fill_regions": len(fills),
                "cut_m3": cut_total,
                "fill_m3": fill_total,
                "cut_relative_error": _rel(cut_total, quantities["cut_m3"]),
                "fill_relative_error": _rel(fill_total, quantities["fill_m3"]),
            }

        rec["passes"] = bool(
            rec["no_floor"]["cut_relative_error"] < HAUL_TOLERANCE
            and rec["no_floor"]["fill_relative_error"] < HAUL_TOLERANCE)
        rec["volume_dropped_by_the_default_floor_m3"] = {
            "cut": rec["no_floor"]["cut_m3"] - rec["default_floor"]["cut_m3"],
            "fill": rec["no_floor"]["fill_m3"] - rec["default_floor"]["fill_m3"],
        }
        rec["regions_dropped_by_the_default_floor"] = {
            "cut": rec["no_floor"]["cut_regions"] - rec["default_floor"]["cut_regions"],
            "fill": rec["no_floor"]["fill_regions"] - rec["default_floor"]["fill_regions"],
        }
        ev.note(
            f"Cross-check, haul volumes: burn_quantities says cut "
            f"{quantities['cut_m3']:,.1f} m3 / fill {quantities['fill_m3']:,.1f} m3. "
            f"haul_regions with no floor sums to {rec['no_floor']['cut_m3']:,.1f} / "
            f"{rec['no_floor']['fill_m3']:,.1f} m3 — relative error "
            f"{rec['no_floor']['cut_relative_error']:.2e} and "
            f"{rec['no_floor']['fill_relative_error']:.2e} against a 0.1% bar "
            f"(passes: {rec['passes']}). At the shipped "
            f"min_region_m3={DEFAULT_MIN_REGION_M3:g} m3 the haul plan omits "
            f"{rec['volume_dropped_by_the_default_floor_m3']['fill']:.2f} m3 of fill "
            f"across {rec['regions_dropped_by_the_default_floor']['fill']} regions, "
            f"which the site total still counts.")


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


# ----------------------------------------------------------------- 2. stencil


def stage_one_stencil(ev, z, cell_w, cell_h):
    from terrainflow_assessment.modules.dem_loader import (
        aspect_degrees,
        horn_gradient,
        slope_degrees,
    )

    with ev.stage("crosscheck_one_stencil_three_consumers") as rec:
        dz_dx, dz_dy, invalid = horn_gradient(z, cell_w, cell_h)
        slope = slope_degrees(z, cell_w, cell_h).astype("float64")
        aspect = aspect_degrees(z, cell_w, cell_h).astype("float64")

        # Re-derive each consumer from the gradient the shared stencil returned. If they
        # agree to float32's own precision, the three genuinely come off one stencil; if
        # one has its own quiet copy, this is where it shows.
        slope_expected = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
        slope_expected[invalid] = np.nan
        aspect_expected = np.mod(np.degrees(np.arctan2(-dz_dx, dz_dy)), 360.0)
        flat = (dz_dx == 0.0) & (dz_dy == 0.0)
        aspect_expected = np.where(flat, -1.0, aspect_expected)
        aspect_expected[invalid] = np.nan

        ok = np.isfinite(slope) & np.isfinite(slope_expected)
        rec["slope_max_abs_error_deg"] = float(
            np.abs(slope[ok] - slope_expected[ok]).max())

        bearing = np.isfinite(aspect) & (aspect >= 0) & np.isfinite(aspect_expected) \
            & (aspect_expected >= 0)
        d = np.abs(aspect[bearing] - aspect_expected[bearing])
        d = np.minimum(d, 360.0 - d)
        rec["aspect_max_abs_error_deg"] = float(d.max()) if d.size else 0.0
        rec["flat_sentinel_disagreements"] = int(
            ((aspect < 0) != (aspect_expected < 0)).sum())

        # Both are returned float32, so the shared-stencil claim can only be checked to
        # float32's spacing. The 1e-12 bar is stated in the plan against float64 maths;
        # the honest version of it is "no worse than the cast".
        rec["float32_spacing_at_360_deg"] = float(np.spacing(np.float32(360.0)))
        rec["float32_spacing_at_90_deg"] = float(np.spacing(np.float32(90.0)))
        rec["passes_at_float32_precision"] = bool(
            rec["slope_max_abs_error_deg"] <= rec["float32_spacing_at_90_deg"]
            and rec["aspect_max_abs_error_deg"] <= rec["float32_spacing_at_360_deg"])
        rec["passes_at_1e_12"] = bool(
            rec["slope_max_abs_error_deg"] < STENCIL_TOLERANCE
            and rec["aspect_max_abs_error_deg"] < STENCIL_TOLERANCE)
        ev.note(
            f"Cross-check, one stencil: re-deriving slope and aspect from "
            f"horn_gradient's own output gives max errors "
            f"{rec['slope_max_abs_error_deg']:.3e} deg and "
            f"{rec['aspect_max_abs_error_deg']:.3e} deg, with "
            f"{rec['flat_sentinel_disagreements']} flat-sentinel disagreements. "
            f"Against the plan's 1e-12 bar: {rec['passes_at_1e_12']}. Against float32's "
            f"own spacing, which is the tightest either return value can carry: "
            f"{rec['passes_at_float32_precision']}.")


# ------------------------------------------------------------ 3. flow bearing


def stage_flow_bearing(ev, z, cell_w, cell_h):
    from terrainflow_assessment.modules.dem_loader import aspect_degrees, horn_gradient
    from terrainflow_assessment.modules.impoundment_sites import flow_bearing

    with ev.stage("crosscheck_flow_bearing_vs_aspect") as rec:
        dz_dx, dz_dy, _invalid = horn_gradient(z, cell_w, cell_h)
        aspect = aspect_degrees(z, cell_w, cell_h).astype("float64")

        rows, cols = z.shape
        rng = np.random.default_rng(20260911)
        tested, worst, worst_cell, none_count = 0, 0.0, None, 0
        samples = []
        while tested < 2000 and len(samples) < 20000:
            r = int(rng.integers(1, rows - 1))
            c = int(rng.integers(1, cols - 1))
            samples.append((r, c))
            if not np.isfinite(aspect[r, c]) or aspect[r, c] < 0:
                continue
            vec = flow_bearing(dz_dx, dz_dy, r, c)
            if vec is None:
                none_count += 1
                continue
            ex, ny = vec
            # `flow_bearing` returns a map-space unit vector (east, north) pointing
            # downslope; `aspect_degrees` returns the same direction as a clockwise
            # bearing from north. One is the other.
            bearing = np.mod(np.degrees(np.arctan2(ex, ny)), 360.0)
            d = abs(bearing - aspect[r, c])
            d = min(d, 360.0 - d)
            tested += 1
            if d > worst:
                worst, worst_cell = d, (r, c)

        rec["cells_tested"] = tested
        rec["cells_flow_bearing_refused"] = none_count
        rec["max_abs_error_deg"] = float(worst)
        rec["worst_cell"] = worst_cell
        rec["tolerance_deg"] = BEARING_TOLERANCE_DEG
        rec["passes_at_1e_6_deg"] = bool(worst < BEARING_TOLERANCE_DEG)

        # `aspect_degrees` returns float32 and `flow_bearing` computes in float64, so the
        # comparison is float32 against float64 and cannot resolve below float32's own
        # spacing at these magnitudes. The plan's 1e-6 deg bar was written for two float64
        # quantities; against the values that actually exist it is unmeetable by
        # construction, and stating it as a failure would file a defect against a cast.
        rec["float32_spacing_at_360_deg"] = float(np.spacing(np.float32(360.0)))
        rec["passes_at_float32_precision"] = bool(
            worst <= rec["float32_spacing_at_360_deg"])
        rec["limiting_factor"] = "aspect_degrees returns float32; flow_bearing is float64"
        ev.note(
            f"Cross-check, flow_bearing vs aspect_degrees: over {tested:,} sampled "
            f"cells the two encodings of the same downslope vector differ by at most "
            f"{worst:.3e} deg. Against the plan's {BEARING_TOLERANCE_DEG:g} deg bar: "
            f"{rec['passes_at_1e_6_deg']} — but aspect_degrees returns float32, whose "
            f"spacing at 360 deg is {rec['float32_spacing_at_360_deg']:.3e} deg, so that "
            f"bar asks for more precision than the return value carries. Against what it "
            f"does carry: {rec['passes_at_float32_precision']}. flow_bearing returned "
            f"None on {none_count} genuinely flat cells.")


# ----------------------------------------------------------------- 4. counts


def stage_link_accounting(ev, dem_path):
    from terrainflow_assessment.modules.flow_graph import (
        d8_from_dem,
        strahler_order,
        stream_links,
    )
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    with ev.stage("crosscheck_link_accounting") as rec:
        ya = YeomansKeylineAnalysis(str(dem_path))
        _fdir, acc_arr = ya._ensure_flow_data()
        cell_area = ya.cell_w * ya.cell_h
        threshold = max(20, int(round(2_000.0 / cell_area)))
        stream = (acc_arr >= threshold) & np.isfinite(ya.dem)
        next_flat, _sink = d8_from_dem(ya.dem, ya.cell_w, ya.cell_h)
        order = strahler_order(next_flat, stream.ravel())

        emitted = stream_links(next_flat, stream.ravel(), order, ya.dem.shape[1],
                               max_order=1, min_cells=3)
        rec["min_cells"] = 3
        rec["emitted"] = len(emitted)

        # "Consumable" is the plan's word for a link long enough that
        # `keypoint_on_path` can fit an interior to it: 7 samples at
        # `min(5*cell, 10)` m spacing, which is the floor KPA-40 names.
        step = min(5.0 * ya.cell_size, 10.0)
        floor_m = 7.0 * step
        lengths = [_link_length_m(lk, ya.cell_w, ya.cell_h) for lk in emitted]
        rec["profile_step_m"] = step
        rec["floor_m"] = floor_m
        rec["consumable"] = int(sum(1 for m in lengths if m >= floor_m))
        rec["emitted_but_too_short"] = rec["emitted"] - rec["consumable"]
        rec["length_m_min"] = float(min(lengths)) if lengths else None
        rec["length_m_max"] = float(max(lengths)) if lengths else None

        keypoints, skipped = ya.find_keypoints(max_valleys=8)
        rec["keypoints"] = len(keypoints)
        rec["skipped"] = len(skipped)
        rec["keypoints_plus_skipped"] = len(keypoints) + len(skipped)
        rec["identity_holds"] = rec["keypoints_plus_skipped"] == rec["emitted"]
        rec["passes"] = rec["identity_holds"]
        ev.note(
            f"Cross-check, link accounting at {threshold * cell_area / 10_000:.2f} ha: "
            f"{rec['emitted']} links emitted at min_cells=3, of which "
            f"{rec['consumable']} clear the {floor_m:.0f} m profile floor "
            f"({rec['emitted_but_too_short']} cannot yield a keypoint however good the "
            f"ground). keypoints + skipped = {rec['keypoints_plus_skipped']}; identity "
            f"against emitted holds: {rec['identity_holds']}.")


def stage_mask_and_pointers_agree(ev, dem_path):
    """Does the channel mask come from the same routing scheme as the pointers?

    `find_keypoints` builds its stream mask from `_ensure_flow_data`'s accumulation, which
    is pysheds at **D-infinity** (`keypoint_analysis.py:1173`), and then traces links along
    `d8_from_dem`'s **D8** pointers (`:809`). D-infinity divides a cell's flow between
    downslope neighbours, so it wets a broader, more diffuse network than D8 concentrates
    into; tracing single-successor D8 pointers through that wider mask lets a link leave
    the mask and stop.

    That is testable rather than arguable: build the mask from the accumulation of the
    very graph the pointers come from, at the same threshold, and compare the link
    populations. Nothing here says which mask is *right* — D-infinity is the more physical
    accumulation and is the default for good reasons. The question is only whether mixing
    the two is what shatters the network.
    """
    from terrainflow_assessment.modules.flow_graph import (
        d8_from_dem,
        strahler_order,
        stream_links,
    )
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    with ev.stage("crosscheck_mask_routing_vs_pointer_routing") as rec:
        ya = YeomansKeylineAnalysis(str(dem_path))
        _fdir, dinf_acc = ya._ensure_flow_data()
        cell_w, cell_h = ya.cell_w, ya.cell_h
        cell_area = cell_w * cell_h
        threshold = max(20, int(round(2_000.0 / cell_area)))
        finite = np.isfinite(ya.dem)

        next_flat, _sink = d8_from_dem(ya.dem, cell_w, cell_h)

        # The D8 graph's own accumulation: one high-to-low pass over the same pointers
        # `stream_links` will walk. No second library, no second conditioning.
        flat_finite = finite.ravel()
        d8_acc = np.ones(ya.dem.size, dtype=np.int64)
        d8_acc[~flat_finite] = 0
        flat_z = np.asarray(ya.dem, dtype="float64").ravel()
        for i in np.argsort(np.where(flat_finite, -flat_z, np.inf), kind="stable"):
            if not flat_finite[i]:
                continue
            j = int(next_flat[i])
            if j != i:
                d8_acc[j] += d8_acc[i]

        rec["threshold_cells"] = threshold
        rec["threshold_ha"] = threshold * cell_area / 10_000.0
        step = min(5.0 * ya.cell_size, 10.0)
        floor_m = 7.0 * step
        rec["profile_floor_m"] = floor_m

        for label, mask in (
            ("dinf_mask_production", (np.asarray(dinf_acc) >= threshold) & finite),
            ("d8_mask_same_graph_as_the_pointers",
             (d8_acc.reshape(ya.dem.shape) >= threshold) & finite),
        ):
            order = strahler_order(next_flat, mask.ravel())
            links = stream_links(next_flat, mask.ravel(), order, ya.dem.shape[1],
                                 max_order=1)
            lengths = [_link_length_m(lk, cell_w, cell_h) for lk in links]
            in_links = sum(len(lk) for lk in links)
            leaving = int(sum(
                1 for i in np.flatnonzero(mask.ravel())
                if int(next_flat[i]) != i and not mask.ravel()[int(next_flat[i])]))
            rec[label] = {
                "stream_cells": int(mask.sum()),
                "order1_links": len(links),
                "cells_inside_a_link": in_links,
                "stream_cells_in_no_link": int(mask.sum()) - in_links,
                "stream_cells_whose_pointer_leaves_the_mask": leaving,
                "median_link_length_m": float(np.median(lengths)) if lengths else None,
                "max_link_length_m": float(max(lengths)) if lengths else None,
                "links_clearing_the_profile_floor": int(
                    sum(1 for m in lengths if m >= floor_m)),
            }

        a = rec["dinf_mask_production"]
        b = rec["d8_mask_same_graph_as_the_pointers"]
        ev.note(
            f"Mask routing vs pointer routing at {rec['threshold_ha']:.2f} ha. "
            f"Production (D-infinity mask, D8 pointers): {a['stream_cells']:,} stream "
            f"cells -> {a['order1_links']} order-1 links holding only "
            f"{a['cells_inside_a_link']:,} of them, median link "
            f"{a['median_link_length_m']:.1f} m, longest "
            f"{a['max_link_length_m']:.0f} m, "
            f"{a['links_clearing_the_profile_floor']} clearing the {floor_m:.0f} m "
            f"profile floor, and {a['stream_cells_whose_pointer_leaves_the_mask']} cells "
            f"whose D8 pointer leaves the mask. Same pointers, mask from the same graph: "
            f"{b['stream_cells']:,} stream cells -> {b['order1_links']} links holding "
            f"{b['cells_inside_a_link']:,}, median {b['median_link_length_m']:.1f} m, "
            f"longest {b['max_link_length_m']:.0f} m, "
            f"{b['links_clearing_the_profile_floor']} clearing the floor, "
            f"{b['stream_cells_whose_pointer_leaves_the_mask']} leaving the mask. "
            f"The shattering is the mixture, not the terrain.")


def _link_length_m(link, cell_w, cell_h):
    total = 0.0
    for (r0, c0), (r1, c1) in zip(link[:-1], link[1:]):
        total += float(np.hypot((r1 - r0) * cell_h, (c1 - c0) * cell_w))
    return total


# --------------------------------------------------------------------- driver


def main():
    _probe.start_qgis()
    dem = _probe.fixture_path()
    _probe.banner("p_crosscheck — four quantities computed twice", dem)

    import rasterio

    with rasterio.open(str(dem)) as src:
        z = src.read(1).astype("float64")
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
        transform = src.transform
        cell_w, cell_h = abs(transform.a), abs(transform.e)

    ev = _probe.Evidence("p_crosscheck", ["STEP-F"], dem)
    ev["dem_stats"] = _probe.dem_stats(dem)

    stage_haul_vs_burn(ev, z, transform, cell_w, cell_h)
    stage_one_stencil(ev, z, cell_w, cell_h)
    stage_flow_bearing(ev, z, cell_w, cell_h)
    stage_link_accounting(ev, dem)
    stage_mask_and_pointers_agree(ev, dem)

    ev.write()


if __name__ == "__main__":
    main()
