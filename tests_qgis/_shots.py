"""
_shots.py — render real widgets and the real map canvas to PNG, then inspect them.

Tier 2 of the test harness. Tier 1 proves the code runs and the state is right;
this proves something was actually *drawn*, and leaves an image behind that a
human (or Claude) can look at.

Works headless because the harness runs Qt on the offscreen platform with GUI
enabled — the widgets are real and laid out, they simply never reach a screen.

Images land in tests_qgis/_shots/ and are overwritten by name each run, so the
directory is always the latest run rather than an ever-growing pile.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SHOTS_DIR = HERE / "_shots"
BASELINE_DIR = HERE / "_shots_baseline"

# Records which images last differed and for how many consecutive runs, so the
# reminder can escalate. A baseline that stays stale is worse than no baseline:
# the same names scroll past every run, get tuned out, and a real regression
# hides among them.
DIFF_STATE = BASELINE_DIR / "diff_state.json"


def shots_dir():
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return SHOTS_DIR


def baseline_dir():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    return BASELINE_DIR


def _settle(passes=5):
    """Let Qt finish layout and painting — no event loop is running for us."""
    from qgis.PyQt.QtCore import QCoreApplication

    for _ in range(passes):
        QCoreApplication.processEvents()


def save_widget(widget, name, size=None):
    """Grab a live widget to tests_qgis/_shots/<name>.png and return the path."""
    window = widget.window()
    window.show()
    widget.show()
    if size is not None:
        widget.resize(*size)
    _settle()

    pixmap = widget.grab()
    path = shots_dir() / f"{name}.png"
    if not pixmap.save(str(path)):
        raise AssertionError(f"failed to write widget screenshot: {path}")
    print(f"\n    shot: {path.name} ({pixmap.width()}x{pixmap.height()})", end="")
    return path


def save_canvas(canvas, name, size=(1200, 900)):
    """Render the map canvas to tests_qgis/_shots/<name>.png and return the path."""
    canvas.resize(*size)
    canvas.window().show()
    canvas.show()
    _settle()
    canvas.refresh()
    canvas.waitWhileRendering()
    _settle()

    path = shots_dir() / f"{name}.png"
    canvas.saveAsImage(str(path))
    if not path.exists():
        raise AssertionError(f"canvas produced no image: {path}")
    print(f"\n    shot: {path.name}", end="")
    return path


# ---------------------------------------------------------------------------
# Inspection — enough to tell "drawn" from "blank" without extra dependencies
# ---------------------------------------------------------------------------

def pixels(path):
    """Load a PNG as an (h, w, 3) uint8 RGB array, or None if unreadable.

    Read through rasterio (GDAL) rather than QImage so that image analysis needs no
    QApplication — the run_all.py orchestrator compares images without booting
    QGIS, and only the worker subprocesses have a Qt application.
    """
    import warnings

    import numpy as np
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    try:
        with warnings.catch_warnings():
            # Screenshots are plain PNGs; GDAL warning about a missing geotransform
            # on every single read is pure noise here.
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(str(path)) as src:
                arr = src.read()            # (bands, h, w)
    except Exception:
        return None
    if arr.shape[0] < 3:                # greyscale -> fake three channels
        arr = np.repeat(arr[:1], 3, axis=0)
    return np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)


def describe(path):
    """Report size, exact distinct-colour count, and the dominant colour's share.

    Every pixel is counted, not a sampled grid: a 1-cell-wide stream line is only
    a handful of pixels across, and a sampling grid coarse enough to be fast in
    pure Python steps straight over it — reporting a layer that clearly drew as
    blank. numpy makes the exact count cheap.
    """
    import numpy as np

    arr = pixels(path)
    if arr is None:
        return {"path": str(path), "null": True, "width": 0, "height": 0, "colours": 0,
                "dominant_share": 1.0, "ink_share": 0.0}

    height, width = arr.shape[0], arr.shape[1]

    packed = (
        arr[:, :, 0].astype(np.uint32)
        | (arr[:, :, 1].astype(np.uint32) << 8)
        | (arr[:, :, 2].astype(np.uint32) << 16)
    )
    values, counts = np.unique(packed, return_counts=True)
    total = packed.size
    dominant = int(counts.max())

    return {
        "path": str(path),
        "null": False,
        "width": width,
        "height": height,
        "colours": int(values.size),
        "dominant_share": dominant / total,
        # Share of pixels that are NOT the most common colour — i.e. actual marks
        # on an otherwise flat background.
        "ink_share": 1.0 - dominant / total,
    }


def _read_diff_state():
    import json

    try:
        return json.loads(DIFF_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"names": [], "runs": 0}


def _write_diff_state(names):
    """Record the outstanding difference, incrementing the streak if unchanged."""
    import json

    previous = _read_diff_state()
    if not names:
        clear_diff_state()
        return 0

    runs = previous.get("runs", 0) + 1 if previous.get("names") == names else 1
    baseline_dir().mkdir(parents=True, exist_ok=True)
    DIFF_STATE.write_text(
        json.dumps({"names": names, "runs": runs}, indent=2), encoding="utf-8"
    )
    return runs


def clear_diff_state():
    try:
        DIFF_STATE.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def snapshot_baseline():
    """Copy the current shots to _shots_baseline/ as the comparison point.

    Run this *before* making a change, so the next run can say which images the
    change actually moved instead of relying on someone's memory of the last one.
    """
    import shutil

    src, dst = shots_dir(), baseline_dir()
    for stale in dst.glob("*.png"):
        stale.unlink()

    names = []
    for png in sorted(src.glob("*.png")):
        shutil.copy2(png, dst / png.name)
        names.append(png.name)

    clear_diff_state()      # accepted: the streak starts over
    return names


def compare_to_baseline(since=None):
    """Compare _shots/*.png against _shots_baseline/*.png.

    `since` is a unix timestamp: images not rewritten during this run are skipped,
    so comparing after a partial run (e.g. `checks_visual` only) does not report
    every other image as unchanged.
    """
    import numpy as np

    rows = []
    for png in sorted(shots_dir().glob("*.png")):
        if since is not None and png.stat().st_mtime < since:
            continue

        reference = BASELINE_DIR / png.name
        if not reference.exists():
            rows.append({"name": png.name, "state": "new"})
            continue

        now, before = pixels(png), pixels(reference)
        if now is None or before is None:
            rows.append({"name": png.name, "state": "unreadable"})
            continue

        if now.shape != before.shape:
            rows.append({
                "name": png.name,
                "state": "resized",
                "detail": f"{before.shape[1]}x{before.shape[0]}"
                          f" -> {now.shape[1]}x{now.shape[0]}",
            })
            continue

        delta = now.astype(np.int16) - before.astype(np.int16)
        differing = np.any(delta != 0, axis=2)
        count = int(differing.sum())
        if count == 0:
            rows.append({"name": png.name, "state": "unchanged"})
        else:
            rows.append({
                "name": png.name,
                "state": "changed",
                "share": count / differing.size,
                "max_delta": int(np.abs(delta).max()),
            })

    return rows


def print_comparison(rows):
    """Human-readable diff summary. Changes are reported, never treated as failures."""
    if not rows:
        print("\nNo images to compare against the baseline.")
        return

    changed = [r for r in rows if r["state"] in ("changed", "resized")]
    new = [r for r in rows if r["state"] == "new"]
    same = [r for r in rows if r["state"] == "unchanged"]

    print("\n" + "-" * 72)
    print(f"Visual diff vs baseline: {len(changed)} changed, {len(new)} new, "
          f"{len(same)} unchanged")
    print("-" * 72)

    for row in changed:
        if row["state"] == "resized":
            print(f"  RESIZED   {row['name']}  {row['detail']}")
        else:
            print(f"  CHANGED   {row['name']}  "
                  f"{row['share']:.2%} of pixels, max channel delta {row['max_delta']}")
    for row in new:
        print(f"  NEW       {row['name']}  (no baseline to compare)")

    outstanding = sorted(r["name"] for r in changed + new)
    runs = _write_diff_state(outstanding)

    if not outstanding:
        print("  every image is pixel-identical to the baseline")
        return

    print(f"\n  baseline: {BASELINE_DIR}")
    print(f"  current:  {SHOTS_DIR}")
    print("\n  Look at the changed image(s). If that is the change you intended,")
    print("  accept them as the new reference:")
    print("\n      .\\run_qgis_tests.ps1 -Accept        (no re-run, just accept)")
    print("      .\\run_qgis_tests.ps1 -Snapshot      (re-run, then accept)")

    if runs >= 2:
        print(
            f"\n  ** REMINDER: the same {len(outstanding)} image(s) have differed from "
            f"the baseline for {runs} runs in a row. **"
        )
        print("  Accept them or investigate — until then this section is noise, and a")
        print("  genuine regression will hide among the names you have learned to skip.")


def assert_rendered(path, context="", min_colours=8, max_dominant=0.995):
    """Fail if the image is empty, single-colour, or effectively one flat wash.

    Deliberately loose: this catches "nothing rendered" and "the layer is invisible",
    which is the class of bug a state assertion cannot see. It is not a
    pixel-diff — the UI is being actively restyled, so exact-image goldens would
    fail on every legitimate change.
    """
    info = describe(path)
    if info["null"]:
        raise AssertionError(f"{context}: unreadable image at {path}")
    if info["width"] < 50 or info["height"] < 50:
        raise AssertionError(
            f"{context}: image is {info['width']}x{info['height']} — widget never laid out ({path})"
        )
    if info["colours"] < min_colours:
        raise AssertionError(
            f"{context}: only {info['colours']} distinct colours — looks blank ({path})"
        )
    if info["dominant_share"] > max_dominant:
        raise AssertionError(
            f"{context}: {info['dominant_share']:.1%} of sampled pixels are one colour "
            f"— nothing meaningful rendered ({path})"
        )
    return info
