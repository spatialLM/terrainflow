"""p_gate_reads — price the "read the whole band to sample a few cells" family.

Three Batch 4 items share one mechanism: M-10 (`snap_point_to_contour_elevation`
reads the full band for one cell), Q-11 (`_calc_dam_wall_metrics` reads the full
band per spin-box tick to sample ~60 cells) and G-9 (the draw tools read the whole
slope band in `__init__`, once per draw action). Before implementing any of them,
price the read itself on the real design — if a full band read is cheap, every one
of the three is unwinnable and none should be coded (gate rule 1).

    $env:PYTHONPATH="F:\\Terrain Flow Design\\TerrainFlow"
    & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_gate_reads.py
"""
import faulthandler
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

faulthandler.enable()

from p_perf_gate import bench, real_dem  # noqa: E402


def main():
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    dem = real_dem()
    with rasterio.open(dem) as src:
        print(f"DEM {src.width}x{src.height} {src.dtypes[0]} "
              f"compress={src.profile.get('compress')} "
              f"({src.width * src.height / 1e6:.2f} M cells)")
        row, col = src.height // 2, src.width // 2

    def full_read():
        with rasterio.open(dem) as src:
            return float(src.read(1)[row, col])

    def full_read_astype():
        # what Q-11 and G-9 actually do: read, then copy to float32
        with rasterio.open(dem) as src:
            return src.read(1).astype("float32")

    def windowed_read():
        with rasterio.open(dem) as src:
            return float(src.read(1, window=Window(col, row, 1, 1))[0, 0])

    def open_only():
        with rasterio.open(dem) as src:
            return src.transform

    t_full = bench(full_read, repeat=5)
    t_astype = bench(full_read_astype, repeat=5)
    t_win = bench(windowed_read, repeat=5)
    t_open = bench(open_only, repeat=5)

    print(f"  open only                {t_open * 1000:8.2f} ms")
    print(f"  open + read(1)[r, c]     {t_full * 1000:8.2f} ms")
    print(f"  open + read(1).astype    {t_astype * 1000:8.2f} ms")
    print(f"  open + windowed 1x1      {t_win * 1000:8.2f} ms")
    print(f"  -> a full band read costs {(t_full - t_win) * 1000:.1f} ms more "
          f"than a windowed one")

    # The slope raster G-9 reads is a sibling product on the same grid; if it is not
    # on disk, the DEM's own size is the right stand-in (same shape, same dtype).
    arr = np.empty((0,))
    del arr


if __name__ == "__main__":
    main()
