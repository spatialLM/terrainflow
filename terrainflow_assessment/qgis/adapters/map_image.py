"""
map_image.py — render a map off-screen, without touching the user's canvas.

A thin wrapper over QGIS's own renderer. The report needs maps at a chosen
extent, size and DPI, drawn from an explicit layer list — none of which
``QgsMapCanvas.saveAsImage`` can give you: it is locked to the canvas's current
size and extent, and it omits rubber bands (``tests_qgis/README.md``).

Layer order is **top-first**, the opposite of how the layer tree reads. Getting
it backwards puts the DEM over everything.
"""

import logging

_log = logging.getLogger(__name__)


def fit_extent(extent, width_px, height_px, margin=0.05):
    """Widen ``extent`` to the image aspect ratio, plus a margin.

    Without this the map comes out stretched: QGIS honours the output size and
    squeezes the extent into it rather than adjusting one for the other.
    """
    from qgis.core import QgsRectangle

    out = QgsRectangle(extent)
    if margin:
        out.scale(1.0 + margin)
    if not width_px or not height_px or out.width() <= 0 or out.height() <= 0:
        return out

    want = float(width_px) / float(height_px)
    have = out.width() / out.height()
    cx, cy = out.center().x(), out.center().y()
    if have < want:
        half = out.height() * want / 2.0
        out.setXMinimum(cx - half)
        out.setXMaximum(cx + half)
    else:
        half = out.width() / want / 2.0
        out.setYMinimum(cy - half)
        out.setYMaximum(cy + half)
    return out


def usable_layers(layers):
    """Drop layers that are missing or whose source has gone.

    ``resolve_layer`` returns ``None`` for a layer the user deleted, but a layer
    whose *source* vanished — everything under ``state.output_dir``, which is
    rmtree'd on plugin unload — still resolves and would render blank. Both have
    to be filtered or the report prints an empty map with no explanation.
    """
    out = []
    for layer in layers or []:
        if layer is None:
            continue
        try:
            if not layer.isValid():
                continue
        except Exception:
            continue
        out.append(layer)
    return out


def layers_extent(layers, crs=None):
    """Combined extent of ``layers``, in ``crs`` if given. None when empty.

    Accumulates from the first usable rectangle rather than seeding an empty one
    — ``QgsRectangle.setMinimal()`` is deprecated, and starting from ``None``
    also gives a clean "nothing to draw" answer.
    """
    from qgis.core import QgsCoordinateTransform, QgsProject, QgsRectangle

    combined = None
    for layer in layers:
        try:
            rect = layer.extent()
            if rect.isEmpty():
                continue
            if crs is not None and layer.crs() != crs:
                rect = QgsCoordinateTransform(
                    layer.crs(), crs,
                    QgsProject.instance()).transformBoundingBox(rect)
            if combined is None:
                combined = QgsRectangle(rect)
            else:
                combined.combineExtentWith(rect)
        except Exception as exc:
            _log.debug("skipping layer extent: %s", exc)
    return combined if combined is not None and not combined.isEmpty() else None


def render_map_image(layers, extent=None, size_px=(1200, 800), dpi=200,
                     crs=None, background=(255, 255, 255), labels=True,
                     decorations=True):
    """Render ``layers`` to a ``QImage``. Returns ``None`` if nothing is drawable.

    ``dpi`` drives millimetre-based symbol and label sizes, so it must be set
    before the render rather than scaled afterwards.

    ``decorations`` paints a scale bar and a north arrow over the finished
    image. The PDF gets its own as layout items; without these the two formats
    would carry different maps, and a site plan with no scale is not a document
    anyone can build from.
    """
    from qgis.core import QgsMapRendererParallelJob, QgsMapSettings
    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QColor

    drawable = usable_layers(layers)
    if not drawable:
        return None

    width, height = int(size_px[0]), int(size_px[1])
    ms = QgsMapSettings()
    ms.setLayers(drawable)                     # index 0 draws on TOP
    if crs is not None:
        ms.setDestinationCrs(crs)
    else:
        crs = drawable[0].crs()
        ms.setDestinationCrs(crs)
    ms.setOutputSize(QSize(width, height))     # size before extent
    ms.setOutputDpi(dpi)

    if extent is None:
        extent = layers_extent(drawable, crs)
        if extent is None:
            return None
    shown = fit_extent(extent, width, height)
    ms.setExtent(shown)
    ms.setBackgroundColor(QColor(*background))
    ms.setFlag(QgsMapSettings.Antialiasing, True)
    ms.setFlag(QgsMapSettings.DrawLabeling, bool(labels))
    ms.setFlag(QgsMapSettings.UseAdvancedEffects, True)
    # Selection is interactive state. Whatever the operator had picked when
    # they hit Export has no meaning to the person reading the report, and a
    # highlighted feature among forty reads as a finding — this flag is on by
    # default, which is how a stuck contour selection reached the map PNGs.
    ms.setFlag(QgsMapSettings.DrawSelection, False)

    job = QgsMapRendererParallelJob(ms)
    job.start()
    job.waitForFinished()
    image = job.renderedImage()
    if image is not None and not image.isNull() and decorations:
        draw_decorations(image, shown, projected=not crs.isGeographic())
    return image


# --------------------------------------------------------------------------- decorations

#: Bar lengths worth printing, in metres. A scale bar reading "137 m" is not a
#: scale bar — the point is that the reader can count it off.
_NICE_LENGTHS = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500,
                 1000, 2000, 2500, 5000, 10000, 20000, 50000)


def nice_bar_length(target_m):
    """The largest tidy length not exceeding ``target_m`` (never zero)."""
    usable = [n for n in _NICE_LENGTHS if n <= target_m]
    return usable[-1] if usable else _NICE_LENGTHS[0]


def _label_text(metres):
    return f"{metres / 1000:g} km" if metres >= 1000 else f"{metres:g} m"


def draw_decorations(image, extent, margin_px=None, projected=True):
    """Paint a scale bar and a north arrow onto a rendered map.

    Metres per pixel comes straight from the extent the image was rendered at,
    so the bar is exact rather than estimated — which is also why ``extent``
    must be the *fitted* extent and not the one that was asked for: ``setExtent``
    widens to the output aspect and the two differ.

    That arithmetic is only metres if the extent is. ``projected=False`` says the
    map was rendered in a geographic CRS, where the extent is in degrees and the
    same division is out by roughly five orders of magnitude — a bar that looks
    entirely reasonable and is wrong. The north arrow still holds, so it is drawn
    and the bar is not: a missing scale is a visible absence, and an incorrect one
    is not.

    Both decorations sit on a translucent white plate. The maps underneath run
    from bare white to dark bush to deep navy water, and no corner is reliably
    light enough to take plain dark ink.
    """
    from qgis.PyQt.QtCore import QRectF, Qt
    from qgis.PyQt.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF

    width, height = image.width(), image.height()
    if not width or not height or extent is None or extent.width() <= 0:
        return image
    metres_per_px = extent.width() / float(width)
    margin = int(margin_px if margin_px is not None
                 else max(8, round(width * 0.012)))
    font_px = max(9, int(round(height * 0.018)))

    bar_m = nice_bar_length(width * 0.22 * metres_per_px)
    bar_px = bar_m / metres_per_px
    bar_h = max(4, int(round(height * 0.008)))

    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        font = QFont("Arial", font_px)
        font.setPixelSize(font_px)
        font.setBold(True)
        painter.setFont(font)
        ink = QColor("#22302e")

        # ---- scale bar, bottom left. Omitted outright on a geographic CRS: the
        # extent is in degrees there and any length this drew would be fiction.
        if projected:
            plate_w = bar_px + 2 * margin * 0.6
            plate_h = bar_h + font_px + margin * 0.9
            left = margin
            top = height - margin - plate_h
            _plate(painter, QRectF(left, top, plate_w, plate_h), QColor, QBrush,
                   QPen, Qt)

            x0 = left + margin * 0.6
            y0 = top + plate_h - margin * 0.45 - bar_h
            # Two segments, filled and hollow, so the bar reads as halves.
            painter.setPen(QPen(ink, max(1.0, bar_h * 0.22)))
            for i in range(2):
                rect = QRectF(x0 + i * bar_px / 2.0, y0, bar_px / 2.0, bar_h)
                painter.setBrush(QBrush(ink if i == 0 else QColor(255, 255, 255)))
                painter.drawRect(rect)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(ink))
            painter.drawText(
                QRectF(x0, top + margin * 0.15, bar_px, font_px * 1.3),
                Qt.AlignLeft | Qt.AlignVCenter, f"0 — {_label_text(bar_m)}")

        # ---- north arrow, top right. Every map here is north-up.
        head = max(14, int(round(height * 0.035)))
        box = head * 0.75
        nx = width - margin - box
        ny = margin
        _plate(painter, QRectF(nx - box * 0.35, ny - box * 0.1,
                               box * 1.7, head + font_px * 1.5),
               QColor, QBrush, QPen, Qt)
        painter.setBrush(QBrush(ink))
        painter.setPen(QPen(QColor(255, 255, 255), max(1.0, head * 0.06)))
        painter.drawPolygon(QPolygonF([
            _qpoint(nx + box / 2.0, ny),
            _qpoint(nx + box, ny + head),
            _qpoint(nx + box / 2.0, ny + head * 0.72),
            _qpoint(nx, ny + head),
        ]))
        painter.setPen(QPen(ink))
        painter.setBrush(Qt.NoBrush)
        painter.drawText(QRectF(nx, ny + head, box, font_px * 1.4),
                         Qt.AlignHCenter | Qt.AlignVCenter, "N")
    finally:
        painter.end()
    return image


def _qpoint(x, y):
    from qgis.PyQt.QtCore import QPointF
    return QPointF(x, y)


def _plate(painter, rect, colour_cls, brush_cls, pen_cls, qt):
    """The translucent backing every decoration sits on."""
    painter.setPen(pen_cls(colour_cls(0, 0, 0, 40)))
    painter.setBrush(brush_cls(colour_cls(255, 255, 255, 205)))
    painter.drawRoundedRect(rect, 3.0, 3.0)
    painter.setBrush(qt.NoBrush)


def save_map_png(path, layers, **kwargs):
    """Render and write a PNG. Returns the path, or ``None`` if nothing drew."""
    image = render_map_image(layers, **kwargs)
    if image is None:
        return None
    return path if image.save(path, "PNG") else None
