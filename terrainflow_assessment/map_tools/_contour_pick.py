"""_contour_pick.py — the contour search both contour-picking map tools share.

Both tools ask the same question: of the contours the project draws, which one
did the user just click? That used to be asked of a single layer — the analysed
"Candidate Contour Swales" — so a swale could only ever be drawn on a contour
the analysis had already ranked, slope-filtered and clipped to the usable area.
Every other contour on screen looked identical and was not pickable. The question
is asked of a list of layers now, so the plain generated contours count too.
"""

from qgis.core import QgsFeatureRequest, QgsGeometry, QgsRectangle

# The two layers a contour can arrive on spell elevation differently: the plain
# gdal:contour output carries gdal_contour's ELEV, the analysed candidates carry
# the ContourFeature attribute name. The rest are what a user's own contour layer
# might reasonably use, since any line layer can be handed to these tools.
_ELEV_FIELDS = ("ELEV", "elev", "elevation", "Elevation", "HEIGHT", "height")


def as_layers(layers):
    """One layer, or a sequence of them, or None → a list with the Nones dropped."""
    if layers is None:
        return []
    if isinstance(layers, (list, tuple, set)):
        return [layer for layer in layers if layer is not None]
    return [layers]


def nearest_contour(layers, point, radius):
    """The contour feature nearest *point* within *radius*, searched across *layers*.

    Returns ``(feature, alive)``. *feature* is None when nothing was in range.
    *alive* is False only when every layer given has been deleted out from under
    the tool, which is the one case a caller should cancel on — one dead layer
    among several is just a layer that offers no candidates.

    Layers are searched in the order given and an exact tie keeps the earlier
    one. Callers pass the analysed candidates first for that reason: a candidate
    contour and a plain contour at the same elevation are the same line drawn
    twice, and the candidate is the one carrying rank and inflow.
    """
    rect = QgsRectangle(
        point.x() - radius, point.y() - radius,
        point.x() + radius, point.y() + radius,
    )
    click_geom = QgsGeometry.fromPointXY(point)
    request = QgsFeatureRequest().setFilterRect(rect)

    best_feature = None
    best_dist = float("inf")
    alive = False
    for layer in as_layers(layers):
        try:
            features = list(layer.getFeatures(request))
        except RuntimeError:
            # This layer was deleted/swapped while the tool was active.
            continue
        alive = True
        for feature in features:
            dist = feature.geometry().distance(click_geom)
            if dist < best_dist:
                best_dist = dist
                best_feature = feature

    return best_feature, alive


def elevation_of(feature, default=0.0):
    """The feature's elevation, read from whichever field name carries it."""
    names = feature.fields().names()
    for fname in _ELEV_FIELDS:
        if fname in names:
            try:
                return float(feature[fname])
            except (ValueError, TypeError):
                return default
    return default
