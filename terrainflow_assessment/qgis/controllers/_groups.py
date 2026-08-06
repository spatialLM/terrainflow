"""
_groups.py — where every layer the plugin creates lands in the QGIS layer tree.

A straight run through the panel used to drop twenty-odd layers flat into the
legend in whatever order they happened to be made. Everything now goes into one
tree per site, arranged in the order the panel's stages produce it::

    Quail Island                        ← the panel's Site Name
      Baseline · 120mm·24h·C0.40·1ha
      Analysis · 120mm·24h·C0.40·1ha
        Contour Analysis
        Keypoint Analysis
      Design · 120mm·24h·C0.40·1ha
        Drawn Earthworks
        Earthwork Re-Analyses · 120mm·24h·C0.40·1ha
      Verify · 120mm·24h·C0.40·1ha

**The site group is keyed by name.** Type a new Site Name and re-run and a second
tree appears beside the first, which is how several analyses live in one project.
Keep the name and the existing groups are reused, their run tag rewritten in
place rather than piling up a near-identical duplicate every run.

Stage groups are found by a stable key stored as a custom property, *not* by
name — the name carries the run tag and therefore changes. Layers are added
collapsed: a raster's expanded colour ramp is over a hundred pixels tall, and
three of them bury the rest of the tree.
"""

from __future__ import annotations

from qgis.core import QgsLayerTree

_KEY_PROP = "terrainflow/group"

# Group paths, as passed by controllers.
SITE = ()
BASELINE = ("baseline",)
ANALYSIS = ("analysis",)
CONTOUR = ("analysis", "contour")
KEYPOINT = ("analysis", "keypoint")
DESIGN = ("design",)
DRAWN = ("design", "drawn")
RERUN = ("design", "rerun")
VERIFY = ("verify",)

# key → (display label, does the run tag get appended?). The tag is only worth
# repeating on groups whose contents are a function of the storm — "Drawn
# Earthworks" holds the same geometry whatever rainfall is assumed.
_LABELS = {
    "baseline": ("Baseline", True),
    "analysis": ("Analysis", True),
    "contour": ("Contour Analysis", False),
    "keypoint": ("Keypoint Analysis", False),
    "design": ("Design", True),
    "drawn": ("Drawn Earthworks", False),
    "rerun": ("Earthwork Re-Analyses", True),
    "verify": ("Verify", True),
}

# Sibling order under each parent key (None = directly under the site group).
_ORDER = {
    None: ["baseline", "analysis", "design", "verify"],
    "analysis": ["contour", "keypoint"],
    "design": ["drawn", "rerun"],
}

DEFAULT_SITE_NAME = "Unnamed Site"


# ---------------------------------------------------------------- lookup / creation

def _is_group(node):
    try:
        return QgsLayerTree.isGroup(node)
    except Exception:
        return False


def site_group(project, site_name=""):
    """The root group for *site_name*, created at the top of the tree if absent.

    Matched against the **direct** children of the tree root only. ``findGroup``
    recurses, and would happily return a nested stage group that shares the name.
    """
    root = project.instance().layerTreeRoot()
    name = (site_name or "").strip() or DEFAULT_SITE_NAME
    for child in root.children():
        if _is_group(child) and child.name() == name:
            return child
    grp = root.insertGroup(0, name)
    grp.setCustomProperty(_KEY_PROP, "site")
    grp.setExpanded(True)
    return grp


def rename_default_site(project, site_name):
    """Give the auto-named group the name the user has just typed.

    Only ever touches a group still called "Unnamed Site". That is not a name
    anyone chose, so adopting it keeps a run made *before* the name was entered
    together with the rest of its analysis instead of splitting one assessment
    across two trees. A group with a real name is left alone — renaming that is
    how an earlier analysis would be lost.
    """
    name = (site_name or "").strip()
    if not name or name == DEFAULT_SITE_NAME:
        return None
    root = project.instance().layerTreeRoot()
    default = None
    for child in root.children():
        if not _is_group(child):
            continue
        if child.name() == name:
            return None      # this name is already a tree of its own
        if (child.name() == DEFAULT_SITE_NAME
                and child.customProperty(_KEY_PROP) == "site"):
            default = child
    if default is not None:
        default.setName(name)
    return default


def _insert_index(parent, parent_key, key):
    """Where a new *key* group belongs among *parent*'s existing children.

    One past the last sibling it should follow: any stage group declared before
    it, plus — everywhere except the site group — any loose layer, so a subgroup
    such as "Contour Analysis" sits under the plain Analysis layers rather than
    above them.
    """
    order = _ORDER.get(parent_key, [])
    want = order.index(key) if key in order else len(order)
    groups_only = parent_key is None
    idx = 0
    for i, child in enumerate(parent.children()):
        if _is_group(child):
            child_key = child.customProperty(_KEY_PROP)
            rank = order.index(child_key) if child_key in order else len(order)
            if rank < want:
                idx = i + 1
        elif not groups_only:
            idx = i + 1
    return idx


def _ensure_child(parent, parent_key, key, tag):
    label, tagged = _LABELS.get(key, (str(key).title(), False))
    name = f"{label} · {tag}" if tagged and tag else label

    for child in parent.children():
        if _is_group(child) and child.customProperty(_KEY_PROP) == key:
            # Same group, new storm — rename rather than leaving the old tag
            # describing layers that have just been replaced.
            if child.name() != name:
                child.setName(name)
            return child

    grp = parent.insertGroup(_insert_index(parent, parent_key, key), name)
    grp.setCustomProperty(_KEY_PROP, key)
    grp.setExpanded(True)
    return grp


def group(project, path=SITE, site_name="", tag=""):
    """Resolve (creating as needed) the group at *path* under *site_name*."""
    node = site_group(project, site_name)
    parent_key = None
    for key in path:
        node = _ensure_child(node, parent_key, key, tag)
        parent_key = key
    return node


# ---------------------------------------------------------------- placing layers

def add_layer(project, layer, path=SITE, site_name="", tag="",
              visible=True, at_top=False):
    """Register *layer* with the project and place it in the group at *path*.

    Added with ``addMapLayer(layer, False)`` so QGIS does not also drop it at the
    top of the flat legend, then attached to the group — and collapsed, which is
    the whole point of routing every layer through here.
    """
    grp = group(project, path, site_name, tag)
    project.instance().addMapLayer(layer, False)
    node = grp.insertLayer(0, layer) if at_top else grp.addLayer(layer)
    if node is not None:
        node.setExpanded(False)
        if not visible:
            node.setItemVisibilityChecked(False)
    return node


def clear_group(project, path=SITE, site_name="", tag=""):
    """Drop every layer inside the group at *path*, keeping the group itself.

    Re-running a stage replaces its outputs; without this the old rasters stay
    behind under identical names.
    """
    grp = group(project, path, site_name, tag)
    for child in list(grp.findLayers()):
        try:
            project.instance().removeMapLayer(child.layerId())
        except Exception:
            pass
    return grp


def collapse(project, layer):
    """Collapse an already-placed layer's legend, ignoring a dead reference."""
    if layer is None:
        return
    try:
        node = project.instance().layerTreeRoot().findLayer(layer.id())
    except Exception:
        return
    if node is not None:
        node.setExpanded(False)


class LayerTreeMixin:
    """Gives a controller ``self.place(layer, G.ANALYSIS)``.

    Mixed into the controllers rather than repeated in each of them; it reads
    ``_project`` / ``_panel`` / ``_state``, which all five already hold.
    """

    def place(self, layer, path=SITE, visible=True, at_top=False):
        return add_layer(
            self._project, layer, path,
            site_name=getattr(self._panel, "site_name", ""),
            tag=getattr(self._state, "run_tag", ""),
            visible=visible, at_top=at_top,
        )

    def group_for(self, path=SITE):
        return group(
            self._project, path,
            site_name=getattr(self._panel, "site_name", ""),
            tag=getattr(self._state, "run_tag", ""),
        )
