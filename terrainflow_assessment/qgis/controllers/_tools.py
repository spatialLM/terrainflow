"""_tools.py — the active map tool, held where it can be found again.

Two reasons a controller must keep a Python reference to the tool it activates.

The obvious one is lifetime. ``QgsMapCanvas.setMapTool`` does not take ownership,
so a tool built as a local goes out of scope when the method returns and the canvas
is left pointing at an object Python has collected. ``baseline.py`` carried a comment
saying exactly this ("keep a reference so the tool is not garbage-collected"); eight
other activation sites did not follow it, and the codebase disagreed with itself
about whether the comment was describing a real hazard or superstition.

The second is teardown. ``unload()`` has to take the plugin's tool off the canvas
before the plugin goes away, or a click on the map arrives at a controller whose
world has been dismantled. It cannot do that without knowing which tools are ours —
``canvas.mapTool()`` alone does not say who owns the thing it returns.
"""


class MapToolMixin:
    """Gives a controller ``use_tool`` / ``release_tool``.

    Mixed in beside :class:`LayerTreeMixin`, and like it, reads ``_canvas``, which
    every controller that activates a tool already holds.
    """

    #: The tool this controller last activated, or None. Assigning through
    #: ``self`` shadows this with an instance attribute, so controllers do not
    #: share one.
    _active_tool = None

    def use_tool(self, tool):
        """Activate *tool* on the canvas and hold it against collection."""
        self._active_tool = tool
        self._canvas.setMapTool(tool)
        return tool

    def release_tool(self):
        """Take this controller's tool off the canvas, if it is still on it.

        Deliberately not ``unsetMapTool(canvas.mapTool())``: by the time teardown
        runs, the active tool may belong to QGIS or to another plugin, and pulling
        that one off is not ours to do.
        """
        tool = getattr(self, "_active_tool", None)
        self._active_tool = None
        if tool is None:
            return
        try:
            if self._canvas.mapTool() is tool:
                self._canvas.unsetMapTool(tool)
        except RuntimeError:
            # The C++ side has already been destroyed; there is nothing to unset
            # and asking again would raise the same thing.
            pass
