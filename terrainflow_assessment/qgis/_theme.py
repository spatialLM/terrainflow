"""The panel's chrome: colours, glyphs and the one-line brush helper.

CLAUDE.md gives *raster* colours a single home in `core/registry/map_palette.py`.
Chrome had none. `_INK` was declared in four files, `_MUTED` in four, `_BAD` in three,
`#1273b5` under four different names in five files — each copy carrying a comment
asserting a shared source that did not exist, and `def _brush` written out verbatim
three times. Every value agreed, which is the point: nothing had gone wrong yet, and
nothing would announce it when it did. A palette maintained by hand in eight places
drifts the first time one of them is edited alone.

This is that home. The names are roles, not colours — `MUTED` says what the text is
for, `#5f7176` says what it looks like — so a widget reads the intent and the value
lives in one place.

**The rule from `map_palette` carries over: a colour has one name here.** If you need
a new shade, add it here; do not redeclare it in the widget that wants it.
"""

# --- Ink ---------------------------------------------------------------------
#: Body text. Near-black with a green cast, so it sits with the map rather than on it.
INK = "#22302e"
#: Secondary text — units, row subtitles, anything that qualifies a number.
MUTED = "#5f7176"
#: Tertiary text: present, deliberately hard to read, never carrying a fact alone.
FAINT = "#8fa0a4"

# --- Surfaces ----------------------------------------------------------------
#: Table and card background.
SURFACE = "#ffffff"
#: The panel behind the surfaces — alternating rows, group backgrounds.
GROUND = "#eef1f2"
#: Ordinary separators.
HAIRLINE = "#dde4e5"
#: A separator that is doing more work: section breaks, table outlines.
HAIRLINE_STRONG = "#c6d1d3"

# --- Status ------------------------------------------------------------------
#: Passes, meets, within tolerance.
GOOD = "#1e8449"
#: Worth a look, not a failure.
WARN = "#b9770e"
#: Fails, exceeds, cannot be built.
BAD = "#c0392b"

# --- Water and growth --------------------------------------------------------
#: Surface water, flow, outflow — the blue that means "this is water moving".
WATER = "#1273b5"
#: Water that soaked in rather than ran off. Reads as a lighter WATER on purpose.
SOAKED = "#79b8dd"
#: Inflow, an active step, a living thing. The green counterpart to WATER.
GROWTH = "#2e7d55"
#: Vegetation/structure grey-green, used where LEAVES is a surface, not a state.
LEAVES = "#c6d1d3"

#: Small monochrome glyphs per earthwork type, kept in the BMP so Qt renders them on
#: a DejaVu host. Byte-identical copies of this dict lived in `tool_menu` and
#: `network_view`; a type added to one and not the other showed a bullet in the other.
GLYPH = {
    "swale": "∿",
    "basin": "▢",
    "dam": "▮",
    "berm": "⌒",
    "diversion": "↘",
}
#: What an unknown type falls back to.
GLYPH_UNKNOWN = "●"

#: Spillway direction, as (glyph, colour). Water leaves through the outflow and arrives
#: through the inflow, so the pair is blue-down / green-up wherever it appears — the
#: tool menu, the spillway table and the map layer all read these.
SPILLWAY_OUT = ("▽", WATER)
SPILLWAY_IN = ("▲", GROWTH)


#: The Run button's accent, its hover, and the ghost fill it uses while disabled.
#: Darker than WATER on purpose — a button is a control, not a reading of the site.
ACCENT = "#15628f"
ACCENT_HOVER = "#0f4f75"
ACCENT_GHOST = "#eef4f8"


def brush(hex_colour):
    """A `QBrush` of *hex_colour*.

    The Qt import stays in the body. These widgets are imported by
    `tests/test_architecture.py`'s import-time scan and by the pure suite, where the
    QGIS runtime is mocked; a module-level `QtGui` import here would make this module
    unimportable outside QGIS for the sake of a three-line helper.
    """
    from qgis.PyQt.QtGui import QBrush, QColor

    return QBrush(QColor(hex_colour))
