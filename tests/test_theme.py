"""G-11. The panel chrome has one home, and the widgets must keep using it.

`map_palette.py` has always been the single source for *raster* colour. Chrome had
none: `_INK` was declared in four widget files, `_MUTED` in four, `_BAD` in three, and
`#1273b5` appeared in five files under four different names — `_FLOW`, `_WATER`,
`_INFO`, `_OUT_COLOUR` — each with a comment asserting a shared source that did not
exist. `def _brush` was written out verbatim three times.

Every copy agreed, which is exactly why it needed fixing before rather than after: a
palette kept by hand in eight places is correct until the first time one of them is
edited alone, and nothing about that edit announces itself.
"""

import pathlib
import re

import pytest

from terrainflow_assessment.qgis import _theme

WIDGETS = pathlib.Path(_theme.__file__).parent / "widgets"

#: The palette as it stood when it was consolidated. Pinned so that changing a colour
#: is a deliberate act with a diff, rather than something that happens to one widget.
EXPECTED = {
    "INK": "#22302e",
    "MUTED": "#5f7176",
    "FAINT": "#8fa0a4",
    "SURFACE": "#ffffff",
    "GROUND": "#eef1f2",
    "HAIRLINE": "#dde4e5",
    "HAIRLINE_STRONG": "#c6d1d3",
    "GOOD": "#1e8449",
    "WARN": "#b9770e",
    "BAD": "#c0392b",
    "WATER": "#1273b5",
    "SOAKED": "#79b8dd",
    "GROWTH": "#2e7d55",
    "LEAVES": "#c6d1d3",
    "ACCENT": "#15628f",
    "ACCENT_HOVER": "#0f4f75",
    "ACCENT_GHOST": "#eef4f8",
}

#: A widget may still spell a colour out when it is genuinely its own. One does.
#:
#: `network_view.py`'s connector grid is `#eef1f0`, which is one digit off `GROUND`'s
#: `#eef1f2`. Nobody now knows whether that was intended or a typo, and the two are
#: indistinguishable on screen — so it is recorded here rather than quietly "corrected"
#: into GROUND, which would be a rendering change made on a guess.
_ALLOWED_LITERALS = {"#eef1f0"}

_HEX = re.compile(r'"(#[0-9a-fA-F]{6})"')


def _widget_files():
    return sorted(p for p in WIDGETS.glob("*.py") if p.name != "__init__.py")


class TestThePaletteIsWhereItSays:
    @pytest.mark.parametrize("name,value", sorted(EXPECTED.items()))
    def test_each_colour_is_what_was_consolidated(self, name, value):
        assert getattr(_theme, name) == value, (
            f"_theme.{name} is {getattr(_theme, name)!r}, was {value!r}. If the change "
            f"is deliberate, update this table in the same commit — that is the point "
            f"of it being here."
        )

    def test_the_glyph_map_covers_every_earthwork_type(self):
        """The dict `tool_menu` and `network_view` each used to hold a copy of."""
        from terrainflow_assessment.core.registry.earthwork_types import all_types

        missing = [t for t in all_types() if t not in _theme.GLYPH]
        assert not missing, (
            f"no glyph for {missing}; the tool menu and the network view both read this "
            f"dict, so a type missing from it renders as a bullet in both"
        )

    def test_the_spillway_pair_is_blue_out_green_in(self):
        """One pair, read by the tool menu, the spillway table and the map layer."""
        assert _theme.SPILLWAY_OUT == ("▽", _theme.WATER)
        assert _theme.SPILLWAY_IN == ("▲", _theme.GROWTH)


class TestNoWidgetKeepsItsOwnCopy:
    """The part that stops it growing back."""

    @pytest.mark.parametrize("path", _widget_files(), ids=lambda p: p.name)
    def test_it_does_not_spell_out_a_colour_the_theme_already_holds(self, path):
        known = {v.lower(): k for k, v in EXPECTED.items()}
        offenders = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for found in _HEX.findall(line):
                low = found.lower()
                if low in _ALLOWED_LITERALS:
                    continue
                if low in known:
                    offenders.append(f"{path.name}:{lineno}: {found} is _theme.{known[low]}")
                else:
                    offenders.append(
                        f"{path.name}:{lineno}: {found} is not in the theme at all — "
                        f"add it there rather than here")
        assert not offenders, (
            "chrome colour declared in a widget instead of the theme:\n  "
            + "\n  ".join(offenders)
        )

    @pytest.mark.parametrize("path", _widget_files(), ids=lambda p: p.name)
    def test_it_does_not_carry_its_own_brush(self, path):
        """`def _brush` was identical in three files, each with its own Qt import."""
        source = path.read_text(encoding="utf-8")
        assert "def _brush" not in source, (
            f"{path.name} defines its own _brush; import it from _theme"
        )
