"""
layout_pdf.py — draw a :class:`Report` onto a QGIS print layout and export a PDF.

The only file in the plugin that knows QGIS layouts exist. It walks the pure
section list from :mod:`terrainflow_assessment.modules.report_model` and puts
each one on a page; it makes no decisions about *what* the report says.

Tables are ``QgsLayoutItemTextTable`` rather than ``QgsLayoutItemHtml``:
``QgsLayoutItemHtml`` needs QtWebKit, which this QGIS happens to ship but which
upstream is removing and which ``metadata.txt``'s ``qgisMinimumVersion=3.22``
does not guarantee. The text table is a ``QgsLayoutMultiFrame`` either way, so
``RepeatUntilFinished`` gives the same automatic pagination with no such bet.

Its one trap, measured rather than assumed: **it does not wrap or shrink to fit
its frame — it silently overflows.** There is no width equivalent of
``RepeatUntilFinished``, so column widths are checked here and the offending
table is narrowed rather than allowed off the page.
"""

import logging
import os

_log = logging.getLogger(__name__)

A4_PORTRAIT = (210.0, 297.0)
A4_LANDSCAPE = (297.0, 210.0)
MARGIN_MM = 15.0
FOOTER_H_MM = 10.0
# A page carrying less than this is treated as still empty, so the next
# section continues on it rather than leaving it as an orphan.
_ORPHAN_MM = 55.0

# QgsLayoutItemTextTable reports its *frame* height from totalSize(), not the
# height its content needs, so a table sized to the space available consumes the
# whole page and the report balloons. Row height is exactly linear in font size
# — measured by binary-searching the smallest frame a table still fits in:
#
#     6 pt -> 4.420 mm/row    8 pt -> 5.069 mm/row
#     7 pt -> 4.730 mm/row    9 pt -> 5.379 mm/row
#
# with the header costing one more row plus 0.5 mm of frame padding.
_ROW_MM_AT_7PT = 4.730
_ROW_MM_PER_PT = 0.320
_TABLE_PAD_MM = 0.5


def table_content_height(n_rows, font_pt=7.0):
    """Millimetres a text table needs for ``n_rows`` plus its header."""
    row = _ROW_MM_AT_7PT + (font_pt - 7.0) * _ROW_MM_PER_PT
    return (n_rows + 1) * row + _TABLE_PAD_MM


# Label text wraps inside its frame, but the frame does not grow to match:
# adjustSizeToText() measures the text on ONE line, so a wrapped paragraph got a
# single line's height and the next block was drawn over its tail. Estimate the
# wrapped height instead.
#: What every label asks for. A request, not a promise — see label_char_factor.
_FONT_FAMILY = "Arial"

_LABEL_CHAR_FACTOR = 0.50      # fallback: Arial's average advance / point size
_LABEL_LINE_FACTOR = 1.32      # line height as a fraction of point size
_PT_TO_MM = 25.4 / 72.0

#: Enough of the alphabet, in roughly report proportions, to average over.
_ADVANCE_SAMPLE = ("The measured storage across every feature is 1,240 m3 "
                   "against 1,315 m3 at this grid (-5.7%).")
_advance_factor_cache = {}


def label_char_factor(family=_FONT_FAMILY):
    """Average glyph advance as a fraction of point size, **measured**.

    0.50 was an assumption, and measuring it says 0.641 for Arial over this
    report's own prose: report text is full of digits, capitals and units, all
    of which are wider than the lowercase average the round number came from.
    Too small a factor means the wrap estimate believes more characters fit per
    line, which under-counts lines, which under-sizes the frame, which draws
    the next block over the tail of this one. Under-estimating is the failure
    the estimate exists to prevent, so it is the direction to get right.

    Measured against Qt's own word-wrapped ``boundingRect`` on three report
    paragraphs at 8.5 and 9 pt, both factors still come out short — the constant
    by as much as 8.6 mm on a paragraph, the measured one by 4.4 mm. Half the
    error, in the safe direction, is worth an extra page in a ten-page document.

    ``family`` is a request, not a fact: ``QFont`` substitutes silently when the
    host has no Arial, and measuring off whatever was actually resolved is what
    makes this correct on a machine that does not. The substitution is logged,
    because a report that paginates differently per machine is worth knowing
    about.

    Falls back to the constant when there is no usable Qt — ``tests/`` imports
    this module under a mocked ``qgis`` and would otherwise measure a stub.
    """
    if family in _advance_factor_cache:
        return _advance_factor_cache[family]

    factor = _LABEL_CHAR_FACTOR
    try:
        from qgis.PyQt.QtGui import QFont, QFontInfo, QFontMetricsF

        probe_pt = 20.0                       # big enough that rounding is noise
        font = QFont(family)
        font.setPointSizeF(probe_pt)

        resolved = QFontInfo(font).family()
        if resolved and resolved.lower() != family.lower():
            _log.info("report font %r is not installed; laying out with %r",
                      family, resolved)

        advance = QFontMetricsF(font).horizontalAdvance(_ADVANCE_SAMPLE)
        measured = float(advance) / len(_ADVANCE_SAMPLE) / probe_pt
        # A stub, a metric-less platform or a pathological font: keep the
        # constant rather than laying the document out on nonsense.
        if 0.2 < measured < 1.5:
            factor = measured
    except Exception:                          # pragma: no cover - no Qt here
        pass

    _advance_factor_cache[family] = factor
    return factor


def estimate_label_height(text, width_mm, size_pt, margin_mm=2.0):
    """Millimetres a wrapped label needs, including its own margins."""
    char_mm = size_pt * label_char_factor() * _PT_TO_MM
    line_mm = size_pt * _LABEL_LINE_FACTOR * _PT_TO_MM
    usable = max(width_mm - 2 * margin_mm, char_mm)
    per_line = max(1, int(usable / char_mm))

    lines = 0
    for paragraph in str(text).split("\n"):
        lines += max(1, -(-len(paragraph) // per_line))   # ceil
    return lines * line_mm + margin_mm


def estimate_label_width(text, size_pt, margin_mm=2.0):
    """Millimetres one unwrapped line of ``text`` needs, margins included.

    The same average-advance approximation as the height estimate, which is
    good enough for laying legend chips out in a row: the cost of being a
    millimetre generous is a slightly wide gap, not an overlap.
    """
    return len(str(text)) * size_pt * label_char_factor() * _PT_TO_MM + margin_mm


# Roughly the width of a 7 pt character, plus the cell padding either side.
_CHAR_MM = 1.35
_CELL_PAD_MM = 3.0
_MIN_COL_MM = 12.0


def _column_widths(headers, rows, total_mm):
    """Share ``total_mm`` between columns in proportion to their content.

    Longest cell rather than average: a column whose widest entry wraps is the
    one that makes a table unreadable, and there is no wrapping to fall back on.
    """
    if not headers:
        return []
    longest = []
    for i, head in enumerate(headers):
        widest = len(str(head))
        for row in rows:
            if i < len(row):
                widest = max(widest, len(str(row[i])))
        longest.append(widest)

    wanted = [max(_MIN_COL_MM, n * _CHAR_MM + _CELL_PAD_MM) for n in longest]
    demand = sum(wanted)
    if demand <= 0:
        return [total_mm / len(headers)] * len(headers)
    # Scale to the frame either way: shrink when the content overflows (the
    # table has no wrapping and would run off the page), grow when it is narrow.
    scale = total_mm / demand
    return [w * scale for w in wanted]

_TONE = {
    "info": ("#eef4f6", "#22302e"),
    "warn": ("#fdf3e3", "#7e4e03"),
    "bad": ("#fdeeec", "#8c2519"),
    "good": ("#eaf6ee", "#14612f"),
}


class LayoutError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Small Qt/QGIS helpers
# ---------------------------------------------------------------------------

def _pt(x, y):
    from qgis.core import QgsLayoutPoint, QgsUnitTypes
    return QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters)


def _size(w, h):
    from qgis.core import QgsLayoutSize, QgsUnitTypes
    return QgsLayoutSize(w, h, QgsUnitTypes.LayoutMillimeters)


def _text_format(size_pt, bold=False, colour="#22302e"):
    from qgis.core import QgsTextFormat
    from qgis.PyQt.QtGui import QColor, QFont

    fmt = QgsTextFormat()
    # setPointSizeF, not the int constructor: `QFont("Arial", int(8.5))` asked for
    # 8 pt, and while `fmt.setSize` below is what the renderer honours, the font
    # carried on reporting a size nothing in the document uses — which is what
    # any measurement taken off it would then believe.
    font = QFont(_FONT_FAMILY)
    font.setPointSizeF(float(size_pt))
    font.setBold(bold)
    fmt.setFont(font)
    fmt.setSize(size_pt)
    fmt.setColor(QColor(colour))
    return fmt


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------

def _section_handlers():
    """Section type -> builder method name.

    Mirrors report_html.SECTION_HANDLERS; a test asserts the two cover exactly
    the same section types, so a new one cannot be added to a single renderer.
    """
    from terrainflow_assessment.modules.report_model import (
        Callout,
        DataTable,
        Heading,
        Hero,
        ImageRef,
        KeyValueTable,
        MapRef,
        PageBreak,
        Paragraph,
        StatGrid,
    )
    return {
        Heading: "_render_heading",
        Paragraph: "_render_paragraph",
        Callout: "_render_callout",
        Hero: "_render_hero",
        StatGrid: "_render_stat_grid",
        KeyValueTable: "_render_key_value_table",
        DataTable: "_render_data_table",
        ImageRef: "_render_image",
        MapRef: "_render_map",
        PageBreak: "_render_page_break",
    }


#: Section type -> method name on ReportLayoutBuilder.
SECTION_HANDLERS = _section_handlers()


class ReportLayoutBuilder:
    """Lays sections out top-to-bottom, starting a page whenever one runs out."""

    def __init__(self, project, report, images=None, maps=None, dpi=200):
        self.project = project
        self.report = report
        self.images = images or {}      # key -> png path
        self.maps = maps or {}          # key -> MapSpec
        self.dpi = dpi
        self.layout = None
        self._page = 0
        self._y = 0.0
        self._landscape = False
        self._next_section = None   # one section of lookahead, set while building

    # -- page management ---------------------------------------------------

    def _page_size(self):
        return A4_LANDSCAPE if self._landscape else A4_PORTRAIT

    def _text_width(self):
        return self._page_size()[0] - 2 * MARGIN_MM

    def _page_bottom(self):
        return self._page_size()[1] - MARGIN_MM - FOOTER_H_MM

    def _new_page(self, landscape=False):
        from qgis.core import QgsLayoutItemPage, QgsLayoutSize, QgsUnitTypes

        self._landscape = landscape
        page = QgsLayoutItemPage(self.layout)
        w, h = self._page_size()
        page.setPageSize(QgsLayoutSize(w, h, QgsUnitTypes.LayoutMillimeters))
        self.layout.pageCollection().addPage(page)
        self._page = self.layout.pageCollection().pageCount() - 1
        self._y = MARGIN_MM
        return page

    def _page_break(self, landscape):
        """Start the next section's page — unless this one is barely used.

        A block that spills off the bottom lands alone at the top of a fresh
        page, and an unconditional break then abandons the rest of it. Carrying
        on instead turns two near-empty pages into one full one.
        """
        if (self._landscape == landscape
                and self._y <= MARGIN_MM + _ORPHAN_MM):
            self._gap(6.0)
            return
        self._new_page(landscape)

    def _room_for(self, height):
        """Start a page when the next block would run past the footer."""
        if self._y + height > self._page_bottom():
            self._new_page(self._landscape)
            return True
        return False

    def _y_on_page(self):
        """The cursor, page-relative.

        ``attemptMove`` interprets its point relative to the given page whenever
        ``page`` is passed, so this must stay page-relative rather than being an
        offset down the whole document.
        """
        return self._y

    # -- primitives --------------------------------------------------------

    def _label(self, text, size_pt=9, bold=False, colour="#22302e",
               width=None, height=None, background=None, indent=0.0,
               gap_after=1.5):
        from qgis.core import QgsLayoutItemLabel
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtGui import QColor

        width = width if width is not None else self._text_width() - indent
        item = QgsLayoutItemLabel(self.layout)
        item.setText(text)
        item.setTextFormat(_text_format(size_pt, bold, colour))
        item.setVAlign(Qt.AlignTop)
        item.setMarginX(1.5)
        item.setMarginY(1.0)
        if background:
            item.setBackgroundEnabled(True)
            item.setBackgroundColor(QColor(background))
        self.layout.addLayoutItem(item)

        if height is None:
            height = estimate_label_height(text, width, size_pt)
        self._room_for(height)
        item.attemptResize(_size(width, height))
        item.attemptMove(_pt(MARGIN_MM + indent, self._y_on_page()),
                         page=self._page)
        self._y += height + gap_after
        return item

    def _gap(self, mm=3.0):
        self._y += mm

    # -- section renderers -------------------------------------------------

    def render(self):
        from qgis.core import QgsPrintLayout

        self.layout = QgsPrintLayout(self.project)
        self.layout.initializeDefaults()
        # initializeDefaults leaves one page; size it and use it as page 1.
        self.layout.pageCollection().clear()
        self._new_page(landscape=False)

        self._cover()
        # One section of lookahead, so a heading can decide whether to turn the
        # page *with* the block it introduces instead of stranding itself at the
        # foot of this one.
        sections = list(self.report.sections)
        for i, section in enumerate(sections):
            self._next_section = sections[i + 1] if i + 1 < len(sections) else None
            self._section(section)
        self._next_section = None
        self._footers()
        return self.layout

    def _cover(self):
        self._label(self.report.title, size_pt=20, bold=True)
        self._label(self.report.subtitle, size_pt=10, colour="#5f7176")
        done = [k for k, v in self.report.completeness.items() if v]
        missing = [k for k, v in self.report.completeness.items() if not v]
        line = "Stages run: " + (", ".join(done) if done else "none")
        if missing:
            line += "   ·   Not run: " + ", ".join(missing)
        self._label(line, size_pt=8, colour="#5f7176")
        self._gap(4.0)

    def _section(self, section):
        """Dispatch through SECTION_HANDLERS so parity with the HTML renderer
        can be asserted rather than assumed."""
        name = SECTION_HANDLERS.get(type(section))
        if name is None:
            _log.warning("no PDF handler for %s", type(section).__name__)
            return
        getattr(self, name)(section)

    # -- one handler per section type --------------------------------------

    def _render_page_break(self, section):
        self._page_break(section.orientation == "landscape")

    #: What a heading reserves for the block after it when that block's height
    #: cannot be estimated — a table, a map, a chart. Enough to be worth turning
    #: the page for without dragging every heading onto a page of its own.
    _HEADING_KEEP_MM = 20.0

    def _following_height(self):
        """Millimetres the section after this one will want, where that is knowable."""
        section = getattr(self, "_next_section", None)
        text = getattr(section, "text", None)
        if not text:
            return self._HEADING_KEEP_MM
        size_pt = 8.5 if type(section).__name__ == "Callout" else 9.0
        return estimate_label_height(text, self._text_width(), size_pt)

    def _render_heading(self, section):
        self._gap(2.0)
        size_pt = 15 if section.level == 1 else 11
        # Turn the page now, with the block this heading introduces, rather than
        # leaving the heading stranded at the foot and its first paragraph
        # overleaf. `_label` only ever asked for room for itself, so a heading
        # always fitted and what it announced frequently did not.
        self._room_for(estimate_label_height(section.text, self._text_width(),
                                             size_pt) + self._following_height())
        self._label(section.text, size_pt=size_pt, bold=True)

    def _render_paragraph(self, section):
        self._label(section.text, size_pt=9)

    def _render_key_value_table(self, section):
        self._table(section.title, ["", ""],
                    [[str(k), str(v)] for k, v in section.rows])

    def _render_data_table(self, section):
        self._table(section.title, section.headers, section.rows,
                    note=section.note)

    def _render_hero(self, hero):
        colour = _TONE.get(hero.tone, _TONE["info"])[1]
        self._label(hero.value, size_pt=34, bold=True, colour=colour, height=16)
        self._label(hero.label, size_pt=11)
        if hero.sub:
            self._label(hero.sub, size_pt=8.5, colour="#5f7176")
        self._gap(2.0)

    def _render_callout(self, callout):
        bg, fg = _TONE.get(callout.tone, _TONE["info"])
        # The title is set apart, as it is in the HTML — there it is a bold block
        # above the prose, and here it was one more line of the same run-on text,
        # so "These measurements are out of date" read as the first sentence of
        # the paragraph it was heading. Two labels sharing one background colour
        # with no gap between them is one box on the page.
        if callout.title:
            self._label(callout.title, size_pt=8.5, colour=fg, background=bg,
                        bold=True, gap_after=0.0)
        self._label(callout.text, size_pt=8.5, colour=fg, background=bg,
                    bold=callout.tone == "bad")
        self._gap(1.5)

    #: Cards per row. Four 8.5 pt cards fit the text column with their labels
    #: still on one line; a fifth squeezes every one of them.
    _CARDS_PER_ROW = 4

    def _render_stat_grid(self, grid):
        """Cards across the text column — a table row would read as data.

        Wrapped, not truncated. This drew ``cards[:4]`` and dropped the rest
        without saying so, while the HTML renderer flex-wraps and shows them all:
        a five-card grid printed four figures on paper and five on screen, with
        nothing on the page to suggest one was missing.
        """
        if not grid.cards:
            return
        per_row = self._CARDS_PER_ROW
        gap = 3.0
        for start in range(0, len(grid.cards), per_row):
            chunk = grid.cards[start:start + per_row]
            # A short last row keeps the full row's card width rather than
            # stretching two cards across the page.
            width = (self._text_width() - gap * (per_row - 1)) / per_row
            self._room_for(20.0)
            top = self._y_on_page()
            for i, card in enumerate(chunk):
                label, value, sub = (list(card) + ["", "", ""])[:3]
                self._card(MARGIN_MM + i * (width + gap), top,
                           width, label, value, sub)
            self._y += 21.0

    def _card(self, x, y, width, label, value, sub):
        from qgis.core import QgsLayoutItemLabel
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtGui import QColor

        item = QgsLayoutItemLabel(self.layout)
        item.setText(f"{label}\n{value}\n{sub}")
        item.setTextFormat(_text_format(8.5))
        item.setVAlign(Qt.AlignTop)
        item.setMarginX(2.0)
        item.setMarginY(1.5)
        item.setBackgroundEnabled(True)
        item.setBackgroundColor(QColor("#f4f7f8"))
        self.layout.addLayoutItem(item)
        item.attemptResize(_size(width, 19.0))
        item.attemptMove(_pt(x, y), page=self._page)

    def _table(self, title, headers, rows, note=""):
        from qgis.core import (
            QgsLayoutFrame,
            QgsLayoutItemTextTable,
            QgsLayoutMultiFrame,
            QgsLayoutTable,
            QgsLayoutTableColumn,
        )

        # No rows, no section — including its title. Printing the heading anyway
        # left an orphan on the page over nothing, while the HTML renderer
        # dropped the table whole; the two now say the same thing, which is
        # nothing. A table that is empty for a *reason* is a Callout in the
        # model, not a headless table here.
        if not rows:
            return
        if title:
            self._label(title, size_pt=10, bold=True)

        table = QgsLayoutItemTextTable(self.layout)
        self.layout.addMultiFrame(table)

        # Columns auto-size to their content and leave the rest of the frame
        # empty, so a three-column table sat in the left third of the page.
        # Share the width out by how much text each column actually carries.
        widths = _column_widths(headers, rows, self._text_width())
        columns = []
        for head, width in zip(headers, widths):
            col = QgsLayoutTableColumn()
            col.setHeading(str(head))
            col.setWidth(width)
            columns.append(col)
        table.setColumns(columns)
        for row in rows:
            table.addRow([str(c) for c in row])

        table.setContentTextFormat(_text_format(7))
        table.setHeaderTextFormat(_text_format(7, bold=True))
        table.setShowEmptyRows(False)
        # Key/value tables have no headings; drawing the band anyway leaves an
        # empty grey stripe above the first row.
        if not any(str(h).strip() for h in headers):
            table.setHeaderMode(QgsLayoutTable.NoHeaders)

        needed = table_content_height(len(rows), 7.0)
        available = self._page_bottom() - self._y
        # Move a table that would only get a sliver rather than splitting it
        # two rows from the bottom of a page.
        if available < min(needed, 30.0):
            self._new_page(self._landscape)
            available = self._page_bottom() - self._y

        height = min(needed, available)
        frame = QgsLayoutFrame(self.layout, table)
        frame.attemptResize(_size(self._text_width(), height))
        frame.attemptMove(_pt(MARGIN_MM, self._y_on_page()), page=self._page)
        table.addFrame(frame)
        # Anything that did not fit spills onto its own frames and pages rather
        # than being clipped — there is no scrollbar in a PDF.
        table.setResizeMode(QgsLayoutMultiFrame.RepeatUntilFinished)

        self._advance_past(table, height)
        if note:
            self._label(note, size_pt=7.5, colour="#5f7176")
        self._gap(2.0)

    def _advance_past(self, table, height):
        """Move the cursor below the table, following any pages it added.

        A multiframe that spills creates its own frames and pages; the cursor has
        to end up under the *last* one or the next section draws over it.
        """
        frames = table.frames()
        if len(frames) > 1:
            last = frames[-1]
            # `last.page()`, not `pageNumberForPoint(last.pagePos())`. The two speak
            # different coordinate spaces: `pagePos()` is relative to the top-left of
            # the frame's own page, while `pageNumberForPoint` expects an absolute
            # layout coordinate. Handing it a page-relative point put every spilled
            # table's continuation back on page 0, so the next section — and the
            # table's own note — were drawn on top of the cover.
            self._page = last.page()
            # And measure from where the frame actually sits, rather than assuming it
            # begins at the top margin. A continuation frame usually does; the first
            # frame of a table that fitted below other content does not.
            self._y = last.pagePos().y() + last.rect().height() + 2.0
        else:
            self._y += height + 2.0

    def _render_image(self, ref):
        from qgis.core import QgsLayoutItemPicture

        path = self.images.get(ref.key)
        if not path or not os.path.exists(path):
            if ref.fallback is not None:
                self._section(ref.fallback)
            return

        from qgis.PyQt.QtGui import QImage
        image = QImage(path)
        if image.isNull() or not image.width():
            return
        # Charts are drawn at their intended printed size, so place them at it.
        # Stretching to the text column blew a one-node diagram up fivefold and
        # made its labels bigger than the boxes they sat in.
        natural_w = image.width() / float(self.dpi) * 25.4
        natural_h = image.height() / float(self.dpi) * 25.4
        max_w = self._text_width()
        max_h = self._page_bottom() - MARGIN_MM
        scale = min(1.0, max_w / natural_w, max_h / natural_h)
        width, height = natural_w * scale, natural_h * scale

        self._room_for(height + 6.0)
        item = QgsLayoutItemPicture(self.layout)
        item.setPicturePath(path)
        item.setResizeMode(QgsLayoutItemPicture.ZoomResizeFrame)
        self.layout.addLayoutItem(item)
        item.attemptResize(_size(width, height))
        item.attemptMove(_pt(MARGIN_MM, self._y_on_page()), page=self._page)
        self._y += height + 1.5
        if ref.caption:
            self._label(ref.caption, size_pt=7.5, colour="#5f7176")
        self._gap(2.0)

    def _render_map(self, ref):
        spec = self.maps.get(ref.key)
        if ref.reason or spec is None:
            reason = ref.reason or "The layers this map needs are not available."
            self._label(reason, size_pt=8.5, colour="#7e4e03",
                        background="#fdf3e3")
            self._gap(2.0)
            return

        from qgis.core import QgsLayoutItemMap
        from qgis.PyQt.QtGui import QColor

        from terrainflow_assessment.qgis.adapters.map_image import fit_extent

        height = min(spec.height_mm, self._page_bottom() - MARGIN_MM - 12.0)
        width = self._text_width()
        self._room_for(height + 8.0)

        item = QgsLayoutItemMap(self.layout)
        self.layout.addLayoutItem(item)
        item.attemptResize(_size(width, height))
        item.attemptMove(_pt(MARGIN_MM, self._y_on_page()), page=self._page)
        if spec.crs is not None:
            item.setCrs(spec.crs)
        item.setLayers(spec.layers)
        if spec.extent is not None:
            item.setExtent(fit_extent(spec.extent, width, height))
        item.setBackgroundColor(QColor(255, 255, 255))
        item.setFrameEnabled(True)
        self._north_arrow(width)
        self._y += height + 1.0

        self._scale_bar(item, spec)
        self._map_key(getattr(ref, "legend", None))
        if ref.caption:
            self._label(ref.caption, size_pt=7.5, colour="#5f7176")
        self._gap(2.0)

    # Legend geometry, in millimetres.
    _KEY_SWATCH_W = 7.0
    _KEY_ROW_H = 4.6
    _KEY_GAP = 3.0
    _KEY_PT = 6.5

    def _north_arrow(self, map_width):
        """North, in the map's top-right corner.

        Drawn rather than pulled from an SVG: QGIS's bundled north arrows sit
        under a search path that differs between installs, and a missing file
        renders as an empty box — worse than no arrow at all. A triangle and
        the letter N are unambiguous and cost two layout items.

        Every map here is north-up; nothing in the render path rotates one, so
        this is a fixed decoration rather than a bearing.
        """
        from qgis.core import QgsLayoutItemLabel, QgsLayoutItemShape
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtGui import QColor

        try:
            head, box = 7.0, 9.0
            x = MARGIN_MM + map_width - box - 1.5
            y = self._y_on_page() + 1.5

            arrow = QgsLayoutItemShape(self.layout)
            arrow.setShapeType(QgsLayoutItemShape.Triangle)
            self.layout.addLayoutItem(arrow)
            arrow.attemptResize(_size(head * 0.6, head))
            arrow.attemptMove(_pt(x + box * 0.3, y), page=self._page)
            symbol = arrow.symbol()
            if symbol is not None:
                # White stroke so the arrow survives being dropped on dark
                # bush or deep water; the maps have no reserved corner.
                symbol.setColor(QColor("#22302e"))
                try:
                    layer = symbol.symbolLayer(0)
                    layer.setStrokeColor(QColor(255, 255, 255))
                    layer.setStrokeWidth(0.4)
                except Exception:
                    pass

            label = QgsLayoutItemLabel(self.layout)
            label.setText("N")
            label.setTextFormat(_text_format(7.0, bold=True))
            label.setHAlign(Qt.AlignHCenter)
            label.setVAlign(Qt.AlignTop)
            label.setBackgroundEnabled(True)
            label.setBackgroundColor(QColor(255, 255, 255, 190))
            self.layout.addLayoutItem(label)
            label.attemptResize(_size(box, 4.6))
            label.attemptMove(_pt(x, y + head + 0.2), page=self._page)
        except Exception as exc:
            _log.debug("north arrow skipped: %s", exc)

    def _map_key(self, entries):
        """The map's legend, as swatch-and-label pairs wrapped across the page.

        The same entries the HTML renderer draws as chips: both read
        ``MapRef.legend``, which the pure model builds, so neither renderer
        decides for itself what a colour on the map means.
        """
        if not entries:
            return
        from qgis.core import QgsLayoutItemLabel
        from qgis.PyQt.QtCore import Qt

        avail = self._text_width()
        self._room_for(self._KEY_ROW_H * 2)
        x, top = MARGIN_MM, self._y_on_page()
        for entry in entries:
            text_w = min(estimate_label_width(entry.label, self._KEY_PT),
                         avail - self._KEY_SWATCH_W - self._KEY_GAP)
            cell = self._KEY_SWATCH_W + 1.5 + text_w + self._KEY_GAP
            if x > MARGIN_MM and x + cell > MARGIN_MM + avail:
                x, top = MARGIN_MM, top + self._KEY_ROW_H
            self._key_swatch(entry, x, top)

            label = QgsLayoutItemLabel(self.layout)
            label.setText(entry.label)
            label.setTextFormat(_text_format(self._KEY_PT, colour="#5f7176"))
            label.setVAlign(Qt.AlignVCenter)
            self.layout.addLayoutItem(label)
            label.attemptResize(_size(text_w, self._KEY_ROW_H))
            label.attemptMove(_pt(x + self._KEY_SWATCH_W + 1.5, top),
                              page=self._page)
            x += cell
        self._y = top + self._KEY_ROW_H + 1.5

    def _key_swatch(self, entry, x, top):
        """One legend swatch. A ramp draws as its stops butted together.

        The gradient is what the reader matches against the map, so its middle
        stops matter — collapsing it to its two ends leaves them hunting for a
        colour the key never showed.
        """
        from qgis.core import QgsLayoutItemShape
        from qgis.PyQt.QtGui import QColor

        colours = [entry.colour]
        if entry.kind == "ramp":
            colours = [c for c in (getattr(entry, "colours", ()) or ())
                       if c] or colours
        band_h = 1.4 if entry.kind == "line" else 3.0
        # A point draws as a disc, not as a swatch-wide oval: the map's marker
        # is round, and the key has to look like the thing it stands for.
        width = band_h if entry.kind == "point" else self._KEY_SWATCH_W
        y = top + (self._KEY_ROW_H - band_h) / 2.0
        step = width / float(len(colours))
        for i, hex_colour in enumerate(colours):
            if not hex_colour:
                continue
            patch = QgsLayoutItemShape(self.layout)
            patch.setShapeType(QgsLayoutItemShape.Ellipse
                               if entry.kind == "point"
                               else QgsLayoutItemShape.Rectangle)
            self.layout.addLayoutItem(patch)
            patch.attemptResize(_size(step, band_h))
            patch.attemptMove(
                _pt(x + (self._KEY_SWATCH_W - width) / 2.0 + i * step, y),
                page=self._page)
            symbol = patch.symbol()
            if symbol is None:
                continue
            symbol.setColor(QColor(hex_colour))
            try:
                symbol.symbolLayer(0).setStrokeStyle(0)   # Qt.NoPen
            except Exception:
                pass

    def _scale_bar(self, map_item, spec):
        """A site plan without a scale is not a document anyone can build from.

        ``applyDefaultSettings`` alone produced a "0 m" bar on a real extent, so
        the segment size is derived from the map width instead.

        Unless the map is in a geographic CRS, where its extent is in degrees and
        every length here would be fiction. A missing scale bar is a visible
        absence; a plausible wrong one is not, and it is the more dangerous of
        the two on a drawing somebody digs from.
        """
        from qgis.core import QgsLayoutItemScaleBar, QgsUnitTypes

        crs = getattr(spec, "crs", None)
        if crs is not None and crs.isGeographic():
            _log.debug("scale bar omitted: map CRS %s is geographic",
                       crs.authid())
            return

        try:
            bar = QgsLayoutItemScaleBar(self.layout)
            # Added to the layout *before* being configured: an item that is not
            # in the layout yet cannot resolve its linked map's scale, so
            # update() sized the bar against a default and drew it a fifth of the
            # width it should have been.
            self.layout.addLayoutItem(bar)
            bar.setStyle("Single Box")
            bar.setLinkedMap(map_item)
            bar.applyDefaultSettings()
            bar.setUnits(QgsUnitTypes.DistanceMeters)
            bar.setUnitLabel("m")
            # Size the segments from what the map ACTUALLY shows, not from the
            # extent it was asked for: setExtent widens to the item's aspect, so
            # the two differ and a bar scaled to the request comes out wrong.
            # applyDefaultSettings on its own leaves 0 m per segment, which
            # renders as a 5 mm stub labelled "0 m".
            shown = map_item.extent()
            if shown is not None and shown.width() > 0:
                target = shown.width() / 5.0
                step = 10 ** max(0, len(str(int(max(target, 1)))) - 1)
                bar.setUnitsPerSegment(max(step, 1))
                bar.setNumberOfSegments(4)
                bar.setNumberOfSegmentsLeft(0)
            bar.setTextFormat(_text_format(6.5))
            bar.update()
            bar.attemptMove(_pt(MARGIN_MM, self._y_on_page()), page=self._page)
            # The bar draws its labels above the boxes, so it needs more room
            # than its box height or the caption lands on top of it.
            self._y += 12.0
        except Exception as exc:
            _log.debug("scale bar skipped: %s", exc)

    def _footers(self):
        """Provenance on every page — which site, which run, when, and the
        standing caveat, so a page photographed on its own still carries it."""
        from qgis.core import QgsLayoutItemLabel
        from qgis.PyQt.QtCore import Qt

        collection = self.layout.pageCollection()
        count = collection.pageCount()
        for page in range(count):
            item = QgsLayoutItemLabel(self.layout)
            item.setText(f"{self.report.footer}    ·    Page {page + 1} of {count}")
            item.setTextFormat(_text_format(6.5, colour="#5f7176"))
            item.setVAlign(Qt.AlignVCenter)
            self.layout.addLayoutItem(item)
            size = collection.page(page).pageSize()
            item.attemptResize(_size(size.width() - 2 * MARGIN_MM, 6.0))
            item.attemptMove(_pt(MARGIN_MM, size.height() - MARGIN_MM + 2.0),
                             page=page)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MapSpec:
    """What a report map needs. ``layers`` is top-first."""

    def __init__(self, layers, extent=None, crs=None, height_mm=95.0):
        self.layers = layers
        self.extent = extent
        self.crs = crs
        self.height_mm = height_mm


def build_layout(project, report, images=None, maps=None, dpi=200):
    return ReportLayoutBuilder(project, report, images, maps, dpi).render()


def export_pdf(path, project, report, images=None, maps=None, dpi=200):
    """Render ``report`` and write a PDF. Returns ``path``.

    The layout is built standalone and discarded — never added to the project's
    layout manager, which would both pollute the user's Layouts panel and save
    references to temp files that vanish when the plugin unloads.
    """
    from qgis.core import Qgis, QgsLayoutExporter

    layout = build_layout(project, report, images, maps, dpi)
    exporter = QgsLayoutExporter(layout)
    settings = QgsLayoutExporter.PdfExportSettings()
    settings.dpi = dpi
    settings.forceVectorOutput = True
    settings.rasterizeWholeImage = False
    settings.exportMetadata = True
    settings.appendGeoreference = False
    try:
        settings.textRenderFormat = Qgis.TextRenderFormat.AlwaysText
    except AttributeError:  # pragma: no cover - older QGIS
        pass

    result = exporter.exportToPdf(path, settings)
    if result != QgsLayoutExporter.Success:
        raise LayoutError(exporter.errorMessage() or f"export failed ({result})")
    return path


def render_page_image(layout, page=0, dpi=96):
    """A layout page as a ``QImage`` — how the PDF gets into the shot diff."""
    from qgis.core import QgsLayoutExporter
    from qgis.PyQt.QtCore import QSize

    return QgsLayoutExporter(layout).renderPageToImage(page, QSize(), dpi)
