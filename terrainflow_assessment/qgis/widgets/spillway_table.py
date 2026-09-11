"""
spillway_table.py — per-feature overflow review, and where the spillways get sited.

The properties dialog answers "is *this* spillway right" one feature at a time, modally,
and freezes its figures the moment it opens. That is the wrong shape for the question a
design actually raises, which is comparative and arrives late: having laid the earthworks
out, which of them can pass what now reaches them?

It arrives late because the numbers move on their own. A feature's design flow is a
function of the whole network above it, so drawing a swale upslope changes what reaches
everything below — twice over, since it both intercepts part of their catchment and, once
full, passes its own peak on. A sill sized last week can be undersized by a line drawn
today with nothing about it appearing to change.

Two decisions worth stating, because both look like omissions:

**Every water-holding feature gets a row, designed or not.** What a spillway needs is a
function of the terrain and the storm, not of whether the user has ticked a box. A
feature with no spillway yet still shows the width it would need — that is what makes
auto-sizing something you can see rather than something you are told about.

**The inlet gets a marker and no numbers.** An outflow is a weir sized to pass a peak; an
inlet is a protected entry that stops the incoming jet cutting the bank. They are
different structures with different jobs, and this module has no basis for sizing the
second. A width in that column would be confidently wrong.

Data contract — ``set_rows(rows, context)``:
  rows : list of dicts from ``EarthworksController._spillway_row``, each
    {index, id, name, ew_type, enabled, designed, sited, inlet_sited, width_auto,
     target_head_m, actual_head_m, freeboard_min_m, standard_freeboard_m,
     peak_flow_m3s, upstream_m3s, required_width_m, sizing_head_m, built_width_m,
     freeboard_m,
     crest_elevation, height_above_floor_m, rim_elevation, lip_elevation,
     containment_source, sill_depth_m, seed_depth_m,
     sill_storage_m3, containment_storage_m3, given_up_m3,
     given_up_pct, surcharge_level_m, surcharge_storage_m3, spillway_insufficient,
     burned_sill_m, actual_spill_level_m, event_level_m, passes_this_event,
     linked_drains, problems, notes, state}

  ``linked_drains`` names the diversion drains graded down from this crest. It travels
  as data and renders through ``notes``, not as a column: it is one more thing that
  moves when the crest moves, which is a sentence, and the row has no width for a
  feature list.

  ``sill_depth_m`` is the crest's drop below containment as *this build* measured it —
  ``rim_elevation - crest_elevation``, and deliberately not the ``drop_below_rim_m`` the
  model stores. The stored partner is only re-based when a design is restored, while this
  datum is recomputed on every build, so after an earthworks re-analysis the two can be
  measured against different rims. The Sill depth column is editable, and a cell that
  rendered the stored figure would move the crest when the user typed back the number
  already on screen. ``seed_depth_m`` is what a feature with **no** spillway would open
  at (``default_sill_depth_m``, type policy — so it knows nothing about a per-feature
  freeboard override, which is correct because a feature with no spillway carries none).
  It is meaningful on an undesigned row only; a designed row's editor seeds from
  ``sill_depth_m``.

  ``sizing_head_m`` is the head ``required_width_m`` was actually solved at: the type's
  design head, or the sill depth where that is shallower. It exists because a capped width
  and an uncapped ``actual_head_m`` do not reconcile through the weir equation and are not
  meant to — one asks how wide a weir must be to pass this flow through *this* notch, the
  other how deep the design intends to run — and a table showing both without saying so
  looks like an arithmetic error. Equal to ``target_head_m`` wherever the sill is deep
  enough, which is the common case and draws no note.

  ``rim_elevation`` is the **containment** level — the level this feature's water is
  actually held to, which on a bermed swale is the berm crest as built. ``lip_elevation``
  is the bare ring minimum beside it. ``notes`` is a parallel channel to ``problems`` and
  the difference is load-bearing: a row goes to ``"fail"`` on any problem at all, so
  anything that is true-and-fine has to travel separately or every bermed swale reads as
  broken.
  context : {intensity_mm_hr, intensity_is_default, has_idf, harvesting_coefficient}
"""

from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QAbstractSpinBox,
    QDoubleSpinBox,
    QHeaderView,
    QLabel,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from terrainflow_assessment.qgis import help_text as H

_INK = "#22302e"
_MUTED = "#5f7176"
_FAINT = "#8fa0a4"
_HAIRLINE = "#dde4e5"
_HAIRLINE_STRONG = "#c6d1d3"
_GROUND = "#eef1f2"
_SURFACE = "#ffffff"
_GOOD = "#1e8449"
_WARN = "#b9770e"
_BAD = "#c0392b"

# Glyph and colour per spillway kind — the same pair the tool menu and the map layer
# use, so the three surfaces read as one thing rather than three.
# Elevations are reported to two decimals, so a disagreement finer than a centimetre
# is one the user cannot see, cannot act on, and would be told about anyway.
_LEVEL_TOLERANCE_M = 0.01

_OUT_GLYPH, _OUT_COLOUR = "▽", "#1273b5"
_IN_GLYPH, _IN_COLOUR = "▲", "#2e7d55"

# Head this far over target before the cell stops reading as "as designed". A centimetre
# is below setting-out resolution on a DEM, so tighter than this is noise.
_HEAD_TOLERANCE_M = 0.01

# "Outflow" and "Inlet" as two columns, where there was one "Sited" cell holding
# both glyphs. The cell was the placement affordance and dispatched `"outflow"`
# for any click in it, whichever marker was under the cursor — so clicking the ▲
# armed the outflow tool, said "Click where X should OVERFLOW", and sited an
# outflow, which is the only kind that gets a notch cut. The signal has always
# been typed `(int, str)  # 'outflow' | 'inflow'` and no `"inflow"` was ever
# emitted from this widget.
_HEADERS = ("Feature", "Peak flow", "Head", "Width", "Freeboard", "Sill depth", "Sill",
            "Storage", "Outflow", "Inlet")

# Named rather than written out at each use. Three behaviours now key off a column
# position -- the placement affordance, the editable cell and the double-click guard --
# and an insertion that shifted any of them would be wrong silently rather than loudly.
# "Sill depth" and not "Depth": `Earthwork.depth` is the feature's own excavation, which
# the earthworks list and the properties dialog both call Depth, and a swale row offering
# to take that number would put the sill a metre down.
_COL_DEPTH = _HEADERS.index("Sill depth")
_COL_WIDTH = _HEADERS.index("Width")
# By name, which is the reason they are named at all: adding a column shifts
# positions, and every one of these found its own again. `_COL_SITED` was
# `len(_HEADERS) - 1` — "the last column" — which is now the inlet.
_COL_SITED = _HEADERS.index("Outflow")
_COL_INLET = _HEADERS.index("Inlet")

# The sill depth editor is the dialog's Spillway depth control, to the digit -- see
# EarthworkPropertiesDialog.spin_spillway_drop. Two controls for one quantity that
# disagreed about what a legal depth is would be two policies.
_DEPTH_MAX_M = 50.0
_DEPTH_STEP_M = 0.05
_DEPTH_DECIMALS = 2

_QSS = f"""
QTableWidget {{
    background: {_SURFACE}; border: 1px solid {_HAIRLINE};
    border-radius: 6px; gridline-color: {_HAIRLINE};
    selection-background-color: #e9f3ee; selection-color: {_INK};
}}
QHeaderView::section {{
    background: {_GROUND}; border: none;
    border-bottom: 1px solid {_HAIRLINE_STRONG};
    padding: 4px 5px; font-size: 10px; font-weight: 700; color: {_MUTED};
}}
"""


class _DepthDelegate(QStyledItemDelegate):
    """A spin box in the Sill depth cell, configured as the properties dialog's is.

    A delegate rather than an editable item, for one reason above the others: a spin box
    has no empty state, and a text cell would hand the user "clear it" as a gesture that
    looks like deleting the spillway. Deleting one is unrecoverable -- there is no undo
    stack anywhere in this plugin -- and is confirmed in exactly one place, where the
    confirmation names the location, the crest, the width and any linked drains. It also
    means there is no parser to get wrong: no locale decimal comma, no "45cm", no way to
    commit a 60 m sill by leaning on a key.

    It writes nothing into the model. The item text is a render of controller state, so
    the authoritative redraw is the rebuild that follows the commit; if the controller
    refuses the edit, the cell honestly goes on showing what is actually stored.
    """

    def __init__(self, table):
        super().__init__(table)
        self._table = table          # the SpillwayTable, not the QTableWidget

    def createEditor(self, parent, option, index):
        editor = QDoubleSpinBox(parent)
        current = self._table.depth_for_editor(index.row())
        # The floor is min(0, current), not 0. A crest sitting *above* its containment
        # has a negative drop -- a real fault that spillway_validity reports -- and a
        # spin box floored at zero would silently correct it on a stray Enter. 0-50 is
        # the range for typing a new depth; round-tripping an existing bad one unchanged
        # is a separate requirement.
        editor.setRange(min(0.0, current), _DEPTH_MAX_M)
        editor.setDecimals(_DEPTH_DECIMALS)
        editor.setSingleStep(_DEPTH_STEP_M)
        editor.setSuffix(" m")
        # No arrows: they are unusable in a 22px row and cost the width the number needs.
        # Up/Down still step by _DEPTH_STEP_M while the editor is open.
        editor.setButtonSymbols(QAbstractSpinBox.NoButtons)
        # A framed spin box is taller than a text row, so Qt would grow the row to fit it
        # -- and the table's height is fixed to eight rows, so growing one scrolls the
        # rest under the user mid-edit.
        editor.setFrame(False)
        editor.setMinimumHeight(0)
        editor.setKeyboardTracking(False)
        editor.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        editor.setValue(current)
        # Stashed so eventFilter can tell a deliberate edit from a click-away.
        editor.setProperty("_tf_opened_at", float(current))
        return editor

    def setEditorData(self, editor, index):
        editor.setValue(self._table.depth_for_editor(index.row()))

    def setModelData(self, editor, model, index):
        editor.interpretText()      # a typed value may not be interpreted yet
        self._table.commit_depth(index.row(), float(editor.value()))

    def eventFilter(self, editor, event):
        """Clicking away without typing must commit nothing.

        Qt commits on focus-out by default. On a row that has no spillway that would
        create one the user never asked for, silently, from a click that landed
        elsewhere -- the worst failure this column can have, because the gesture that
        causes it is the gesture for changing your mind. A value the user actually
        changed still commits on the way out: losing a deliberate keystroke without
        saying so is the opposite error, not a safer one.
        """
        try:
            is_focus_out = event.type() == event.Type.FocusOut
        except AttributeError:                       # older PyQt enum placement
            is_focus_out = event.type() == event.FocusOut
        if is_focus_out:
            opened_at = editor.property("_tf_opened_at")
            if opened_at is not None and abs(float(editor.value()) - float(opened_at)) < 1e-9:
                self.closeEditor.emit(editor, QAbstractItemDelegate.RevertModelCache)
                return True
        return super().eventFilter(editor, event)


class SpillwayTable(QWidget):
    """Per-feature overflow review; clicking the markers sites the structures."""

    feature_selected = pyqtSignal(int)          # row index into the earthwork manager
    place_requested = pyqtSignal(int, str)      # index, 'outflow' | 'inflow'
    edit_requested = pyqtSignal(int)
    depth_edited = pyqtSignal(int, float)       # index, sill depth below containment (m)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        self._rows = []
        self._filling = False       # a rebuild is writing cells; ignore anything they emit
        self._displayed_ids = []    # feature id per table row, for keeping the user's place
        self._pending = None        # rows that arrived while an editor was open

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        # The two placement columns say what a click on them does, because it is
        # not the same thing: one arms the outflow tool and one the inlet tool,
        # and they were a single cell that always armed the first.
        for column, tip in ((_COL_SITED, H.SPILLWAY_LOCATION),
                            (_COL_INLET, H.SPILLWAY_INLET)):
            head = self.table.horizontalHeaderItem(column)
            if head is not None:
                head.setToolTip(tip)
        self.table.verticalHeader().setVisible(False)
        # Widened from NoEditTriggers, which was the only thing making this table
        # read-only. Triggers are view-wide, so they are *not* what keeps the other
        # columns inert -- QTableWidgetItem's default flags already include
        # ItemIsEditable, and _fill_row clears it on every cell it does not want typed
        # in. That is the whole of "nothing else in this table is editable", and it is
        # per-cell, which is the layer the rule actually lives at.
        #
        # SelectedClicked is what makes a pass down the column quick: click the row,
        # click the cell, type. It is only safe alongside the delegate's rule that a
        # click-away without typing commits nothing -- clicking rows to highlight
        # features on the map is an ordinary gesture, and an editor opening under it
        # would otherwise be a trap. The two ship together or neither does.
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet(_QSS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        # Held on self: setItemDelegateForColumn does not take ownership, and a delegate
        # that is only referenced by the call is collected out from under the view.
        self._depth_delegate = _DepthDelegate(self)
        self.table.setItemDelegateForColumn(_COL_DEPTH, self._depth_delegate)
        self._depth_delegate.closeEditor.connect(self._on_editor_closed)
        lay.addWidget(self.table)

        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet(f"color: {_MUTED}; font-size: 10.5px;")
        lay.addWidget(self.footer)

        self.setVisible(False)

    # ------------------------------------------------------------------ API

    def set_rows(self, rows, context=None):
        """Populate from the controller's row dicts (empty list hides the widget).

        Two things happen around the rebuild that did not have to before this table
        could be typed in.

        **A rebuild that lands mid-edit is held.** Every design change reaches here, and
        so does every storm or soil control, so a rebuild can arrive while the user is
        halfway through a depth on row 17. Replacing the items would take the editor and
        the half-typed value with them. The rows are stashed instead and drawn when the
        editor closes -- which it does on focus-out, so the hold is bounded.

        **The user's place is kept.** With thirty-one features against an eight-row
        viewport, a rebuild that reset the scroll would put them back at row 1 after
        every single edit. The current row is restored by feature *id* rather than by
        row number, because a feature added or removed between builds shifts every
        position after it.
        """
        rows = list(rows or [])
        if self.table.state() == QAbstractItemView.EditingState:
            # Asked of the view rather than tracked on a flag of our own: a flag that
            # missed its clear would wedge this table permanently, and the view cannot
            # be wrong about whether it is editing.
            self._pending = (rows, context)
            return
        self._pending = None
        self._rows = rows
        if not rows:
            self.setVisible(False)
            self.table.setRowCount(0)
            self.footer.setText("")
            return

        anchor_id = None
        current = self.table.currentRow()
        if 0 <= current < self.table.rowCount() and current < len(self._displayed_ids):
            anchor_id = self._displayed_ids[current]
        anchor_col = self.table.currentColumn()
        scroll = self.table.verticalScrollBar().value()
        row_count_changed = self.table.rowCount() != len(rows)

        self._filling = True
        try:
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                self._fill_row(r, row)
        finally:
            self._filling = False
        self._displayed_ids = [row.get("id") for row in rows]

        # A full re-measure only when the shape changed. Re-measuring every build makes
        # the columns shimmy under the cursor as a dash becomes "0.60 m", which is
        # unpleasant to type next to and pointless when the same features are still
        # listed.
        if row_count_changed:
            self.table.resizeColumnsToContents()
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        else:
            self._grow_columns_to_fit()
        self._size_to_contents(len(rows))
        self.footer.setText(self._footer_text(rows, context or {}))
        self.setVisible(True)
        self._restore_place(anchor_id, anchor_col, scroll)

    def _grow_columns_to_fit(self):
        """Widen any column whose content no longer fits. Never narrow one.

        The shape-changed guard above stops the columns shimmying while the user types,
        and on its own it also stops a column ever widening — so a column that gains
        content it has never held before shows an ellipsis instead, for as long as the
        same features are listed. That is not hypothetical: this table opens on a design
        whose features have no spillway yet, so Sill is laid out for an em dash, and the
        moment a depth is typed the elevation it exists to show renders as "84…". The
        same fill that adds the number is the one the guard skips.

        Growing without shrinking keeps both properties. A column only ever moves when it
        would otherwise hide something, and it never moves out from under the cursor to
        reclaim space a shorter string gave back.
        """
        header = self.table.horizontalHeader()
        for c in range(self.table.columnCount()):
            if header.sectionResizeMode(c) != QHeaderView.Interactive:
                continue          # column 0 stretches; leave the layout to own it
            wanted = self.table.sizeHintForColumn(c)
            if wanted > self.table.columnWidth(c):
                self.table.setColumnWidth(c, wanted)

    def _restore_place(self, anchor_id, anchor_col, scroll):
        """Put the current cell and the scroll offset back after a rebuild."""
        if anchor_id is not None:
            try:
                target = self._displayed_ids.index(anchor_id)
            except ValueError:
                target = None                     # the feature is gone; leave it be
            if target is not None:
                col = anchor_col if 0 <= anchor_col < len(_HEADERS) else 0
                blocked = self.table.blockSignals(True)
                try:
                    self.table.setCurrentCell(target, col)
                finally:
                    self.table.blockSignals(blocked)
        self.table.verticalScrollBar().setValue(scroll)

    def _on_editor_closed(self, _editor, _hint):
        """Draw whatever arrived while the editor was open.

        Deferred a turn: this runs while the view is still tearing the editor down, and
        a rebuild from inside that is the same hazard commit_depth avoids.
        """
        if self._pending is None:
            return
        QTimer.singleShot(0, self._drain_pending)

    def _drain_pending(self):
        pending, self._pending = self._pending, None
        if pending is not None:
            self.set_rows(*pending)

    def depth_for_editor(self, table_row):
        """What the spin box opens on: this sill's depth, or what a fresh one would be."""
        if not (0 <= table_row < len(self._rows)):
            return 0.0
        row = self._rows[table_row]
        depth = row.get("sill_depth_m")
        if depth is None:
            depth = row.get("seed_depth_m")
        try:
            return float(depth)
        except (TypeError, ValueError):
            return 0.0

    def commit_depth(self, table_row, value):
        """Report a committed depth -- one event-loop turn later, deliberately.

        The apply path ends in a full rebuild of this table. Running that from inside
        the delegate would have setRowCount replacing the item under an editor Qt has
        not finished closing, because commitData calls setModelData *before*
        closeEditor. Handing it to the event loop lets the teardown finish first.
        """
        if self._filling or not (0 <= table_row < len(self._rows)):
            return
        index = self._rows[table_row].get("index")
        if index is None:
            return
        depth = float(value)
        QTimer.singleShot(0, lambda: self.depth_edited.emit(index, depth))

    def summary(self):
        """One-line state for the section header — counts, worst first."""
        if not self._rows:
            return ""
        live = [r for r in self._rows if r.get("state") != "disabled"]
        if not live:
            return f"{len(self._rows)} features · all disabled"
        bad = sum(1 for r in live if r.get("state") == "fail")
        unsited = sum(1 for r in live
                      if r.get("state") in ("undesigned", "unsited", "ok")
                      and not r.get("sited"))
        bits = [f"{len(live)} feature{'s' if len(live) != 1 else ''}"]
        if bad:
            bits.append(f"{bad} needs attention")
        elif unsited:
            bits.append(f"{unsited} not sited")
        return " · ".join(bits)

    # ------------------------------------------------------------------ internals

    def _on_cell_clicked(self, row, col):
        if not (0 <= row < len(self._rows)):
            return
        data = self._rows[row]
        index = data.get("index")
        if index is None:
            return
        self.feature_selected.emit(index)
        # The markers double as the placement affordance. Siting a spillway used to
        # mean selecting the feature in a different widget first, then finding a row in
        # the tool menu — a coupling nobody discovers. Here the thing you click is the
        # thing you are placing.
        # The thing you click is the thing you are placing — which is what the
        # comment above has always claimed and what one shared cell could not do.
        if data.get("state") != "disabled":
            if col == _COL_SITED:
                self.place_requested.emit(index, "outflow")
            elif col == _COL_INLET:
                self.place_requested.emit(index, "inflow")

    def _on_cell_double_clicked(self, row, col):
        # Double-click already meant "open the properties dialog" everywhere on this
        # table, and it is also the gesture that opens an editor. On the one editable
        # column the editor wins, or the modal would come up over it and its OK would
        # then write its own spillway over whatever was typed. Every other cell in the
        # same row still opens the dialog.
        if col == _COL_DEPTH and self._depth_editable(self._rows[row] if 0 <= row < len(self._rows) else {}):
            return
        if 0 <= row < len(self._rows):
            index = self._rows[row].get("index")
            if index is not None:
                self.edit_requested.emit(index)

    def _fill_row(self, r, row):
        state = row.get("state")
        disabled = state == "disabled"

        # Positional, and in lock-step with _HEADERS. A check pins the two together.
        cells = [
            (row.get("name", "—"), _FAINT if disabled else _INK,
             Qt.AlignmentFlag.AlignLeft),
            self._flow_cell(row, disabled),
            self._head_cell(row, disabled),
            self._width_cell(row, disabled),
            self._freeboard_cell(row, disabled),
            self._depth_cell(row, disabled),
            self._sill_cell(row, disabled),
            self._storage_cell(row, disabled),
            self._sited_cell(row, disabled),
            self._inlet_cell(row, disabled),
        ]
        # Built once per row, not once per cell: it is a ten-paragraph string and there
        # are nine cells and up to thirty-one rows behind every rebuild.
        tooltip = self._row_tooltip(row)
        editable = self._depth_editable(row)
        for c, (text, colour, align) in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setForeground(_brush(colour))
            item.setTextAlignment(int(align | Qt.AlignmentFlag.AlignVCenter))
            item.setToolTip(tooltip)
            # QTableWidgetItem is editable by *default*, so this table was read-only
            # only because of its edit triggers. Now that those are open, the flag has
            # to come off every cell that is not the one editable column -- otherwise
            # every column accepts typing and silently discards it on the next rebuild,
            # which looks from the outside exactly like it worked.
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if c == _COL_WIDTH:
                # The one cell whose figure cannot be reconciled with the head beside it
                # by the weir equation, and it is not meant to be: where the sill is
                # shallower than the design head the width is solved at what the sill can
                # pass, while Head and Freeboard stay on what the type designs for, so
                # that a sill too shallow for its storm still reads as one. The properties
                # dialog says this inline under its own Min-width row; without it here the
                # table shows two numbers that look like an arithmetic error.
                note = self._width_note(row)
                if note:
                    item.setToolTip(note)
            if c == _COL_DEPTH:
                item.setToolTip(H.SPILLWAY_DEPTH_COLUMN)
                # The editor reads this, never the cell text. Not EditRole:
                # QTableWidgetItem.setText writes DisplayRole and EditRole together, so
                # a float there would sit beside a formatted string in the same slot.
                item.setData(Qt.ItemDataRole.UserRole, self.depth_for_editor(r))
                if editable:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, c, item)

    @staticmethod
    def _depth_editable(row):
        """Whether this row's sill depth can be typed in.

        One condition, and it is the properties dialog's own: a depth is measured down
        from the containment level, so without one there is no datum to resolve a crest
        against and nothing to store. It covers every ineligible row for free, because
        `_spillway_row` returns before it reads the datums on a disabled feature and on
        one with no design flow.
        """
        return row.get("state") != "disabled" and row.get("rim_elevation") is not None

    @staticmethod
    def _depth_cell(row, disabled):
        """The sill's drop below containment -- the one figure on this row you set.

        Deliberately `sill_depth_m` and not the stored `drop_below_rim_m`: the stored one
        is re-based only when a design is restored, so after an earthworks re-analysis it
        can be measured against a rim nothing computes any more. Rendering it would mean
        typing back the number on screen moved the crest -- a no-op edit that is not one.

        A feature with no spillway shows a dash rather than the depth it would open at.
        The Width column prints "2.4 m needed" for the same state, but Width cannot be
        typed in; here a number that is not the stored number is the very thing the round
        trip above has to rule out. The seed appears when the editor opens.
        """
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        if row.get("rim_elevation") is None:
            # Same word _freeboard_cell uses for the same cause -- one vocabulary for it.
            text = "no DEM" if row.get("state") == "no_datum" else "—"
            return (text, _FAINT, Qt.AlignmentFlag.AlignRight)
        depth = row.get("sill_depth_m")
        if depth is None:
            return ("—", _WARN, Qt.AlignmentFlag.AlignRight)
        # A crest above the level that contains it is a fault, not a shallow sill, and
        # spillway_validity reports it. The editor keeps it reachable rather than
        # clamping it away; the colour says it is not a working number.
        colour = _BAD if depth < 0 else _INK
        return (f"{depth:.2f} m", colour, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _flow_cell(row, disabled):
        if disabled:
            return ("disabled", _FAINT, Qt.AlignmentFlag.AlignRight)
        flow = row.get("peak_flow_m3s")
        if not flow:
            return ("no flow", _FAINT, Qt.AlignmentFlag.AlignRight)
        return (f"{flow * 1000:,.0f} L/s", _MUTED, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _head_cell(row, disabled):
        """Target while the width is free; achieved once a width is committed.

        Never both. They are the same number whenever the width tracks, so showing two
        columns would imply a distinction that does not exist in that mode — and hide
        the one case where it does.
        """
        if disabled or not row.get("peak_flow_m3s"):
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        target = row.get("target_head_m")
        if row.get("width_auto") or row.get("actual_head_m") is None:
            return (f"{target:.2f} m target", _MUTED, Qt.AlignmentFlag.AlignRight)
        actual = row["actual_head_m"]
        over = actual - (target or 0.0)
        colour = _INK if over <= _HEAD_TOLERANCE_M else _BAD
        return (f"{actual:.2f} m actual", colour, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _width_cell(row, disabled):
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        required = row.get("required_width_m")
        if not required:
            # A sill with no depth left to spill through is not a feature with no width
            # requirement, and rendering both as a faint dash says nothing about the most
            # undersized sill it is possible to draw. `spillway_validity` skips both of
            # its width checks here too — there is no requirement for them to test — so
            # this cell is the only thing that can name the condition. The dialog's own
            # words, so the two surfaces describe it identically.
            #
            # Two words, matching "no DEM" and "no flow" elsewhere in these cells: the
            # column is sized to "4.0 m auto" and `set_rows` deliberately skips
            # `resizeColumnsToContents` when the row count has not changed, so a longer
            # string is not laid out again -- it is elided to "no depth to ..." mid-edit,
            # which the tooltip then has to rescue. The tooltip carries the sentence.
            if row.get("sizing_head_m") is not None and row["sizing_head_m"] <= 0:
                return ("no depth", _BAD, Qt.AlignmentFlag.AlignRight)
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        if not row.get("designed"):
            # Sized but not committed to: the point of showing it is that the number
            # exists before the decision does.
            return (f"{required:.1f} m needed", _WARN, Qt.AlignmentFlag.AlignRight)
        built = row.get("built_width_m") or 0.0
        if row.get("width_auto"):
            return (f"{built:.1f} m auto", _MUTED, Qt.AlignmentFlag.AlignRight)
        short = built + 0.01 < required
        return (f"{built:.1f} m built", _BAD if short else _INK,
                Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _width_note(row):
        """The sentence explaining a sill-limited width, or "" where it is not one.

        Computed copy rather than a `help_text` constant — it quotes a per-row figure, and
        `test_tooltip_copy_lives_in_help_text` exempts exactly that. The wording is the
        properties dialog's, deliberately: one fact, one phrasing.
        """
        sizing = row.get("sizing_head_m")
        target = row.get("target_head_m")
        if sizing is None or target is None or sizing >= target - 0.005:
            return ""
        if sizing <= 0:
            return H.SPILLWAY_WIDTH_SILL_LIMITED_NONE
        return (f"{H.SPILLWAY_WIDTH_SILL_LIMITED}\n\n"
                f"Solved at the {sizing:.2f} m this sill can pass, not the "
                f"{target:.2f} m this type designs for.")

    @staticmethod
    def _freeboard_cell(row, disabled):
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        if row.get("rim_elevation") is None:
            return ("no DEM", _FAINT, Qt.AlignmentFlag.AlignRight)
        freeboard = row.get("freeboard_m")
        if freeboard is None:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        minimum = row.get("freeboard_min_m") or 0.0
        if freeboard < 0:
            colour = _BAD
        elif freeboard + 1e-6 < minimum:
            colour = _BAD
        elif freeboard < minimum + 0.05:
            colour = _WARN
        else:
            colour = _GOOD
        return (f"{freeboard:.2f} m", colour, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _sill_cell(row, disabled):
        """The level water is designed to leave at — and whether the terrain agrees.

        Three elevations describe one sill and they are allowed to disagree: the
        designed crest, the sill **as burned** into the DEM, and where the finished pond
        was actually measured letting go. The column shows the designed figure, because
        that is the one the user set; a disagreement is marked and named in the tooltip,
        because a mark is enough to make someone look and a third decimal place in a
        narrow column is not.
        """
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        crest = row.get("crest_elevation")
        if crest is None:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        burned = row.get("burned_sill_m")
        actual = row.get("actual_spill_level_m")
        if burned is not None and burned > crest + _LEVEL_TOLERANCE_M:
            # The bank is still standing where the notch was meant to be.
            return (f"{crest:.2f} m  not cut", _BAD, Qt.AlignmentFlag.AlignRight)
        if actual is not None and burned is not None                 and actual > burned + _LEVEL_TOLERANCE_M:
            # A notch was cut and the pond is leaving somewhere else anyway.
            return (f"{crest:.2f} m  bypassed", _BAD, Qt.AlignmentFlag.AlignRight)
        if actual is not None and actual < crest - _LEVEL_TOLERANCE_M:
            # Something lower on the rim is the control; the sill is not.
            return (f"{crest:.2f} m  not control", _WARN,
                    Qt.AlignmentFlag.AlignRight)
        if row.get("passes_this_event") is False:
            return (f"{crest:.2f} m  untested", _MUTED, Qt.AlignmentFlag.AlignRight)
        colour = _GOOD if burned is not None else _INK
        return (f"{crest:.2f} m", colour, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _storage_cell(row, disabled):
        """What this sill leaves the feature holding, and what that costs.

        The decision the user is actually making when they move a crest, readable across
        the whole design instead of one modal dialog at a time. Blank until the feature
        has been flooded — a zero here would read as "this sill gives up nothing", which
        is the opposite of not knowing.
        """
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        held = row.get("sill_storage_m3")
        pct = row.get("given_up_pct")
        if held is None:
            return ("not measured", _FAINT, Qt.AlignmentFlag.AlignRight)
        if pct is None:
            return (f"{held:,.0f} m³", _MUTED, Qt.AlignmentFlag.AlignRight)
        # Giving something up is what a spillway is for, so this is only coloured once
        # the sill rather than the excavation has become what sizes the feature.
        colour = _WARN if pct >= 40 else _INK
        return (f"{held:,.0f} m³  −{pct:.0f}%", colour, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _sited_cell(row, disabled):
        """The outflow half. Its own column since the inlet got one of its own —
        two placements dispatched from one cell could only ever arm one tool."""
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignCenter)
        out = _OUT_GLYPH if row.get("sited") else f"{_OUT_GLYPH}·"
        colour = _MUTED if row.get("sited") else _WARN
        return (out, colour, Qt.AlignmentFlag.AlignCenter)

    @staticmethod
    def _inlet_cell(row, disabled):
        """The inlet half, with the same sited/unsited colouring as the outflow.

        Amber for unsited reads the same way here as beside it: nothing has been
        placed. An inlet is optional, so this is an affordance rather than a
        shortfall — the tooltip says which.
        """
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignCenter)
        inlet = _IN_GLYPH if row.get("inlet_sited") else f"{_IN_GLYPH}·"
        colour = _MUTED if row.get("inlet_sited") else _WARN
        return (inlet, colour, Qt.AlignmentFlag.AlignCenter)

    def _row_tooltip(self, row):
        """Everything the cells had to compress, in words."""
        name = row.get("name", "This feature")
        if row.get("state") == "disabled":
            return (f"{name} is disabled, so it takes no catchment and has nothing to "
                    f"pass. Re-enable it to size its overflow.")

        bits = []
        flow = row.get("peak_flow_m3s")
        if flow:
            upstream = row.get("upstream_m3s") or 0.0
            own = flow - upstream
            line = f"Peak design flow {flow * 1000:,.0f} L/s"
            if upstream > 0:
                line += (f" — {own * 1000:,.0f} from its own catchment, "
                         f"{upstream * 1000:,.0f} arriving from features upslope")
            bits.append(line + ".")
        else:
            bits.append(
                f"{name} has no design flow yet — run Baseline, and set a peak "
                f"intensity, before its overflow can be sized.")

        if not row.get("designed") and row.get("required_width_m"):
            bits.append(
                f"No spillway designed yet. At {row['target_head_m']:.2f} m of head it "
                f"would need a {row['required_width_m']:.1f} m sill.")

        if row.get("rim_elevation") is None and flow:
            bits.append(
                "No spill level — the crest and freeboard cannot be checked until a "
                "DEM is loaded.")
        elif row.get("freeboard_m") is not None:
            minimum = row.get("freeboard_min_m") or 0.0
            bits.append(
                f"At the design flow the water surface sits {row['freeboard_m']:.2f} m "
                f"below the level this feature is held to; this type is designed for "
                f"{minimum:.2f} m.")

        held = row.get("sill_storage_m3")
        full = row.get("containment_storage_m3")
        if held is not None and full is not None:
            given = row.get("given_up_m3") or 0.0
            bits.append(
                f"With the crest where it is, this holds {held:,.0f} m³ of the "
                f"{full:,.0f} m³ it would hold with no spillway — giving up "
                f"{given:,.0f} m³. Both measured by flooding this feature alone on the "
                f"terrain model.")

        source = row.get("containment_source")
        if source == "measured":
            bits.append(
                f"The level above ({row['rim_elevation']:.2f} m) is where the built "
                f"feature was measured to pond to, not where bare ground sits.")
        elif source == "berm":
            bits.append(
                f"The level above ({row['rim_elevation']:.2f} m) is the companion berm's "
                f"crest as built — the water is held by the bank, not the hillside.")
        elif row.get("ew_type") == "dam":
            bits.append(
                "For a dam the level above is the wall crest you specified, so this "
                "margin is measured against the wall you intend to build rather than "
                "against existing ground.")

        crest = row.get("crest_elevation")
        burned = row.get("burned_sill_m")
        actual = row.get("actual_spill_level_m")
        if crest is not None and (burned is not None or actual is not None):
            line = f"Designed sill {crest:.2f} m."
            if burned is not None:
                line += f" The burn cut it to {burned:.2f} m"
                if burned > crest + _LEVEL_TOLERANCE_M:
                    line += (" — higher than the design, so the notch was refused and "
                             "the bank is still there. Check the message bar for why.")
                elif burned < crest - _LEVEL_TOLERANCE_M:
                    line += (" — lower, because the ground along the notch was already "
                             "under the sill, so cutting it moved nothing.")
                else:
                    line += " exactly."
            if actual is not None:
                line += f" The finished pond was measured letting go at {actual:.2f} m"
                if burned is not None and actual > burned + _LEVEL_TOLERANCE_M:
                    line += (" — above the notch, so the water is leaving somewhere "
                             "else and the spillway is not taking it.")
                elif actual < crest - _LEVEL_TOLERANCE_M:
                    line += (" — below the sill, so a lower point on the rim is the "
                             "control and this spillway never comes into play.")
                else:
                    line += ", which is the sill."
            bits.append(line)

        surcharge = row.get("surcharge_storage_m3")
        full = row.get("containment_storage_m3")
        held = row.get("sill_storage_m3")
        if surcharge is not None and full is not None and held is not None:
            line = (f"At the design storm the water stands at "
                    f"{row['surcharge_level_m']:.2f} m, holding {surcharge:,.0f} m³ — "
                    f"between the {held:,.0f} m³ the sill holds and the "
                    f"{full:,.0f} m³ that would reach the top of the structure. That "
                    f"band is the spillway doing its job.")
            if row.get("spillway_insufficient"):
                line += (" It reaches the top, so the spillway is not passing enough: "
                         "water leaves over the structure as well as through it.")
            bits.append(line)

        if row.get("passes_this_event") is False and row.get("event_level_m") is not None:
            bits.append(
                f"The modelled event fills this to {row['event_level_m']:.2f} m, which "
                f"is below the sill — so this spillway passes nothing in this run. It "
                f"is untested rather than proven; size it against the design storm you "
                f"want it to survive, not this one.")

        for note in row.get("notes", []):
            bits.append(note)

        for problem in row.get("problems", []):
            bits.append("⚠ " + problem)

        bits.append("Type in the Sill depth column to set the sill. Click the markers "
                    "to site the overflow; double-click any other cell for the full "
                    "properties.")
        return "\n\n".join(bits)

    def _footer_text(self, rows, context):
        live = [r for r in rows if r.get("state") != "disabled"]
        bits = []

        # Leads, because it qualifies every number above it: an intensity nobody chose
        # is not a design storm, and the widths are linear in it.
        if context.get("intensity_is_default") and not context.get("has_idf"):
            bits.append(
                f"Every flow here is at {context.get('intensity_mm_hr', 0):.0f} mm/hr — "
                f"the starting value, not a lookup for your site. Enter your HIRDS "
                f"table, or use Compare on the Baseline stage, before treating these "
                f"widths as sized."
            )
        if context.get("harvesting_coefficient"):
            bits.append(
                "The runoff coefficient is a water-harvesting figure, calibrated for "
                "ordinary rain rather than the extreme event on saturated ground. It "
                "will undersize every overflow below."
            )

        failing = [r for r in live if r.get("state") == "fail"]
        if failing:
            worst = failing[0]
            problem = (worst.get("problems") or ["needs attention"])[0]
            bits.append(f"{worst.get('name')}: {problem}")

        undesigned = [r for r in live if not r.get("designed")]
        if undesigned:
            names = ", ".join(r.get("name", "?") for r in undesigned[:3])
            more = "" if len(undesigned) <= 3 else f" (+{len(undesigned) - 3} more)"
            bits.append(
                f"{len(undesigned)} feature{'s' if len(undesigned) != 1 else ''} have no "
                f"spillway designed: {names}{more}. They will still overflow — at "
                f"whichever point of the containing ground happens to be lowest. Type a "
                f"depth in the Sill depth column to design one."
            )

        unsited = [r for r in live if r.get("designed") and not r.get("sited")]
        if unsited:
            bits.append(
                f"{len(unsited)} sized but not sited. A width without a location is not "
                f"yet a design — place it where you can armour it."
            )

        # Named once, quietly, because it decides whether the capacity figures elsewhere
        # on the panel can be carried to site.
        #
        # This used to say only that storage is reported to full depth and a crest set
        # low holds less than that — a true caveat about a question nothing answered.
        # The Storage column answers it now, so the line says which figure is which
        # instead, and falls back to the caveat only while nothing has been measured.
        crested = [r for r in live if r.get("crest_elevation") is not None]
        if crested:
            if any(r.get("sill_storage_m3") is not None for r in crested):
                bits.append(
                    "The storage column is what each feature holds up to its sill, "
                    "measured on the terrain model. The capacity figures elsewhere on "
                    "the panel are to full depth, so they are the larger number."
                )
            else:
                bits.append(
                    "Storage elsewhere on the panel is reported to each feature's full "
                    "depth, not down to its crest, so a crest set well below the spill "
                    "level holds less than the capacity figure states. Run Re-analyse "
                    "with Earthworks and this list will measure what each sill leaves."
                )

        return "  ·  ".join(bits)

    def _size_to_contents(self, n_rows):
        header = self.table.horizontalHeader().height()
        row_h = self.table.rowHeight(0) if n_rows else 22
        self.table.setFixedHeight(header + row_h * min(n_rows, 8) + 6)


def _brush(hex_colour):
    from qgis.PyQt.QtGui import QBrush, QColor
    return QBrush(QColor(hex_colour))
