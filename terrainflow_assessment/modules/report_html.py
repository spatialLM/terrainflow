"""
report_html.py — render a :class:`Report` as a self-contained HTML page.

The second renderer for the shared document model. It walks exactly the same
section list as :mod:`terrainflow_assessment.qgis.adapters.layout_pdf`, so the
PDF and the HTML cannot disagree about what the report says: they disagree only
about paper size and page breaks.

That guarantee is enforced, not just intended. :data:`SECTION_HANDLERS` is the
renderer's dispatch table, and a test asserts it covers every concrete
``Section`` subclass and matches the PDF renderer's table exactly — so adding a
section type to one renderer without the other fails the suite.

Self-contained: styles are inline and every image is embedded as a data URI, so
the file can be emailed on its own.
"""

import base64
import html
import logging
import os

_log = logging.getLogger(__name__)

# Presentation follows the plugin's original HTML report: a flat-UI palette on a
# grey field, white cards with soft shadows, #2980b9 as the single accent (rules,
# table headers, section bars) and amber for anything that wants attention. The
# document model decides what is said; this decides only how it looks.
_ACCENT = "#2980b9"

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f0f2f5; color: #2c3e50; line-height: 1.6; }
.page { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
h1 { font-size: 2rem; color: #1a252f; border-bottom: 3px solid #2980b9;
     padding-bottom: 12px; margin-bottom: 8px; }
.subtitle { color: #7f8c8d; font-size: 1rem; }
.stages { color: #7f8c8d; font-size: 0.8rem; text-transform: uppercase;
          letter-spacing: 0.5px; margin-bottom: 32px; }
h2 { font-size: 1.4rem; color: #1a252f; margin: 32px 0 12px;
     border-left: 4px solid #2980b9; padding-left: 12px; }
h3 { font-size: 1.1rem; color: #34495e; margin: 20px 0 8px; }
p { margin-bottom: 12px; }
.card { background: #fff; border-radius: 10px; padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }
.card > h3:first-child { margin-top: 0; }
.hero { background: #fff; border-radius: 10px; padding: 28px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px;
        border-left: 6px solid #2980b9; }
.hero .value { font-size: 3rem; font-weight: 700; line-height: 1.1; }
.hero .label { font-size: 1.1rem; color: #34495e; margin-top: 4px; }
.hero .sub { font-size: 0.9rem; color: #7f8c8d; margin-top: 6px; }
.hero.good { border-left-color: #27ae60; } .hero.good .value { color: #27ae60; }
.hero.warn { border-left-color: #f39c12; } .hero.warn .value { color: #b9770e; }
.hero.bad  { border-left-color: #c0392b; } .hero.bad  .value { color: #c0392b; }
.hero.info .value { color: #2c3e50; }
.stats-grid { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }
.stat-card { background: #fff; border-radius: 8px; padding: 16px 20px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.08); min-width: 180px; flex: 1; }
.stat-label { font-size: 0.8rem; color: #7f8c8d; text-transform: uppercase;
              letter-spacing: 0.5px; margin-bottom: 4px; }
.stat-value { font-size: 1.5rem; font-weight: 700; color: #2c3e50; }
.stat-unit { font-size: 0.9rem; font-weight: 400; color: #7f8c8d; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
thead { background: #2980b9; color: white; }
th { padding: 10px 12px; text-align: left; font-weight: 600;
     vertical-align: top; }
td { padding: 9px 12px; border-bottom: 1px solid #ecf0f1; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f8f9fa; }
tbody th { background: #f8f9fa; color: #34495e; font-weight: 600;
           border-bottom: 1px solid #ecf0f1; width: 38%; }
.note { color: #7f8c8d; font-size: 0.85rem; margin-top: 10px; }
.chart-wrap { background: #fff; border-radius: 10px; padding: 20px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }
.chart-wrap img { max-width: 100%; height: auto; display: block;
                  margin: 0 auto; }
.chart-wrap figcaption { color: #7f8c8d; font-size: 0.85rem; margin-top: 12px;
                         text-align: center; }
.map-key { display: flex; flex-wrap: wrap; justify-content: center;
           gap: 8px 18px; margin-top: 14px; }
.key-item { display: inline-flex; align-items: center; gap: 6px;
            font-size: 0.8rem; color: #5f7176; }
.key-swatch { display: inline-block; border: 1px solid rgba(0,0,0,0.25); }
.key-ramp { display: inline-block; width: 46px; height: 10px;
            border: 1px solid rgba(0,0,0,0.25); }
.callout { padding: 16px 20px; border-radius: 6px; margin-bottom: 24px;
           border-left: 4px solid #f39c12; background: #fff3cd; }
.callout .t { display: block; font-weight: 700; margin-bottom: 6px; }
.callout p { font-size: 0.9rem; margin-bottom: 6px; }
.callout p:last-child { margin-bottom: 0; }
.callout.info { background: #eaf2f8; border-left-color: #2980b9; }
.callout.good { background: #eafaf1; border-left-color: #27ae60; }
.callout.bad  { background: #fdedec; border-left-color: #c0392b;
                color: #922b21; font-weight: 600; }
.wide { overflow-x: auto; }
.wide table { min-width: 820px; }
footer { text-align: center; color: #bdc3c7; font-size: 0.8rem;
         margin-top: 40px; padding-top: 16px; border-top: 1px solid #ecf0f1; }
@media print {
  body { background: #fff; }
  .page { max-width: none; padding: 0; }
  .card, .hero, .stat-card, .chart-wrap { box-shadow: none;
                                          border: 1px solid #ecf0f1; }
  .brk { page-break-before: always; }
  h2 { page-break-after: avoid; }
  table, .chart-wrap, .callout { page-break-inside: avoid; }
  thead { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
"""


def _esc(text):
    return html.escape(str(text if text is not None else ""))


def _para(text):
    """Blank-line separated text into paragraphs, newlines into breaks."""
    blocks = [b for b in str(text).split("\n\n") if b.strip()]
    return "".join(
        "<p>" + "<br>".join(_esc(line) for line in b.split("\n")) + "</p>"
        for b in blocks)


def _data_uri(path):
    try:
        with open(path, "rb") as handle:
            return "data:image/png;base64," + base64.b64encode(
                handle.read()).decode("ascii")
    except OSError as exc:
        _log.info("could not embed %s: %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
# Section handlers — one per Section type, same set as the PDF renderer
# ---------------------------------------------------------------------------

def _heading(section, ctx):
    level = 2 if section.level == 1 else 3
    cls = ' class="brk"' if section.level == 1 else ""
    return f"<h{level}{cls}>{_esc(section.text)}</h{level}>"


def _paragraph(section, ctx):
    return _para(section.text)


def _callout(section, ctx):
    # "warn" is the bare .callout — the amber, left-barred box the original
    # report used for its caveats — so only the other tones add a modifier.
    tone = section.tone if section.tone in ("info", "bad", "good") else ""
    title = (f'<span class="t">{_esc(section.title)}</span>'
             if section.title else "")
    return f'<div class="callout {tone}">{title}{_para(section.text)}</div>'


def _hero(section, ctx):
    tone = section.tone if section.tone in ("info", "warn", "bad", "good") else "info"
    sub = f'<div class="sub">{_esc(section.sub)}</div>' if section.sub else ""
    return (f'<div class="hero {tone}"><div class="value">{_esc(section.value)}'
            f'</div><div class="label">{_esc(section.label)}</div>{sub}</div>')


def _stat_grid(section, ctx):
    if not section.cards:
        return ""
    cells = []
    for card in section.cards:
        label, value, sub = (list(card) + ["", "", ""])[:3]
        unit = (f'<div class="stat-unit">{_esc(sub)}</div>' if sub else "")
        cells.append(f'<div class="stat-card">'
                     f'<div class="stat-label">{_esc(label)}</div>'
                     f'<div class="stat-value">{_esc(value)}</div>'
                     f"{unit}</div>")
    return f'<div class="stats-grid">{"".join(cells)}</div>'


def _card(title, inner):
    """Every table sits on its own white card, as the original report did."""
    heading = f"<h3>{_esc(title)}</h3>" if title else ""
    return f'<div class="card">{heading}{inner}</div>'


def _key_value_table(section, ctx):
    if not section.rows:
        return ""
    body = "".join(
        f'<tr><th scope="row">{_esc(k)}</th><td>{_esc(v)}</td></tr>'
        for k, v in section.rows)
    return _card(section.title, f"<table><tbody>{body}</tbody></table>")


def _data_table(section, ctx):
    if not section.rows:
        return ""
    head = ""
    if any(str(h).strip() for h in section.headers):
        head = ("<thead><tr>"
                + "".join(f"<th>{_esc(h)}</th>" for h in section.headers)
                + "</tr></thead>")
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>"
        for row in section.rows)
    note = f'<p class="note">{_esc(section.note)}</p>' if section.note else ""
    table = f"<table>{head}<tbody>{body}</tbody></table>"
    if section.wide:
        # The per-feature schedules are wider than the page; scroll them inside
        # their own card rather than letting the document scroll sideways.
        table = f'<div class="wide">{table}</div>'
    return _card(section.title, table + note)


def _image_ref(section, ctx):
    path = (ctx.get("images") or {}).get(section.key)
    if not path or not os.path.exists(path):
        # Same rule as the PDF: never a null image, always the fallback the
        # model carries, so a chart is never the sole carrier of a number.
        if section.fallback is not None:
            return render_section(section.fallback, ctx)
        return ""
    return _figure(path, section.caption)


def _map_ref(section, ctx):
    path = (ctx.get("maps") or {}).get(section.key)
    if section.reason or not path or not os.path.exists(path):
        reason = section.reason or "The layers this map needs are not available."
        return f'<div class="callout">{_para(reason)}</div>'
    return _figure(path, section.caption,
                   extra=_legend(getattr(section, "legend", None)))


def _legend(entries):
    """The map key, as chips. Empty string when the map has nothing to explain.

    A gradient stop renders as a real CSS gradient rather than as its endpoints,
    because the reader is matching a wash on the map against it, not a pair of
    colours.
    """
    if not entries:
        return ""
    chips = []
    for entry in entries:
        colours = list(getattr(entry, "colours", ()) or [])
        if entry.kind == "ramp" and len(colours) > 1:
            swatch = ('<span class="key-ramp" style="background:linear-gradient'
                      f'(to right,{",".join(_esc(c) for c in colours)})"></span>')
        else:
            # A point is a disc the size of the marker it stands for, not a
            # swatch-wide oval; a line is a rule; a fill is a block.
            shape = {
                "point": "width:10px;height:10px;border-radius:50%",
                "fill": "width:16px;height:10px;border-radius:2px",
            }.get(entry.kind, "width:16px;height:3px;border-radius:1px")
            swatch = (f'<span class="key-swatch" style="background:'
                      f'{_esc(entry.colour)};{shape}"></span>')
        chips.append(f'<span class="key-item">{swatch}'
                     f'<span>{_esc(entry.label)}</span></span>')
    return f'<div class="map-key">{"".join(chips)}</div>'


def _figure(path, caption, extra=""):
    uri = _data_uri(path)
    if not uri:
        return ""
    cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
    return (f'<div class="chart-wrap"><img alt="{_esc(caption)}" src="{uri}">'
            f"{extra}{cap}</div>")


def _page_break(section, ctx):
    # Only meaningful in print; on screen the document scrolls continuously.
    return '<div class="brk"></div>'


def _handlers():
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
        Heading: _heading,
        Paragraph: _paragraph,
        Callout: _callout,
        Hero: _hero,
        StatGrid: _stat_grid,
        KeyValueTable: _key_value_table,
        DataTable: _data_table,
        ImageRef: _image_ref,
        MapRef: _map_ref,
        PageBreak: _page_break,
    }


#: Section type -> handler. The PDF renderer keeps an equivalent table and a
#: test asserts the two cover exactly the same types.
SECTION_HANDLERS = _handlers()


def render_section(section, ctx):
    handler = SECTION_HANDLERS.get(type(section))
    if handler is None:
        _log.warning("no HTML handler for %s", type(section).__name__)
        return ""
    return handler(section, ctx)


def render_html(report, images=None, maps=None):
    """The whole document as one self-contained HTML string."""
    ctx = {"images": images or {}, "maps": maps or {}}
    done = [k for k, v in report.completeness.items() if v]
    missing = [k for k, v in report.completeness.items() if not v]
    stages = "Stages run: " + (", ".join(done) if done else "none")
    if missing:
        stages += " &nbsp;·&nbsp; Not run: " + ", ".join(missing)

    body = "".join(render_section(s, ctx) for s in report.sections)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(report.title)}</title>"
        f"<style>{_CSS}</style></head><body><div class=\"page\">"
        f"<h1>{_esc(report.title)}</h1>"
        f'<p class="subtitle">{_esc(report.subtitle)}</p>'
        f'<p class="stages">{stages}</p>'
        f"{body}"
        f"<footer>{_esc(report.footer)}</footer>"
        "</div></body></html>"
    )


def write_html(path, report, images=None, maps=None):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_html(report, images=images, maps=maps))
    return path
