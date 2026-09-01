"""The proofreading sheet. One local HTML page per batch.

    python outreach/tools/sheet.py batch-01
    python outreach/tools/sheet.py batch-01 --open

Writes `outreach/outbox/<batch>/review.html` - eighteen drafts on one scrollable
page instead of eighteen files, grouped by tier, each with its gate findings and
a Copy button.

Two things about this file that are deliberate:

**It is local and it stays local.** It carries the names and addresses of real
people. It is written inside outbox/, which is gitignored and enforced by
check.py. Never publish it, never paste it into a hosted page, never attach it.

**Copy puts PLAIN TEXT on the clipboard, not HTML.** Pasting HTML into Gmail
carries markup and makes the email an HTML email, which is what guardrails.md
rule 9 exists to prevent. The button hands the clipboard the same string
`check.py --render` would print, so an ordinary Ctrl+V lands as plain text.

A draft with a HARD finding is shown so you can see what is wrong, but its Copy
button is disabled. The gate is not routed around from here either.
"""

from __future__ import annotations

import html
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import (LINK_MARKERS, ROOT, VIDEO_UNSET, load_yaml,   # noqa: E402
                   links_from_cfg, render_body, run, split_doc, subject_of)

ROUTE_NOTE = {
    "direct": "Copy, paste into the mail client, send.",
    "form": "Copy, then open the form and paste it in. No attachment.",
    "lookup": "Address not resolved yet - this one is not sendable.",
    "warm": "You already have a route to this one.",
}

# Grouping is the tracker's own Tier column, which is send priority. There is no
# A/B/C depth tier any more: every contact has a hand-written fit behind it, so
# every email is a researched one and all of them get read.
PRIORITY_NOTE = {
    "1": "Highest priority. Send these first.",
    "2": "",
    "3": "",
    "4": "",
    "5": "",
    "?": "No tracker priority recorded.",
}

CSS = """
:root{--bg:#fbfaf7;--fg:#1a1a18;--mut:#6b6b63;--line:#e0ded6;--card:#fff;
--warn:#8a6d1f;--hard:#a12d2d;--ok:#2f6b45;--accent:#2b5c8a}
@media(prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e8e6e1;--mut:#9a988f;
--line:#2d2f34;--card:#1d1f23;--warn:#d9b45a;--hard:#e0736e;--ok:#7bbd93;--accent:#7fb3dd}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.25rem 6rem;
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:760px;margin:0 auto}
h1{font-size:1.3rem;margin:0 0 .25rem}
.sub{color:var(--mut);margin:0 0 2rem;font-size:.9rem}
.banner{background:var(--card);border-left:3px solid var(--warn);padding:.75rem 1rem;
margin-bottom:1.5rem;border-radius:0 4px 4px 0;font-size:.9rem}
h2{font-size:.78rem;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);
margin:2.5rem 0 .35rem;font-weight:600}
h2 .note{text-transform:none;letter-spacing:0;font-weight:400;display:block;margin-top:.2rem}
.brief{margin-top:.75rem;border-top:1px solid var(--line);padding-top:.5rem}
.brief summary{cursor:pointer;color:var(--accent);font-size:.85rem;user-select:none}
.briefbody{font-size:.85rem;line-height:1.5;margin-top:.5rem}
.briefbody h4{margin:.9rem 0 .2rem;font-size:.8rem;letter-spacing:.02em;
text-transform:uppercase;color:var(--mut)}
.briefbody h4.danger{color:var(--hard)}
.briefbody ul{margin:.2rem 0 .2rem 1.1rem;padding:0}
.briefbody li{margin:.2rem 0}
.briefbody p{margin:.2rem 0;color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:1.1rem 1.25rem;margin-bottom:1rem}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;flex-wrap:wrap}
.org{font-weight:600}
.to{color:var(--mut);font-size:.85rem;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.subj{margin:.6rem 0 .1rem;font-size:.8rem;color:var(--mut)}
.subj b{color:var(--fg);font-weight:600;font-size:.95rem}
pre{white-space:pre-wrap;word-wrap:break-word;margin:.7rem 0 0;font:inherit}
.findings{margin:.7rem 0 0;font-size:.85rem}
.f{display:block;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.8rem}
.HARD{color:var(--hard)}.WARN{color:var(--warn)}
.clean{color:var(--ok);font-size:.85rem;margin-top:.7rem}
button{background:var(--accent);color:#fff;border:0;border-radius:4px;
padding:.4rem .85rem;font:inherit;font-size:.85rem;cursor:pointer}
button:disabled{background:var(--line);color:var(--mut);cursor:not-allowed}
.blocked{color:var(--hard);font-size:.85rem;font-weight:600}
.route{display:inline-block;background:var(--line);color:var(--fg);border-radius:3px;
padding:.05rem .4rem;margin-right:.4rem;font-size:.75rem;letter-spacing:.03em}
.hd .note{color:var(--mut);font-size:.8rem;margin-top:.2rem}
.to a{color:var(--accent)}
"""

JS = """
document.addEventListener('click', async e => {
  const b = e.target.closest('button[data-copy]'); if (!b) return;
  // Plain text on the clipboard, never HTML - pasting HTML into Gmail makes the
  // email an HTML email, which is the thing this pipeline avoids.
  try { await navigator.clipboard.writeText(
          document.getElementById(b.dataset.copy).textContent);
        b.textContent = 'Copied'; }
  catch (_) { b.textContent = 'Press Ctrl+C';
        const r = document.createRange(); r.selectNodeContents(
          document.getElementById(b.dataset.copy));
        const s = getSelection(); s.removeAllRanges(); s.addRange(r); }
  setTimeout(() => { b.textContent = 'Copy'; }, 1800);
});
"""


BRIEF_CSS_MARK = None


def brief_html(slug: str) -> str:
    """brief.md as collapsed HTML. Empty string when there is no card.

    A deliberately small markdown subset - headings, bullets, bold, italics -
    because the file is written by build_briefs.py and nothing else, so the
    shapes are known. Anything unrecognised is escaped and shown as a paragraph
    rather than dropped, since silently losing a DO NOT USE line would be the
    one failure that matters here.
    """
    src = ROOT / "recipients" / slug / "brief.md"
    if not src.exists():
        return ""

    out, in_list, para = [], False, []

    def close():
        nonlocal in_list
        # A wrapped paragraph is one paragraph. Emitting each source line as its
        # own <p> split every **bold** span across two elements and left the
        # markers showing, which is how the header of every card was rendering.
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in src.read_text(encoding="utf-8").split("\n")[1:]:
        line = raw.rstrip()
        if not line.strip():
            close()
            continue
        if line.startswith("## "):
            close()
            title = line[3:].strip()
            cls = "danger" if title.startswith("DO NOT USE") else ""
            out.append('<h4 class="{}">{}</h4>'.format(cls, html.escape(title)))
            continue
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + inline(line[2:].strip()) + "</li>")
            continue
        if in_list:
            close()
        para.append(line.strip())
    close()

    return ('<details class="brief"><summary>Facts about this recipient '
            '&mdash; verified, and what not to use</summary>'
            '<div class="briefbody">' + "".join(out) + "</div></details>")


def inline(s: str) -> str:
    """**bold**, *italic* and nothing else. Escaped first, so no markup leaks."""
    s = html.escape(s)
    parts = s.split("**")
    s = "".join(x if i % 2 == 0 else "<b>" + x + "</b>" for i, x in enumerate(parts))
    parts = s.split("*")
    if len(parts) % 2:
        s = "".join(x if i % 2 == 0 else "<i>" + x + "</i>" for i, x in enumerate(parts))
    return s


def build(batch: str) -> Path:
    cfg = load_yaml(ROOT / "config.yaml")
    outdir = ROOT / "outbox" / batch
    if not outdir.exists():
        raise SystemExit("no such batch: {}".format(outdir))

    links = links_from_cfg(cfg)
    sender = str((cfg.get("sender") or {}).get("email", ""))
    unset = [n for n in LINK_MARKERS if VIDEO_UNSET.search(links[n])]
    if VIDEO_UNSET.search(sender):
        unset.append("sender.email")
    video_ready = not unset

    drafts = sorted(p for p in outdir.glob("*.md") if not p.name.endswith(".sent.md"))
    if not drafts:
        raise SystemExit("no drafts in {}".format(outdir))

    by_tier, blocked, warned = {}, 0, 0
    for path in drafts:
        front, sections = split_doc(path.read_text(encoding="utf-8"))
        rep = run(path)
        hard = rep.hard
        blocked += 1 if hard else 0
        warned += 1 if (rep.items and not hard) else 0
        by_tier.setdefault(str(front.get("priority", "?")), []).append(
            (path, front, sections, rep, hard))

    rows = []
    for tier in ("1", "2", "3", "4", "5", "?"):
        items = by_tier.get(tier)
        if not items:
            continue
        rows.append('<h2>Tier {} &middot; {} draft{}<span class="note">{}</span></h2>'
                    .format(tier, len(items), "" if len(items) == 1 else "s",
                            html.escape(PRIORITY_NOTE.get(tier, ""))))
        for path, front, sections, rep, hard in items:
            eid = "t-" + path.stem
            body = render_body(sections, links=links if video_ready else None)
            text = "{}\n\n{}".format(subject_of(sections), body)

            findings = ""
            if rep.items:
                findings = '<div class="findings">' + "".join(
                    '<span class="f {}">[{}] {} {}</span>'.format(
                        lv, lv, html.escape(c), html.escape(m))
                    for lv, c, m in sorted(rep.items,
                                           key=lambda r: 0 if r[0] == "HARD" else 1)
                ) + "</div>"
            else:
                findings = '<div class="clean">gate clean</div>'

            if hard:
                btn = '<span class="blocked">BLOCKED &middot; {} hard</span>'.format(hard)
            elif not video_ready:
                btn = '<button disabled>Not ready: {}</button>'.format(html.escape(unset[0]))
            else:
                btn = '<button data-copy="{}">Copy</button>'.format(eid)

            route = str(front.get("route", "")).lower()
            target = str(front.get("to", ""))
            # A form is opened, not written to. Making it a link is the whole
            # difference between "paste this somewhere" and one click.
            if route == "form" and target.lower().startswith(("http://", "https://")):
                shown = '<a href="{0}" target="_blank" rel="noopener">{0}</a>'.format(
                    html.escape(target))
            else:
                shown = html.escape(target)

            rows.append(
                '<div class="card"><div class="hd"><div>'
                '<div class="org">{org}</div>'
                '<div class="to"><span class="route">{route}</span> {to}</div>'
                '<div class="note">{rnote}</div></div>{btn}</div>'
                '<div class="subj">Subject &nbsp;<b>{subj}</b></div>'
                '<pre id="{eid}">{text}</pre>{findings}{brief}</div>'.format(
                    org=html.escape(str(front.get("org", path.stem))),
                    route=html.escape(route or "?"), to=shown,
                    rnote=html.escape(ROUTE_NOTE.get(route, "")),
                    subj=html.escape(subject_of(sections)),
                    eid=eid, text=html.escape(text), findings=findings, btn=btn,
                    brief=brief_html(path.stem)))

    banner = ""
    if not video_ready:
        banner = ('<div class="banner"><b>Not ready to send: {}.</b> Drafts are '
                  'shown for proofreading and copying is disabled. Set these in '
                  'config.yaml and rebuild the sheet - every draft takes them from '
                  'there.</div>'.format(html.escape(", ".join(unset))))

    page = (
        "<!doctype html><html><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>{batch} &middot; proofread</title><style>{css}</style></head><body>"
        "<div class=wrap><h1>{batch}</h1>"
        "<p class=sub>{n} drafts &middot; {blocked} blocked &middot; {warned} with warnings"
        " &middot; local file, do not share</p>{banner}{rows}</div>"
        "<script>{js}</script></body></html>"
    ).format(batch=html.escape(batch), css=CSS, js=JS, banner=banner,
             n=len(drafts), blocked=blocked, warned=warned, rows="".join(rows))

    out = outdir / "review.html"
    out.write_text(page, encoding="utf-8")
    print("\n{}".format(out))
    print("  {} drafts, {} blocked, {} with warnings".format(len(drafts), blocked, warned))
    if not video_ready:
        print("  copy disabled - config.yaml is missing: {}".format(", ".join(unset)))
    print("\n  Local file. It names real people - do not publish or attach it.")
    return out


def main(argv: list) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    out = build(argv[0])
    if "--open" in argv:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
