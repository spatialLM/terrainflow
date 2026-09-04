"""The roster. Reads the outreach tracker and tracks what state each contact is in.

    python outreach/tools/queue.py --import-xlsx <TerrainFlow_Outreach_Tracker.xlsx>
    python outreach/tools/queue.py --status
    python outreach/tools/queue.py --batch
    python outreach/tools/queue.py --sent <slug> [...]
    python outreach/tools/queue.py --decline <slug> [...]

The tracker is the source of truth and it already carries the hard part: a
category, a region and a hand-written "Why It's a Fit" for every contact. None of
that is re-derived. The importer only files it, works out how each contact is
actually reached, and maps the category onto a bracket family.

Everything this writes lives under gitignored paths. It never touches a tracked
file.
"""

from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import ROOT, load_yaml          # noqa: E402  same-dir reuse

RECIPIENTS = ROOT / "recipients"
CSV_PATH = ROOT / "recipients.csv"
SUPPRESSION = ROOT / "suppression.txt"

STATES = ("new", "researched", "drafted", "sent", "replied", "declined")

# Tracker column letters. Kept here rather than inline so a reshuffled sheet is
# one edit rather than a hunt.
COL = {"priority": "A", "category": "B", "org": "C", "region": "D",
       "fit": "E", "url": "F", "contact": "G", "route": "H", "notes": "I"}


# ---------------------------------------------------------------- xlsx

def read_xlsx(path: Path) -> list:
    """Rows as dicts, stdlib only.

    openpyxl is not installed and will not be added for one spreadsheet. This
    handles both shared-string and inline-string cells, which is the whole of
    what the tracker uses.
    """
    z = zipfile.ZipFile(path)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in si.iter(ns + "t")) for si in ss.iter(ns + "si")]
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(ns + "row"):
        cells = {}
        for c in row.iter(ns + "c"):
            col = "".join(ch for ch in (c.get("r") or "") if ch.isalpha())
            v, inline = c.find(ns + "v"), c.find(ns + "is")
            if inline is not None:
                val = "".join(t.text or "" for t in inline.iter(ns + "t"))
            elif c.get("t") == "s" and v is not None and shared:
                val = shared[int(v.text)]
            else:
                val = v.text if v is not None else ""
            cells[col] = (val or "").strip()
        rows.append(cells)
    return rows


# ---------------------------------------------------------------- mapping

# Ordered: first match wins. Built from the tracker's own Category column, so a
# new category that matches nothing is reported rather than guessed at.
BRACKET_RULES = [
    (1, r"permaculture|rainwater harvesting standards"),
    (5, r"^academic\b.*(swale|flood|natural flood|stormwater|data science|hydrolog)"
        r"|swale design"),
    (6, r"open-source (gis|geospatial)"),
    (4, r"\bnbs\b|suds|flood engineering|engineering firm"),
    (3, r"^nz\b"),
    (2, r"restoration|watershed|water-harvesting ngo|conservation|agroforestry"
        r"|reforestation|grazing|development agency|water research|peatland"),
]

# Three rows are categorised "Already on your list" and carry no usable signal,
# so they are placed by name. Anything else unmatched is reported, not guessed.
BY_ORG = {
    "wsp": 3,
    "opw": 4,
    "office of public works": 4,
    "bord na": 2,
}


def bracket_for(category: str, org: str):
    cat = (category or "").lower()
    for bracket, pattern in BRACKET_RULES:
        if re.search(pattern, cat):
            return bracket
    low = (org or "").lower()
    for key, bracket in BY_ORG.items():
        if key in low:
            return bracket
    return None


def route_for(route_cell: str) -> str:
    """How this contact is actually reached.

    The tracker writes this as prose, so it is classified rather than read. An
    address is unambiguous; everything else is a form, a lookup, or a route the
    user already has.
    """
    r = (route_cell or "").strip()
    # An address with a note after it is still an address. Requiring the cell to
    # be nothing but the address routed two sendable contacts to `form`.
    if address_of(r):
        return "direct"
    low = r.lower()
    if re.search(r"check .*(page|directory)|search linkedin|not confirmed", low):
        return "lookup"
    if "already have" in low:
        return "warm"
    return "form"


def address_of(route_cell: str) -> str:
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", route_cell or "")
    return m.group().lower() if m else ""


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s[:60] or "unnamed"


# ---------------------------------------------------------------- records

FIELDS = ("org", "contact", "email", "form_url", "url", "region", "category",
          "bracket", "priority", "route", "state", "batch", "notes")


def read_record(d: Path) -> dict:
    p = d / "record.yaml"
    return load_yaml(p) if p.exists() else {}


def write_record(d: Path, rec: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for k in FIELDS:
        if rec.get(k) not in (None, ""):
            lines.append('{}: "{}"'.format(k, str(rec[k]).replace('"', "'")))
    (d / "record.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fit(d: Path, rec: dict, fit: str, notes: str) -> None:
    """The tracker's own research, filed as `owner` evidence.

    This is Liam's judgement rather than a fetched quote, so it is trusted but
    never verbatim-checked. /email-research adds `web` evidence alongside it.
    """
    lines = ["# {}".format(rec.get("org", "")), "",
             "source: outreach tracker (owner evidence - not machine-checkable)", ""]
    if rec.get("category"):
        lines += ["## Category", rec["category"], ""]
    if rec.get("region"):
        lines += ["## Region", rec["region"], ""]
    if fit:
        lines += ["## Why it is a fit", fit, ""]
    if notes:
        lines += ["## Notes", notes, ""]
    (d / "fit.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- import

def cmd_import_xlsx(path: Path, cfg: dict) -> int:
    if not path.exists():
        print("no such file: {}".format(path))
        return 2
    rows = read_xlsx(path)
    header_at = next((i for i, r in enumerate(rows)
                      if (r.get("C") or "").lower().startswith("organisation")), 1)
    data = [r for r in rows[header_at + 1:] if (r.get(COL["org"]) or "").strip()]

    seen_slug, seen_addr = {}, {}
    added = updated = 0
    warnings, unmapped = [], []

    for i, row in enumerate(data, start=header_at + 2):
        org = row.get(COL["org"], "").strip()
        route_cell = row.get(COL["route"], "")
        route = route_for(route_cell)
        addr = address_of(route_cell)

        slug = slugify(org)
        if slug in seen_slug:
            slug = "{}-{}".format(slug, i)[:70]
            warnings.append("line {}: second row for {} - filed as '{}'".format(i, org, slug))
        seen_slug[slug] = org

        if addr:
            if addr in seen_addr:
                warnings.append("line {}: {} duplicates {}".format(i, addr, seen_addr[addr]))
            seen_addr.setdefault(addr, org)

        bracket = bracket_for(row.get(COL["category"], ""), org)
        if bracket is None:
            unmapped.append("{}  ({})".format(org, row.get(COL["category"], "")))

        d = RECIPIENTS / slug
        existing = read_record(d)
        rec = dict(existing)
        rec.update({
            "org": org,
            "contact": row.get(COL["contact"], ""),
            "email": addr,
            "form_url": "" if addr else route_cell,
            "url": row.get(COL["url"], ""),
            "region": row.get(COL["region"], ""),
            "category": row.get(COL["category"], ""),
            "bracket": bracket or "",
            "priority": row.get(COL["priority"], ""),
            "route": route,
            "notes": row.get(COL["notes"], ""),
        })
        rec.setdefault("state", "new")
        write_record(d, rec)
        write_fit(d, rec, row.get(COL["fit"], ""), row.get(COL["notes"], ""))
        updated += 1 if existing else 0
        added += 0 if existing else 1

    for w in warnings:
        print("  ! " + w)
    if unmapped:
        print("\n  categories that matched no bracket - place these by hand:")
        for u in unmapped:
            print("    " + u)
    print("\n{} added, {} updated, {} rows read".format(added, updated, len(data)))
    print("next: /email-research --batch")
    return 0


def cmd_import_csv(cfg: dict) -> int:
    """For contacts added after the tracker. Same fields, one row each."""
    if not CSV_PATH.exists():
        print("no {}".format(CSV_PATH))
        print("columns: org,contact,email,url,region,category,priority,notes")
        return 2
    n = 0
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            if not row.get("org"):
                continue
            d = RECIPIENTS / slugify(row["org"])
            rec = dict(read_record(d))
            rec.update({k: v for k, v in row.items() if v})
            rec["route"] = route_for(row.get("email", ""))
            rec["bracket"] = rec.get("bracket") or bracket_for(row.get("category", ""), row["org"]) or ""
            rec.setdefault("state", "new")
            write_record(d, rec)
            n += 1
    print("{} rows imported from recipients.csv".format(n))
    return 0


# ---------------------------------------------------------------- reporting

def all_records():
    if not RECIPIENTS.exists():
        return []
    out = []
    for d in sorted(RECIPIENTS.iterdir()):
        if d.is_dir():
            rec = read_record(d)
            if rec:
                out.append((d, rec))
    return out


def _tally(recs, key):
    out = {}
    for _, rec in recs:
        out[str(rec.get(key, "") or "?")] = out.get(str(rec.get(key, "") or "?"), 0) + 1
    return dict(sorted(out.items()))


def cmd_status(cfg: dict) -> int:
    recs = all_records()
    if not recs:
        print("nothing imported yet - run --import-xlsx <tracker.xlsx>")
        return 0

    names = cfg.get("brackets") or {}
    routes = cfg.get("routes") or {}
    print("\n{} contacts".format(len(recs)))
    print("\n  state     " + "  ".join("{} {}".format(k, v) for k, v in _tally(recs, "state").items()))
    print("  priority  " + "  ".join("T{} {}".format(k, v) for k, v in _tally(recs, "priority").items()))

    print("\n  route")
    for k, v in _tally(recs, "route").items():
        print("    {:<8} {:>3}   {}".format(k, v, routes.get(k, "")))

    print("\n  bracket")
    for k, v in _tally(recs, "bracket").items():
        label = names.get(k, names.get(str(k), "unassigned - place by hand"))
        print("    {:<3} {:>3}   {}".format(k, v, label))

    lookups = [r.get("org") for _, r in recs if r.get("route") == "lookup"]
    if lookups:
        print("\n  {} need an address resolved before they can be sent:".format(len(lookups)))
        for org in lookups:
            print("    " + org)

    flagged = [(r.get("org"), r.get("notes", "")) for _, r in recs
               if re.search(r"not self-published|lookup service|rarely|not confirmed",
                            r.get("notes", ""), re.I)]
    if flagged:
        print("\n  provenance warnings from the tracker:")
        for org, note in flagged:
            print("    {:<38} {}".format((org or "")[:38], note[:70]))
    return 0


def cmd_batch(cfg: dict) -> int:
    """Form the next batch, highest tracker priority first.

    Capped by config. A brand-new sending address has no reputation, so the
    direct sends are spread rather than fired in one afternoon.
    """
    size = int((cfg.get("send") or {}).get("batch_size", 18))
    existing = {rec.get("batch") for _, rec in all_records() if rec.get("batch")}
    n = 1
    while "batch-{:02d}".format(n) in existing:
        n += 1
    label = "batch-{:02d}".format(n)

    ready = [(d, r) for d, r in all_records()
             if r.get("state") in ("researched", "drafted") and not r.get("batch")]
    if not ready:
        print("nothing ready - research and draft some first")
        return 0
    ready.sort(key=lambda dr: (str(dr[1].get("priority", "9")), dr[1].get("org", "")))
    picked = ready[:size]
    for d, rec in picked:
        rec["batch"] = label
        write_record(d, rec)

    print("\n{}  ({} of {} ready)".format(label, len(picked), len(ready)))
    for d, rec in picked:
        print("  T{} {:<8} {:<40} {}".format(rec.get("priority", "?"), rec.get("route", ""),
                                             (rec.get("org") or "")[:40], d.name))
    if len(ready) > size:
        print("\n  {} held back for the next batch.".format(len(ready) - size))
    print("\nnext: /email-proof {}".format(label))
    return 0


def cmd_sent(slugs: list) -> int:
    for slug in slugs:
        d = RECIPIENTS / slug
        rec = read_record(d)
        if not rec:
            print("  ? no such contact: {}".format(slug))
            continue
        rec["state"] = "sent"
        write_record(d, rec)
        print("  sent  {}".format(rec.get("org", slug)))
    return 0


def cmd_decline(slugs: list) -> int:
    """Declined, bounced, or asked not to hear again. The gate reads this file."""
    lines = []
    for slug in slugs:
        d = RECIPIENTS / slug
        rec = read_record(d)
        if not rec:
            print("  ? no such contact: {}".format(slug))
            continue
        rec["state"] = "declined"
        write_record(d, rec)
        target = rec.get("email") or rec.get("form_url") or slug
        lines.append("{}  # {}".format(target, rec.get("org", slug)))
        print("  suppressed  {}".format(rec.get("org", slug)))
    if lines:
        with SUPPRESSION.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0


def main(argv: list) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cfg = load_yaml(ROOT / "config.yaml")
    cmd, rest = argv[0], argv[1:]
    if cmd == "--import-xlsx":
        if not rest:
            print("--import-xlsx needs the path to the tracker")
            return 2
        return cmd_import_xlsx(Path(rest[0]), cfg)
    if cmd == "--import":
        return cmd_import_csv(cfg)
    if cmd == "--status":
        return cmd_status(cfg)
    if cmd == "--batch":
        return cmd_batch(cfg)
    if cmd == "--sent":
        return cmd_sent(rest)
    if cmd == "--decline":
        return cmd_decline(rest)
    print("unknown: {}".format(cmd))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
