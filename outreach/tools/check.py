"""The gate. Stdlib only, deterministic, runs before anything is sent.

    python outreach/tools/check.py outreach/outbox/batch-01/some-org.md
    python outreach/tools/check.py --render outreach/outbox/batch-01/some-org.md
    python outreach/tools/check.py --render --wrap <draft>   # to read, not to paste
    python outreach/tools/check.py --privacy
    python outreach/tools/check.py --templates
    python outreach/tools/check.py --selftest

Two checks are the point and the rest are convenience:

1. PRIVACY    this repo is public. Nothing naming a real person may be tracked
              by git. .gitignore is a request; this is the test.
2. EVIDENCE   every personalised claim resolves to a verbatim quote captured
              from the organisation's own site. The model cannot assert anything
              about a recipient it did not fetch.

--render is the only way to get send-ready text, and it refuses on a HARD
finding. You cannot copy an email that has not been through the gate.

Exit 0 clean or warnings only. Exit 1 on any HARD finding.

The YAML and n-gram helpers are lifted from gitas-voice/tools/check.py, which is
the same design pointed at a different problem. Kept as a copy rather than an
import because these two repos must not depend on each other.
"""

from __future__ import annotations

import re
import subprocess
import textwrap
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # outreach/
REPO = ROOT.parent                                     # TerrainFlow/

# Must mirror the outreach block in TerrainFlow/.gitignore. Duplicated on
# purpose: if the two drift, the selftest is what notices.
PRIVATE_PATHS = [
    "outreach/recipients.csv",
    "outreach/recipients",
    "outreach/outbox",
    "outreach/replies",
    "outreach/corpus",
    "outreach/suppression.txt",
]


# ---------------------------------------------------------------- mini YAML
# pyyaml is not a dependency of this repo's tooling and will not become one for
# an email pipeline. Handles exactly the subset our own files use: nested maps,
# block and inline lists, quoted and bare scalars, comments.


def _scalar(raw: str):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        return [_scalar(p) for p in _split_inline(inner)] if inner else []
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1].replace("''", "'")
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d*\.\d+", raw):
        return float(raw)
    return raw


def _key(raw: str) -> str:
    """Unquote a mapping key.

    Values go through _scalar, which strips quotes; keys do not. A quoted key
    that keeps its quote marks can never match a word-boundary pattern built
    from it, which silently disables the rule instead of failing loudly.
    """
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1].replace("''", "'")
    return raw


def _split_inline(s: str) -> list:
    out, buf, quote = [], "", None
    for ch in s:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote, buf = ch, buf + ch
        elif ch == ",":
            out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf)
    return out


def _strip_comment(line: str) -> str:
    out, quote = "", None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out += ch
    return out.rstrip()


def parse_yaml(text: str):
    lines = []
    for raw in text.splitlines():
        body = _strip_comment(raw)
        if body.strip():
            lines.append((len(body) - len(body.lstrip()), body.strip()))

    def parse(idx: int, indent: int):
        if idx < len(lines) and lines[idx][1].startswith("- "):
            items = []
            while idx < len(lines) and lines[idx][0] == indent and lines[idx][1].startswith("- "):
                items.append(_scalar(lines[idx][1][2:]))
                idx += 1
            return items, idx
        node = {}
        while idx < len(lines) and lines[idx][0] == indent:
            key, _, rest = lines[idx][1].partition(":")
            idx += 1
            if rest.strip():
                node[_key(key)] = _scalar(rest)
            elif idx < len(lines) and lines[idx][0] > indent:
                node[_key(key)], idx = parse(idx, lines[idx][0])
            else:
                node[_key(key)] = None
        return node, idx

    return parse(0, 0)[0] if lines else {}


def load_yaml(path: Path):
    if not path.exists():
        return {}
    return parse_yaml(path.read_text(encoding="utf-8"))


def leaves(node) -> list:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [x for v in node for x in leaves(v)]
    if isinstance(node, dict):
        return [x for v in node.values() for x in leaves(v)]
    return []


# ---------------------------------------------------------------- text utils

WORD = re.compile(r"[a-z0-9']+")
# Working notes. Always stripped - they are for the gate, not the reader.
MARKER = re.compile(r"\[(?:e:\d+|f:TF-\d+)\]")
# Content placeholders. Substituted when a URL is supplied and otherwise left
# VISIBLE, so a preview shows "overview is here: [video]" rather than a sentence
# that trails off into nothing and reads as a bug.
LINK_MARKER = re.compile(r"\[(?:video|repo|linkedin)\]")
# An unset config value is empty or still says TODO.
VIDEO_UNSET = re.compile(r"^\s*$|todo", re.I)
# Filled from config at render time. Never pasted into a draft: 51 drafts is 51
# find-and-replaces, and one of them gets missed.
LINK_MARKERS = ("video", "repo", "linkedin")
E_REF = re.compile(r"\[e:(\d+)\]")
F_REF = re.compile(r"\[f:(TF-\d+)\]")
SECTION = re.compile(r"^([A-Z][A-Z -]{2,})\s*$")


def normalise(text: str) -> list:
    return WORD.findall(text.lower())


def ngrams(words: list, n: int) -> set:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def split_doc(md: str):
    """Return (frontmatter dict, {SECTION: body}).

    A file with no frontmatter and no headings comes back as everything under
    "", so a bare draft still gets checked rather than silently passing.
    """
    lines = md.splitlines()
    front = {}
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                front = parse_yaml("\n".join(lines[1:i]))
                lines = lines[i + 1:]
                break
    out, head, buf = {}, "", []
    for line in lines:
        m = SECTION.match(line)
        if m:
            out.setdefault(head, "\n".join(buf))
            head, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    out.setdefault(head, "\n".join(buf))
    return front, out


def dedent(text: str) -> str:
    body = [ln for ln in text.splitlines()]
    pads = [len(ln) - len(ln.lstrip()) for ln in body if ln.strip()]
    pad = min(pads) if pads else 0
    return "\n".join(ln[pad:] if len(ln) >= pad else ln for ln in body).strip("\n")


def render_body(sections: dict, links: dict = None, wrap: bool = False) -> str:
    """The text as the recipient will see it - markers stripped, indent removed.

    `links` fills [video], [repo] and [linkedin]. Empty at check time, which is
    what lets a batch be drafted and proofread before the video or the sending
    address exist; --render supplies the real URLs and refuses if config has not
    been given them.

    Stripping a marker leaves the space that preceded it, so `sites [e:1],`
    becomes `sites ,` and a marker that wrapped onto its own line leaves a line
    beginning with a full stop. Both are small and both would be noticed, so the
    orphaned whitespace is closed up here rather than left for the eye.
    """
    raw = dedent(sections.get("BODY", sections.get("", "")))
    for name, url in (links or {}).items():
        if url and not VIDEO_UNSET.search(url):
            raw = raw.replace("[{}]".format(name), url)
    text = MARKER.sub("", raw)
    text = re.sub(r"[ \t]*\n[ \t]*([.,;:?!])", r"\1", text)   # pulled onto its own line
    text = re.sub(r"[ \t]+([.,;:?!])", r"\1", text)           # left dangling before punctuation
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text).rstrip()

    # Each prose paragraph becomes ONE line, because this text is pasted into a
    # Gmail compose window and Gmail treats every newline as a real break. Hard
    # wrapping at 78 columns is the old mailing-list convention and it pastes as
    # a ragged narrow column that looks broken on a phone. Let the mail client
    # wrap; it knows the width and we do not.
    #
    # `wrap` is for reading in a terminal, never for the copy path. A paragraph
    # holding a URL is left exactly as written either way - that is the
    # signature, and wrapping a link is how a link stops working.
    out = []
    for para in re.split(r"\n\s*\n", text):
        lines = para.splitlines()
        # A line that IS a URL is the signature, where the break is deliberate.
        # A URL sitting inside a sentence - which is what [video] becomes - is
        # ordinary prose and joins like any other. Testing for "http" anywhere
        # in the paragraph conflated the two and left the video sentence broken
        # across lines.
        if any(ln.strip().startswith(("http://", "https://")) for ln in lines) \
                or not any(len(ln) > 50 for ln in lines):
            out.append(para)
            continue
        joined = " ".join(ln.strip() for ln in lines)
        out.append(textwrap.fill(joined, width=78, break_long_words=False,
                                 break_on_hyphens=False) if wrap else joined)
    return "\n\n".join(out)


def subject_of(sections: dict) -> str:
    return dedent(sections.get("SUBJECT", "")).strip()


EVIDENCE_LINE = re.compile(
    r"^\s*e(\d+)\s*\|\s*(?:(web|owner)\s*\|\s*)?(\S+)\s*\|\s*(.+?)\s*$")


def evidence_of(sections: dict) -> dict:
    """{'1': (provenance, src, quote)} from the EVIDENCE block.

    Pipe-delimited rather than nested YAML so the block stays something you can
    read and hand-edit in the middle of proofreading.

    Two provenances, both verbatim-checkable but against different files:

      web    a quote from a page that was fetched   -> research.md
      owner  a line from the outreach tracker       -> fit.md

    `owner` exists because the tracker already carries a hand-written "Why It's
    a Fit" for every contact, which is better material than anything a scrape
    produces. It is not a loophole: it still has to appear verbatim in the file
    queue.py wrote, so it cannot be invented either.

    A line with no provenance token is read as `web`.
    """
    out = {}
    for line in sections.get("EVIDENCE", "").splitlines():
        m = EVIDENCE_LINE.match(line)
        if m:
            out[m.group(1)] = (m.group(2) or "web", m.group(3), m.group(4))
    return out


# ---------------------------------------------------------------- report


class Report:
    def __init__(self) -> None:
        self.items = []

    def add(self, level: str, check: str, msg: str) -> None:
        self.items.append((level, check, msg))

    @property
    def hard(self) -> int:
        return sum(1 for lv, _, _ in self.items if lv == "HARD")

    def has(self, check: str) -> bool:
        return any(c == check for _, c, _ in self.items)

    def hard_in(self, check: str) -> bool:
        return any(lv == "HARD" and c == check for lv, c, _ in self.items)

    def render(self) -> str:
        if not self.items:
            return "  clean - no findings"
        order = {"HARD": 0, "WARN": 1}
        rows = sorted(self.items, key=lambda r: order.get(r[0], 2))
        return "\n".join("  [{:4}] {:11} {}".format(lv, chk, msg) for lv, chk, msg in rows)


# ---------------------------------------------------------------- checks


def _git(args: list, cwd: Path):
    try:
        p = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                           text=True, timeout=30)
        return p.returncode, p.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, ""


def check_privacy(rep: Report, repo: Path = REPO) -> None:
    """Nothing naming a real person may be tracked by git.

    This repo is public and a push cannot be taken back. The .gitignore block is
    a request; this is the test. Reports the offending paths rather than a count,
    because the fix is `git rm --cached <path>` and you need the path to run it.
    """
    code, _ = _git(["rev-parse", "--git-dir"], repo)
    if code is None:
        rep.add("WARN", "privacy", "git not available - could not verify")
        return
    if code != 0:
        rep.add("WARN", "privacy", "not a git repository - could not verify")
        return
    code, out = _git(["ls-files", "--"] + PRIVATE_PATHS, repo)
    tracked = [ln for ln in out.splitlines() if ln.strip()]
    for path in tracked:
        rep.add("HARD", "privacy",
                "TRACKED BY GIT: {} - run: git rm --cached '{}'".format(path, path))
    if not tracked:
        # A path that exists but is not ignored is not yet a leak, but it is one
        # `git add .` away from being one.
        for p in PRIVATE_PATHS:
            if not (repo / p).exists():
                continue
            code, _ = _git(["check-ignore", "-q", p], repo)
            if code not in (0, None):
                rep.add("HARD", "privacy",
                        "{} exists and is NOT ignored - add it to .gitignore".format(p))


def check_evidence(front: dict, sections: dict, rep: Report,
                   recipients_dir: Path) -> None:
    """Every [e:N] resolves, and every quote is verbatim in that org's research.

    The failure this exists to stop is an invented detail - "I saw your work on
    the Waikato project" - which reads as researched and burns the contact when
    it is wrong. Nothing may be asserted about a recipient that was not fetched.
    """
    body = sections.get("BODY", "")
    ev = evidence_of(sections)
    used = set(E_REF.findall(body))

    for ref in sorted(used):
        if ref not in ev:
            rep.add("HARD", "evidence", "[e:{}] has no entry in EVIDENCE".format(ref))
    for ref in sorted(set(ev) - used):
        rep.add("WARN", "evidence", "e{} is captured but never used".format(ref))

    for ref, (prov, src, _) in sorted(ev.items()):
        if prov == "web" and not src.lower().startswith(("http://", "https://")):
            rep.add("HARD", "evidence", "e{} is web evidence but the source is not a URL: {}"
                    .format(ref, src))

    slug = front.get("id")
    if not slug:
        rep.add("HARD", "evidence", "frontmatter has no id - cannot locate the evidence files")
        return

    # Each provenance is checked against the file that provenance comes from.
    files = {"web": recipients_dir / str(slug) / "research.md",
             "owner": recipients_dir / str(slug) / "fit.md"}
    hay = {}
    for prov, path in files.items():
        if path.exists():
            hay[prov] = " ".join(path.read_text(encoding="utf-8", errors="ignore").split()).lower()

    for ref, (prov, _, quote) in sorted(ev.items()):
        if prov not in hay:
            rep.add("HARD", "evidence",
                    "e{} is {} evidence but {} does not exist for '{}'"
                    .format(ref, prov, files[prov].name, slug))
            continue
        if " ".join(quote.split()).lower() not in hay[prov]:
            rep.add("HARD", "evidence",
                    'e{} quote not verbatim in {}: "{}"'.format(ref, files[prov].name, quote))


def check_classify(front: dict, rep: Report, cfg: dict, recipients_dir: Path) -> None:
    """The bracket is derived from the org's own site, and cited, or it fails.

    A wrong bracket does not produce a weak email, it produces a confidently
    irrelevant one. `unclassified` is a real outcome and must not be a guess.
    """
    slug = front.get("id")
    declared = front.get("brackets")
    if isinstance(declared, int):
        declared = [declared]
    declared = list(declared or [])
    if not declared:
        rep.add("HARD", "classify", "frontmatter declares no brackets")
        return

    # The tracker assigns the bracket by hand for all 51 contacts, so record.yaml
    # is the normal source and classification.yaml only exists for a contact
    # added later with no category of its own.
    path = recipients_dir / str(slug) / "classification.yaml"
    if not path.exists():
        rec = load_yaml(recipients_dir / str(slug) / "record.yaml")
        tracked = rec.get("bracket")
        if tracked in (None, ""):
            rep.add("HARD", "classify",
                    "no bracket for '{}' in record.yaml and no classification.yaml"
                    .format(slug))
            return
        if str(declared[0]) != str(tracked):
            rep.add("HARD", "classify",
                    "frontmatter bracket {} does not match the tracker's {}"
                    .format(declared[0], tracked))
        if len(declared) > 1:
            rep.add("WARN", "classify",
                    "secondary bracket {} is not in the tracker - make sure it is "
                    "evidenced".format(declared[1]))
        return
    cls = load_yaml(path)

    conf = str(cls.get("confidence", "")).lower()
    if conf == "unclassified":
        rep.add("HARD", "classify", "confidence is 'unclassified' - hold this one out")
    elif conf not in ("high", "medium", "low"):
        rep.add("HARD", "classify", "confidence '{}' is not high/medium/low".format(conf))

    order = [cls.get("primary")] + ([cls.get("secondary")] if cls.get("secondary") else [])
    order = [b for b in order if b is not None]
    if declared != order:
        rep.add("HARD", "classify",
                "brackets {} do not match classification {} (primary first)".format(declared, order))

    # Every bracket used must carry its own quote. A secondary asserted with no
    # evidence is the same fabrication as an invented personal detail.
    cited = set()
    for row in cls.get("evidence") or []:
        parts = [p.strip() for p in str(row).split("|")]
        if len(parts) < 3 or not parts[1].lower().startswith("http"):
            rep.add("HARD", "classify", "malformed evidence row: {}".format(row))
            continue
        try:
            cited.add(int(parts[0]))
        except ValueError:
            rep.add("HARD", "classify", "evidence row has no bracket number: {}".format(row))
    for b in declared:
        if b not in cited:
            rep.add("HARD", "classify", "bracket {} is claimed with no cited quote".format(b))

    weights = cls.get("weights") or {}
    vals = sorted((float(v) for v in weights.values() if v is not None), reverse=True)
    margin = float((cfg.get("classify") or {}).get("primary_margin", 0.15))
    if len(vals) >= 2 and (vals[0] - vals[1]) < margin:
        rep.add("WARN", "classify",
                "primary leads secondary by {:.2f}, under the {:.2f} margin - review this one"
                .format(vals[0] - vals[1], margin))


def check_blend(front: dict, sections: dict, rep: Report, cfg: dict, lex: dict) -> None:
    """Two brackets maximum, one opener, one ask.

    A blend that reaches 50/50 is a template neither reader recognises, and an
    email asking for four things earns none of them.
    """
    declared = front.get("brackets")
    if isinstance(declared, int):
        declared = [declared]
    declared = list(declared or [])
    cap = int((cfg.get("classify") or {}).get("max_brackets", 2))
    if len(declared) > cap:
        rep.add("HARD", "blend",
                "{} source brackets declared, cap is {} - this is a review case, not a blend"
                .format(len(declared), cap))

    body = render_body(sections).lower()
    families = []
    for name, phrases in (lex.get("cta_families") or {}).items():
        if any(str(p).lower() in body for p in (phrases or [])):
            families.append(name)
    limit = int((cfg.get("check") or {}).get("cta_families_hard", 3))
    if len(families) >= limit:
        rep.add("HARD", "blend",
                "{} distinct asks ({}) - an email asking for all of them earns none"
                .format(len(families), ", ".join(sorted(families))))


def check_placeholder(sections: dict, rep: Report) -> None:
    """Nothing unfilled survives to send. The one that gets noticed is 'Hi {{name}}'."""
    text = dedent(sections.get("BODY", "")) + "\n" + subject_of(sections)
    text = MARKER.sub("", text)
    text = re.sub(r"<https?://[^>]*>", "", text)
    patterns = [
        (r"\{\{.*?\}\}", "mustache placeholder"),
        (r"<[A-Za-z][A-Za-z _-]{1,30}>", "angle placeholder"),
        (r"\bTODO\b|\bTBC\b|\bFIXME\b", "TODO left in"),
        (r"\[(?:name|org|first ?name|organisation|company)\]", "square placeholder"),
        (r"_{3,}", "blank line to fill"),
        (r"\bXX+\b", "XX placeholder"),
    ]
    for pat, label in patterns:
        for m in re.finditer(pat, text, re.I):
            rep.add("HARD", "placeholder", '{}: "{}"'.format(label, m.group()[:40]))


ADDRESS = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
ROUTES = ("direct", "form", "lookup", "warm")


def check_addressing(front: dict, rep: Report, supp: Path) -> None:
    """How this one is reached, and whether it may be reached at all.

    Two thirds of the list has no email address - most are web forms, six are
    academics whose address has still to be found on a faculty page. Without a
    declared route, a form URL in the `to` field would look exactly like
    something sendable, which is how a draft ends up pasted into the wrong box.
    """
    to = str(front.get("to", "")).strip()
    route = str(front.get("route", "")).strip().lower()

    if route not in ROUTES:
        rep.add("HARD", "addressing",
                "route is '{}' - must be one of {}".format(route or "unset", ", ".join(ROUTES)))
    if not to:
        rep.add("HARD", "addressing", "frontmatter has no 'to'")
        return

    if route in ("direct", "warm"):
        if not ADDRESS.match(to):
            rep.add("HARD", "addressing",
                    "route is {} but 'to' is not an address: {}".format(route, to[:60]))
    elif route == "form":
        if ADDRESS.match(to):
            rep.add("HARD", "addressing",
                    "route is form but 'to' is an address - set route: direct instead")
    elif route == "lookup":
        # An unresolved lookup must never reach a batch. /email-research either
        # finds the published address and flips this to direct, or reports that
        # it could not - it never guesses first.last@institution.
        rep.add("HARD", "addressing",
                "route is lookup - resolve the address from a published page first")

    if not supp.exists():
        return
    for raw in supp.read_text(encoding="utf-8", errors="ignore").splitlines():
        entry = raw.split("#")[0].strip().lower()
        if entry and entry == to.lower():
            rep.add("HARD", "addressing", "{} is on the suppression list".format(to[:60]))


CLAIMY = re.compile(
    # No time units. Every email in this campaign asks for "30 minutes", which is
    # the ask rather than a claim about the software, and firing on it would put
    # the same false positive on all 51 drafts.
    r"\b\d+\s*(?:%|percent|m3|m\u00b3|litres|metres|mm|x faster)\b"
    r"|\bd-?infinity\b|\bd8\b|\bcurve number\b|\bmanning\b|\bcoverage\b|\bLTR\b"
    r"|\bcamels\b|\bwhitebox\w*\b", re.I)

# The validation paragraph is the strongest thing in the email and the easiest
# to overstate by one word. TF-30/31/32 are the claims; TF-33 is "one catchment
# so far, more to be tested before release". Separated, the claim stops being
# true and a hydrologist stops reading.
VALIDATION_CLAIMS = {"TF-30", "TF-31", "TF-32"}
VALIDATION_QUALIFIER = "TF-33"

OVERCLAIM = [
    (r"\b(?:is|was|has been|been|fully|independently|externally)\s+validated\b",
     "the tool is not validated - one catchment is (TF-71)"),
    (r"\bpeer[- ]reviewed\b", "nothing here is peer-reviewed"),
    (r"\bextensively tested\b", "one catchment (TF-33)"),
    (r"\bproven\b", "nothing is proven - say what was measured"),
    (r"\bbenchmarked\b(?!.{0,60}\bone catchment\b)",
     "implies a benchmarking programme that does not exist (TF-71)"),
    (r"\bmatches\s+\w+\s+exactly\b", "TF-32 says almost cell for cell, not exactly"),
]


def check_validation(sections: dict, rep: Report) -> None:
    """A validation claim never travels without its qualifier.

    Citations are read from the RAW body: render_body strips every marker, so
    looking for [f:TF-30] in rendered text finds nothing and the check silently
    passes everything. The prose scan below does use the rendered text, because
    an overclaim is about the words the recipient reads.
    """
    body = sections.get("BODY", "")
    refs = set(F_REF.findall(body))
    used = VALIDATION_CLAIMS & refs
    if used and VALIDATION_QUALIFIER not in refs:
        rep.add("HARD", "validation",
                "cites {} without [f:{}] - the one-catchment qualifier is what makes "
                "the claim true".format(", ".join(sorted(used)), VALIDATION_QUALIFIER))

    low = render_body(sections).lower()
    for pattern, why in OVERCLAIM:
        m = re.search(pattern, low)
        if m:
            rep.add("HARD", "validation", '"{}" - {}'.format(m.group().strip(), why))

    # Named in prose but cited to nothing: the reader can check CAMELS-NZ and
    # WhiteboxTools, so these two get held to the fact bank.
    if re.search(r"\bcamels\b|\bwhitebox\w*\b", low) and not used:
        rep.add("WARN", "validation",
                "names CAMELS-NZ or WhiteboxTools without citing TF-30/31/32")


def check_facts(sections: dict, rep: Report, facts: Path) -> None:
    """Cited fact IDs resolve, and a numeric claim without one gets a look.

    Which sentences are factual is not something a regex can decide, so this
    reports what it can honestly see: unresolvable IDs are hard, an uncited
    number is a warning for a human to read.
    """
    body = sections.get("BODY", "")
    known = set()
    if facts.exists():
        known = set(re.findall(r"^\s*(TF-\d+)", facts.read_text(encoding="utf-8"), re.M))
    elif F_REF.search(body):
        rep.add("HARD", "facts", "facts.md is missing - no cited ID can be resolved")
        return
    for ref in sorted(set(F_REF.findall(body))):
        if ref not in known:
            rep.add("HARD", "facts", "{} is not in facts.md".format(ref))

    for sent in re.split(r"(?<=[.?!])\s+", dedent(body)):
        if CLAIMY.search(MARKER.sub("", sent)) and not F_REF.search(sent):
            rep.add("WARN", "facts",
                    'claim with no [f:TF-nn]: "{}"'.format(" ".join(sent.split())[:70]))


def check_signoff(sections: dict, rep: Report, cfg: dict, lex: dict) -> None:
    body = render_body(sections)
    low = body.lower()
    sender = cfg.get("sender") or {}
    name = str(sender.get("name", "")).strip()
    repo = str((cfg.get("project") or {}).get("repo", "")).strip()

    if name and name.lower() not in low:
        rep.add("HARD", "signoff", "signature does not contain '{}'".format(name))
    if repo and repo.lower() not in low and "[repo]" not in sections.get("BODY", ""):
        rep.add("WARN", "signoff", "no link to the repo - they cannot check your work")
    if not any(str(p).lower() in low for p in (lex.get("easy_out") or [])):
        rep.add("HARD", "signoff",
                "no easy-out line - one sentence, your own words, so it reads as a person")
    for host in lex.get("tracking_hosts") or []:
        if str(host).lower() in low:
            rep.add("HARD", "signoff", "tracking or shortened link: {}".format(host))


def links_from_cfg(cfg: dict) -> dict:
    """The three config-filled link markers, by marker name."""
    proj = cfg.get("project") or {}
    return {"video": str(proj.get("video", "") or ""),
            "repo": str(proj.get("repo", "") or ""),
            "linkedin": str(proj.get("linkedin", "") or "")}


def check_video(sections: dict, rep: Report, cfg: dict) -> None:
    """The email exists to get the video watched, so it has to link it.

    Links are markers rather than pasted URLs, so 51 drafts can be written and
    proofread before the video and the LinkedIn profile exist, and then all take
    the real URLs from one place. A URL pasted into each draft is 51
    find-and-replaces and one of them gets missed.
    """
    body = sections.get("BODY", "")
    if "[video]" not in body:
        rep.add("WARN", "video",
                "no [video] marker - the email asks them to watch something, link it")
    links = links_from_cfg(cfg)
    for name in LINK_MARKERS:
        if "[{}]".format(name) in body and VIDEO_UNSET.search(links[name]):
            rep.add("WARN", "video",
                    "config project.{} is not set yet - fine for drafting, "
                    "--render will refuse until it is".format(name))


def check_lexicon(sections: dict, rep: Report, lex: dict, cfg: dict) -> None:
    body = render_body(sections)
    subject = subject_of(sections)
    text = subject + "\n" + body
    low = text.lower()

    banned = lex.get("banned_hard") or {}
    for term in leaves(banned):
        t = str(term)
        if t == "!":
            n = text.count("!")
            if n:
                rep.add("HARD", "lexicon", "{} exclamation mark(s)".format(n))
            continue
        if re.search(r"\b" + re.escape(t.lower()) + r"\b", low):
            rep.add("HARD", "lexicon", 'banned: "{}"'.format(t))

    for bad, good in (lex.get("prefer") or {}).items():
        if re.search(r"\b" + re.escape(str(bad).lower()) + r"\b", low):
            rep.add("WARN", "lexicon", '"{}" -> {}'.format(bad, good))

    # "sized", "prized" and friends end in -ized without being -ize verbs, and
    # flagged on every draft. Only words with a real stem before the suffix are
    # candidates, and the short false friends are named.
    NOT_IZE = {"sized", "sizes", "sizing", "prized", "prizes", "prizing",
               "seized", "seizing", "capsized"}
    for m in re.finditer(r"\b(\w{4,})(?:ize|ized|izing|ization)\b", low):
        if m.group() not in NOT_IZE:
            rep.add("WARN", "locale", '"{}" - en-NZ uses -ise'.format(m.group()))

    chk = cfg.get("check") or {}
    words = len(normalise(body))
    cap = int(chk.get("body_words_warn", 180))
    if words > cap:
        rep.add("WARN", "length", "{} words, target under {}".format(words, cap))
    if subject:
        scap = int(chk.get("subject_words_warn", 9))
        if len(normalise(subject)) > scap:
            rep.add("WARN", "length",
                    "subject is {} words, truncates on a phone over ~{}"
                    .format(len(normalise(subject)), scap))
    else:
        rep.add("HARD", "length", "no SUBJECT")

    rhythm = lex.get("rhythm") or {}
    sents = [s for s in re.split(r"(?<=[.?!])\s+", body) if len(normalise(s)) > 2]
    if sents:
        counts = [len(normalise(s)) for s in sents]
        lo, hi = (rhythm.get("mean_sentence_words") or [10, 20])[:2]
        mean = sum(counts) / len(counts)
        if not lo <= mean <= hi:
            rep.add("WARN", "rhythm",
                    "mean sentence {:.1f} words, target {}-{}".format(mean, lo, hi))
        first = int(rhythm.get("opening_sentence_max_words", 18))
        if counts[0] > first:
            rep.add("WARN", "rhythm",
                    "opening sentence {} words - it is read in a preview pane"
                    .format(counts[0]))


def check_sameness(rep: Report, cfg: dict, templates: Path) -> None:
    """If two bracket templates read the same, the segmentation is cosmetic.

    Same n-gram machinery as the gitas-voice firewall, pointed the other way:
    there it proves two texts are far enough apart to be original, here it
    proves five texts are far enough apart to be worth writing separately.
    """
    if not templates.exists():
        rep.add("WARN", "sameness", "no templates/ yet - nothing to compare")
        return
    n = int((cfg.get("check") or {}).get("ngram", 5))
    thresh = float((cfg.get("check") or {}).get("sameness_warn", 0.25))
    grams = {}
    for f in sorted(templates.glob("*.md")):
        _, sec = split_doc(f.read_text(encoding="utf-8"))
        g = ngrams(normalise(render_body(sec)), n)
        if g:
            grams[f.stem] = g
    names = sorted(grams)
    if len(names) < 2:
        rep.add("WARN", "sameness", "fewer than two templates - nothing to compare")
        return
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = grams[a] & grams[b]
            frac = len(shared) / min(len(grams[a]), len(grams[b]))
            if frac >= thresh:
                rep.add("WARN", "sameness",
                        "{} and {} share {:.0%} of their {}-grams - is this really two brackets?"
                        .format(a, b, frac, n))


# ---------------------------------------------------------------- driver


def run(path: Path, recipients_dir: Path = None, supp: Path = None,
        facts: Path = None, cfg: dict = None, lex: dict = None) -> Report:
    recipients_dir = recipients_dir or (ROOT / "recipients")
    supp = supp or (ROOT / "suppression.txt")
    facts = facts if facts is not None else (ROOT / "facts.md")
    cfg = cfg if cfg is not None else load_yaml(ROOT / "config.yaml")
    lex = lex if lex is not None else load_yaml(ROOT / "lexicon.yaml")

    front, sections = split_doc(path.read_text(encoding="utf-8"))
    rep = Report()
    kind = str(front.get("kind", "draft")).lower()

    check_privacy(rep)
    check_lexicon(sections, rep, lex, cfg)
    check_video(sections, rep, cfg)
    check_validation(sections, rep)
    check_placeholder(sections, rep)
    check_signoff(sections, rep, cfg, lex)
    check_facts(sections, rep, facts)
    check_blend(front, sections, rep, cfg, lex)

    # A template has no recipient, so the recipient-shaped checks would only
    # report that it is not a person. Everything above still applies to it.
    if kind != "template":
        check_evidence(front, sections, rep, recipients_dir)
        check_classify(front, rep, cfg, recipients_dir)
        check_addressing(front, rep, supp)
    return rep


# ---------------------------------------------------------------- selftest

CFG = {
    "sender": {"name": "Liam Murphy"},
    "project": {"repo": "https://github.com/spatialLM/terrainflow"},
    "check": {"body_words_warn": 180, "subject_words_warn": 9, "cta_families_hard": 3,
              "ngram": 5, "sameness_warn": 0.25},
    "classify": {"primary_margin": 0.15, "max_brackets": 2},
}

CLEAN = """---
id: fixture-org
org: Fixture Org
to: someone@example.org
route: direct
tier: B
brackets: [2]
status: draft
---

SUBJECT
  Seeing where the water goes first

BODY
  Kia ora Sam,

  You mention planting along the streambanks [e:1], which is the kind of work
  where the order matters more than the total. I have built a QGIS plugin that
  shows where surface water concentrates on a site, and what changes when you
  put a swale or a bund in [f:TF-04].

  There is a short video of it running on a real catchment [video]. If it looks
  useful, I would like half an hour of your time to hear what it gets wrong.

  If not, just say and I will leave you alone.

  Liam Murphy
  https://github.com/spatialLM/terrainflow

EVIDENCE
  e1 | https://example.org/projects | planting along the streambanks
"""


def _fixture_tree(tmp: Path) -> None:
    d = tmp / "recipients" / "fixture-org"
    d.mkdir(parents=True, exist_ok=True)
    (d / "research.md").write_text(
        "Fixture Org\n\nFrom the projects page: they have been planting along the\n"
        "streambanks since 2019 across the upper catchment.\n", encoding="utf-8")
    (d / "fit.md").write_text(
        "# Fixture Org\n\nsource: outreach tracker\n\n## Why it is a fit\n"
        "Runs a volunteer planting programme across the upper catchment.\n",
        encoding="utf-8")
    (d / "classification.yaml").write_text(
        "primary: 2\nconfidence: high\nweights:\n  b2: 0.71\n  b4: 0.19\n"
        "evidence:\n  - \"2 | https://example.org/projects | planting along the streambanks\"\n",
        encoding="utf-8")
    (tmp / "facts.md").write_text(
        "TF-04  Earthworks are burned into a copy of the DEM and the analysis re-runs.\n"
        "TF-30  Tested against CAMELS-NZ.\n"
        "TF-33  One catchment so far, more to be tested before release.\n",
        encoding="utf-8")


def _run_text(text: str, tmp: Path, lex: dict, cfg: dict = None) -> Report:
    p = tmp / "draft.md"
    p.write_text(text, encoding="utf-8")
    return run(p, recipients_dir=tmp / "recipients", supp=tmp / "suppression.txt",
               facts=tmp / "facts.md", cfg=cfg or CFG, lex=lex)


def selftest() -> int:
    """A gate never seen to fail is not known to work.

    Every check gets both directions: it fires on the thing it exists to catch,
    and it stays quiet on a clean draft. A check that only ever passes is
    indistinguishable from one that is switched off.
    """
    import tempfile

    lex = load_yaml(ROOT / "lexicon.yaml")
    if not lex:
        print("  FAIL - lexicon.yaml did not load")
        return 1

    ok = True
    results = []

    def case(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        results.append((label, passed, detail))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _fixture_tree(tmp)

        base = _run_text(CLEAN, tmp, lex)
        hard = [m for lv, c, m in base.items if lv == "HARD" and c != "privacy"]
        case("clean draft passes", not hard, "; ".join(hard[:2]))

        r = _run_text(CLEAN.replace("planting along the streambanks",
                                    "restoring the Waikato wetland complex"), tmp, lex)
        case("evidence catches invention", r.hard_in("evidence"))

        r = _run_text(CLEAN.replace("e1 | https://example.org/projects", "e1 | not-a-url"),
                      tmp, lex)
        case("evidence rejects non-URL", r.hard_in("evidence"))

        r = _run_text(CLEAN.replace("[e:1]", ""), tmp, lex)
        case("evidence flags unused quote",
             any(c == "evidence" and "never used" in m for _, c, m in r.items))

        # The tracker's own "Why It's a Fit" is legitimate evidence, but it is
        # not a loophole: it still has to appear verbatim in the fit.md that
        # queue.py wrote from the sheet.
        OWNER = CLEAN.replace(
            "You have riparian planting across forty\n  sites [e:1]",
            "You run a volunteer planting programme across the upper catchment [e:1]"
        ).replace(
            "e1 | https://example.org/projects | planting along the streambanks",
            "e1 | owner | tracker | volunteer planting programme across the upper catchment")
        r = _run_text(OWNER, tmp, lex)
        case("owner evidence passes from fit.md", not r.hard_in("evidence"),
             "; ".join(m for lv, c, m in r.items if c == "evidence")[:90])
        r = _run_text(OWNER.replace("volunteer planting programme across the upper catchment",
                                    "flagship programme across the Waikato", 2), tmp, lex)
        case("owner evidence catches invention", r.hard_in("evidence"))

        r = _run_text(CLEAN.replace("route: direct", "route: form"), tmp, lex)
        case("addressing rejects a form with an address", r.hard_in("addressing"))
        r = _run_text(CLEAN.replace("route: direct", "route: lookup"), tmp, lex)
        case("addressing blocks an unresolved lookup", r.hard_in("addressing"))
        r = _run_text(CLEAN.replace("route: direct\n", ""), tmp, lex)
        case("addressing needs a route", r.hard_in("addressing"))

        r = _run_text(CLEAN.replace("brackets: [2]", "brackets: [3]"), tmp, lex)
        case("classify catches mismatch", r.hard_in("classify"))

        (tmp / "recipients" / "fixture-org" / "classification.yaml").write_text(
            "primary: 2\nconfidence: unclassified\nweights:\n  b2: 0.2\n"
            "evidence:\n  - \"2 | https://example.org/projects | planting along the streambanks\"\n",
            encoding="utf-8")
        r = _run_text(CLEAN, tmp, lex)
        case("classify blocks unclassified", r.hard_in("classify"))
        _fixture_tree(tmp)

        r = _run_text(CLEAN.replace("brackets: [2]", "brackets: [2, 4, 5]"), tmp, lex)
        case("blend caps at two brackets", r.hard_in("blend"))

        r = _run_text(CLEAN.replace("I would like half an hour of your time",
                                    "give it a go, open an issue, and let me know"),
                      tmp, lex)
        case("blend catches four asks", r.hard_in("blend"))

        r = _run_text(CLEAN.replace("Kia ora Sam", "Kia ora {{first_name}}"), tmp, lex)
        case("placeholder catches mustache", r.hard_in("placeholder"))

        r = _run_text(CLEAN.replace("[f:TF-04]", "[f:TF-99]"), tmp, lex)
        case("facts catches unknown ID", r.hard_in("facts"))

        # The claim has to sit in its own sentence. Dropped into a sentence that
        # already carries an ID, the check is right to stay quiet - which is how
        # this case failed the first time it was written.
        r = _run_text(CLEAN.replace("If not, just say",
                                    "It is 40 percent faster than the alternatives.\n\n"
                                    "  If not, just say"),
                      tmp, lex)
        case("facts warns on uncited number",
             any(c == "facts" and lv == "WARN" for lv, c, _ in r.items))

        r = _run_text(CLEAN.replace("If not, just say and I will leave you alone.", ""),
                      tmp, lex)
        case("signoff needs an easy out", r.hard_in("signoff"))

        r = _run_text(CLEAN.replace("Liam Murphy\n", "Sent from my iPhone\n"), tmp, lex)
        case("signoff needs the sender", r.hard_in("signoff"))

        r = _run_text(CLEAN.replace("https://github.com/spatialLM/terrainflow",
                                    "https://bit.ly/tflow"), tmp, lex)
        case("signoff rejects a shortener", r.hard_in("signoff"))

        r = _run_text(CLEAN.replace("Kia ora Sam,", "Kia ora Sam, I hope this email finds you well."),
                      tmp, lex)
        case("lexicon catches an AI tell", r.hard_in("lexicon"))

        r = _run_text(CLEAN.replace("hear what it gets wrong.", "hear what it gets wrong!"),
                      tmp, lex)
        case("lexicon catches the exclamation", r.hard_in("lexicon"))

        (tmp / "suppression.txt").write_text("someone@example.org  # said no thanks\n",
                                             encoding="utf-8")
        r = _run_text(CLEAN, tmp, lex)
        case("suppression blocks a decline", r.hard_in("addressing"))
        (tmp / "suppression.txt").unlink()

        # The validation paragraph is the strongest material available and the
        # easiest to overstate by one word. Both directions matter: the earlier
        # fact bank forbade the claim outright, which would have blocked the
        # best thing in the email.
        VAL = CLEAN.replace(
            "shows where surface water concentrates on a site",
            "was tested against CAMELS-NZ [f:TF-30], on one catchment so far [f:TF-33]")
        r = _run_text(VAL, tmp, lex)
        case("validation passes when qualified", not r.hard_in("validation"),
             "; ".join(m for lv, c, m in r.items if c == "validation")[:90])
        r = _run_text(VAL.replace(", on one catchment so far [f:TF-33]", ""), tmp, lex)
        case("validation blocks unqualified claim", r.hard_in("validation"))
        r = _run_text(CLEAN.replace("I have built a QGIS plugin",
                                    "I have built a validated QGIS plugin that has been validated"),
                      tmp, lex)
        case("validation blocks overclaim", r.hard_in("validation"))
        r = _run_text(CLEAN.replace("a QGIS plugin", "a QGIS plugin checked against CAMELS-NZ"),
                      tmp, lex)
        case("validation warns on uncited CAMELS",
             any(c == "validation" and lv == "WARN" for lv, c, _ in r.items))

        # Drafting a batch before the video exists is the whole point of the
        # marker, so an unset URL warns rather than blocks - and --render is
        # where it becomes a refusal.
        r = _run_text(CLEAN, tmp, lex)
        case("video warns while unset",
             any(c == "video" and lv == "WARN" for lv, c, _ in r.items))
        r = _run_text(CLEAN.replace(" [video]", ""), tmp, lex)
        case("video warns when unlinked",
             any(c == "video" and "no [video] marker" in m for _, c, m in r.items))
        r = _run_text(CLEAN, tmp, lex,
                      cfg=dict(CFG, project=dict(CFG["project"],
                                                 video="https://youtu.be/abc123")))
        case("video quiet once set", not any(c == "video" for _, c, _ in r.items))

        # Markers are working notes and must never reach the recipient.
        _, sec = split_doc(CLEAN)
        rendered = render_body(sec)
        filled = render_body(sec, links={"video": "https://youtu.be/abc123"})
        case("render fills the video link",
             "https://youtu.be/abc123" in filled and "[video]" not in filled)
        # A stripped link marker leaves a sentence trailing into nothing, which
        # reads as a bug when proofreading. Unfilled markers stay visible.
        case("preview keeps unfilled link markers", "[video]" in rendered)
        case("render strips markers", "[e:1]" not in rendered and "[f:" not in rendered)
        case("render keeps the words", "streambanks" in rendered and "Liam Murphy" in rendered)
        case("render leaves no orphan space",
             not re.search(r"[ \t]+[.,;:?!]|^[.,;:?!]", rendered, re.M),
             repr(next((ln for ln in rendered.splitlines()
                        if re.search(r"[ \t]+[.,;:?!]|^[.,;:?!]", ln)), "")))

        # A template is not a person and must not be asked to be one.
        tpl = CLEAN.replace("---\nid: fixture-org", "---\nkind: template\nid: t2")
        r = _run_text(tpl, tmp, lex)
        case("template skips recipient checks",
             not r.hard_in("evidence") and not r.hard_in("classify"))

        # sameness: two texts that are the same text
        t = tmp / "templates"
        t.mkdir(exist_ok=True)
        long_body = " ".join("the water moves downhill across the paddock and pools".split() * 12)
        for name in ("a", "b"):
            (t / "{}.md".format(name)).write_text(
                "---\nkind: template\n---\n\nSUBJECT\n  x\n\nBODY\n  " + long_body + "\n",
                encoding="utf-8")
        r = Report()
        check_sameness(r, CFG, t)
        case("sameness spots a fake bracket",
             any(c == "sameness" and "share" in m for _, c, m in r.items))
        (t / "b.md").write_text(
            "---\nkind: template\n---\n\nSUBJECT\n  y\n\nBODY\n  "
            + " ".join("consent evidence for council stormwater assessment reports".split() * 12)
            + "\n", encoding="utf-8")
        r = Report()
        check_sameness(r, CFG, t)
        case("sameness quiet on real difference",
             not any(c == "sameness" and "share" in m for _, c, m in r.items))

    # The .gitignore block and PRIVATE_PATHS are duplicated on purpose. This is
    # the thing that notices when they drift apart.
    gi = REPO / ".gitignore"
    if gi.exists():
        text = gi.read_text(encoding="utf-8")
        missing = [p for p in PRIVATE_PATHS
                   if p not in text and p.rstrip("/") + "/" not in text]
        case("gitignore covers every private path", not missing, ", ".join(missing))
    else:
        case("gitignore covers every private path", False, ".gitignore not found")

    rp = Report()
    check_privacy(rp)
    case("privacy check runs against the real repo", not rp.hard_in("privacy"),
         "; ".join(m for lv, c, m in rp.items if c == "privacy")[:160])

    width = max(len(lbl) for lbl, _, _ in results)
    for lbl, passed, detail in results:
        line = "  {:<{w}}  {}".format(lbl, "PASS" if passed else "FAIL", w=width)
        if not passed and detail:
            line += "  <- " + detail
        print(line)
    print("\nselftest: {}".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ---------------------------------------------------------------- cli


def _resolve(arg: str) -> Path:
    p = Path(arg)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), REPO, ROOT):
        if (base / p).exists():
            return base / p
    return p


def main(argv: list) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    if argv[0] == "--selftest":
        return selftest()

    if argv[0] == "--privacy":
        rep = Report()
        check_privacy(rep)
        print("\nprivacy")
        print(rep.render())
        print()
        if rep.hard:
            print("BLOCKED - personal data is tracked by git. Fix before you push.")
            return 1
        print("OK - nothing naming a person is tracked.")
        return 0

    if argv[0] == "--templates":
        cfg = load_yaml(ROOT / "config.yaml")
        rep = Report()
        check_sameness(rep, cfg, ROOT / "templates")
        print("\ntemplates")
        print(rep.render())
        print()
        return 0

    render = argv[0] == "--render"
    args = [a for a in argv[1:] if a != "--wrap"] if render else argv
    wrap = "--wrap" in argv
    if not args:
        print("--render needs a draft path")
        return 2

    path = _resolve(args[0])
    if not path.exists():
        print("not found: {}".format(path))
        return 2

    rep = run(path)

    if render:
        if rep.hard:
            print("BLOCKED - {} hard finding(s). Nothing to copy until they are fixed.\n"
                  .format(rep.hard), file=sys.stderr)
            print(rep.render(), file=sys.stderr)
            return 1
        front, sections = split_doc(path.read_text(encoding="utf-8"))
        cfg = load_yaml(ROOT / "config.yaml")
        links = links_from_cfg(cfg)
        body = sections.get("BODY", "")
        unset = [n for n in LINK_MARKERS
                 if "[{}]".format(n) in body and VIDEO_UNSET.search(links[n])]
        if unset:
            print("BLOCKED - this draft links {} and config.yaml has no URL for {}.\n"
                  "          Set it there, then render. Never paste a URL into a draft."
                  .format(", ".join("[" + n + "]" for n in unset),
                          " / ".join("project." + n for n in unset)), file=sys.stderr)
            return 1
        sender = str((cfg.get("sender") or {}).get("email", ""))
        if VIDEO_UNSET.search(sender):
            print("BLOCKED - config.yaml sender.email is not set. Decide which address\n"
                  "          these go out from before rendering anything to send.",
                  file=sys.stderr)
            return 1
        route = str(front.get("route", ""))
        print("Route:   {}".format(route))
        print("{}  {}".format("Form:   " if route == "form" else "To:     ",
                              front.get("to", "")))
        print("Subject: {}".format(subject_of(sections)))
        print()
        print(render_body(sections, links=links, wrap=wrap))
        return 0

    print("\n{}".format(path.name))
    print(rep.render())
    print()
    if rep.hard:
        print("BLOCKED - {} hard finding(s). status stays 'draft'.".format(rep.hard))
        return 1
    warns = len(rep.items)
    tail = " ({} warning(s) to read first)".format(warns) if warns else ""
    print("OK to mark status: checked{}".format(tail))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
