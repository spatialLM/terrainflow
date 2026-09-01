# outreach/ — cold email for the TerrainFlow intro video

One base email, six bracket families, 51 contacts from
`TerrainFlow_Outreach_Tracker.xlsx`. Every contact gets the full researched
email — there is no short version and no depth tier, because the tracker already
carries a hand-written fit for all of them.

## THE THREE RULES THAT MATTER

**1. This repo is public. Nothing naming a real person may be tracked.**

`recipients.csv`, `recipients/`, `outbox/`, `replies/`, `corpus/` and
`suppression.txt` are gitignored **and** enforced — `tools/check.py` runs
`git ls-files` over them and hard-fails if anything is tracked. Never paste a real
name, address or organisation into any other file here; they are all tracked. Use
`example.org` in examples.

**2. Nothing may be asserted about a recipient that was not read.**

Every personalised claim carries `[e:n]` with a provenance, both verbatim-checked:

- `web` — a quote from a fetched page → checked against `research.md`
- `owner` — a line from the tracker → checked against `fit.md`

`owner` is not a loophole; it still has to be verbatim in the file `queue.py`
wrote from the sheet.

**3. The validation claim never travels without its qualifier.**

`TF-30`/`TF-31`/`TF-32` — CAMELS-NZ, the 34-year catchment, the WhiteboxTools
match — may not appear without `TF-33`, "one catchment so far". Hard fail. Never
write "validated", "proven", "peer-reviewed" or "benchmarked" (`TF-71`).

[facts.md](facts.md) was rebuilt from the base email, which supersedes the repo
README — the README still lists validation as a roadmap item and is out of date.

## The commands

| Command | Does |
|---|---|
| `/email-base` | Ingest and interrogate a base email |
| `/email-research` | One verifiable specific per opener; resolve `lookup` addresses |
| `/email-variant` | Base + bracket + evidence → the full draft |
| `/email-proof` | Gate, review sheet, record what was sent |
| `/email-learn` | Diff draft vs sent → cited rule proposals |
| `/email-reply` | Classify a reply and draft a response |

## Always read before generating

[guardrails.md](guardrails.md) · [voice.md](voice.md) · [facts.md](facts.md) ·
[lexicon.yaml](lexicon.yaml) · the relevant `brackets/*.md` ·
**[base/2026-09-01-intro-video.md](base/2026-09-01-intro-video.md)** — the base
email is the structure; do not redesign it.

## Routes — how each contact is reached

18 `direct`, 24 `form`, 6 `lookup`, 3 `warm`. Declared in frontmatter and checked:
a form URL can never be mistaken for a sendable address, and a `lookup` with no
resolved address is hard-failed so it cannot reach a batch.

## Links are markers, never URLs

`[video]`, `[repo]`, `[linkedin]` are filled from `config.yaml` at render time.
Fifty-one drafts is fifty-one find-and-replaces and one gets missed. `--render`
refuses while any of them, or `sender.email`, is unset.

## Tools

```powershell
python outreach\tools\check.py --selftest              # 39 cases, both directions
python outreach\tools\check.py --privacy               # before any push
python outreach\tools\queue.py --import-xlsx <tracker.xlsx>
python outreach\tools\queue.py --status
python outreach\tools\check.py <draft>
python outreach\tools\check.py --render <draft>        # send-ready, unwrapped
python outreach\tools\sheet.py <batch> --open          # the proofreading page
```

`--render` output is deliberately **unwrapped** — one line per paragraph. Gmail
treats every newline as a real break, so hard-wrapped text pastes as a ragged
column. It is also the only way to get copyable text, and it refuses on a hard
finding.

## Rules

- Structure follows the base email: opener, who I am, why I built it, The Tool,
  Validation, Demonstration, before-release, the ask, sign-off. ~450–500 words.
- `TF-11` (LINZ) only for bracket 3. Everyone else gets `TF-10` — open elevation
  data anywhere. 45 of 51 contacts are outside New Zealand.
- State a limit in every email. `TF-70` minimum; brackets 4 and 5 also `TF-71`.
- Two ask-families: `call` + `react`. Never ask an educator for coverage or reach.
- **No attachment.** The design report is offered in a reply.
- Plain text, no HTML, no tracking or shortened links.
- en-NZ spelling. Zero exclamation marks.
- If there is no specific opener, hold the contact. That is the base email's own
  instruction and it is not ours to soften.
