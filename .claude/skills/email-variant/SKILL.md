---
name: email-variant
description: Write the per-contact drafts from the base email, using each bracket's facts and order, with a verified tailored opener. Use after research and classification.
---

# /email-variant

Every contact gets the full email. There is no short version and no A/B/C depth
tier — the tracker carries a hand-written fit for all 51, so all 51 are
researched contacts and each one gets a real email.

## Read

`outreach/guardrails.md` · `outreach/voice.md` · `outreach/facts.md` ·
`outreach/lexicon.yaml` · **`outreach/base/2026-09-01-intro-video.md`** ·
the contact's `record.yaml`, `fit.md` and `research.md` · their `brackets/*.md`

The base email is the structure. Do not redesign it.

## The structure — every email, in this order

1. **Tailored opener** — 1–2 sentences, specific, every claim carrying `[e:n]`
2. **Who I am** — `TF-60`
3. **Why I built it** — the rewilding site that lost plantings to drought,
   `TF-62`. This is the strongest paragraph in the email. Keep it.
4. **The Tool** — the numbered end-to-end list, ordered by what *this bracket*
   cares about
5. **Validation** — `TF-30`/`TF-31`, and `TF-33` in the same breath, always
6. **Demonstration** — the five-minute video, `[video]`. **No attachment.**
7. **Why before release rather than after** — `TF-03`
8. **The ask** — half an hour, naming their project, plus the written-feedback
   fallback
9. **Sign-off** — name, role, `[repo]` · `[linkedin]`

What varies by bracket: the opener, the order of The Tool list, which facts lead,
which limit is stated, and the wording of the ask. What does not vary: the
structure, the origin story, the validation qualifier.

Target ~450–500 words, the length of the base. The gate warns above 520.

## Per-bracket emphasis

| Bracket | Leads with | Limit to state |
|---|---|---|
| 1 Permaculture | `TF-62` origin, then `TF-27` keyline, `TF-23` cascade | `TF-70` |
| 2 Restoration NGOs | `TF-62` origin, then `TF-10` open data anywhere | `TF-70`, `TF-33` |
| 3 NZ land & freshwater | `TF-11` LINZ, `TF-30`–`TF-33` a NZ dataset | `TF-70`, `TF-72` |
| 4 NbS / flood engineering | `TF-70` the limit *first*, then validation, then method | `TF-70` **and** `TF-71` |
| 5 Swale / NFM academics | `TF-30`, `TF-31`, `TF-33` — validation is the opener | `TF-71`, `TF-74` |
| 6 Open-source geospatial | `TF-02` AGPL, then `TF-50`, `TF-51`, `TF-52` | `TF-71`, `TF-72` |

**`TF-11` (LINZ) appears only for bracket 3.** For everyone else `TF-10` — public
LiDAR where it exists, coarser open elevation data anywhere it does not — is the
fact that makes the tool relevant at all. Forty-five of fifty-one contacts are
outside New Zealand.

## The draft file

```
---
id: <slug>
org: <Org name>
to: <address, or the form URL>
route: direct | form | warm
priority: <tracker Tier 1-5>
brackets: [2]
batch: batch-01
status: draft
---

SUBJECT
  Open-source QGIS tool for water-harvesting earthworks — feedback before release

BODY
  Hi <name>,

  <tailored opener with [e:n]>
  ...
  Thanks,
  Liam Murphy
  Civil & Environmental Engineer — Christchurch, NZ
  [repo] · [linkedin]

EVIDENCE
  e1 | web   | https://example.org/projects | verbatim quote from research.md
  e2 | owner | tracker                      | verbatim line from fit.md
```

`route: lookup` is never drafted — the gate hard-fails it. Resolve the address
first or leave the contact out of the batch.

## Hard rules

- **Links are markers.** `[video]`, `[repo]`, `[linkedin]` — never a pasted URL.
  Fifty-one drafts is fifty-one find-and-replaces and one gets missed.
- **`TF-30`/`TF-31`/`TF-32` never appear without `TF-33`.** Hard fail. The
  one-catchment qualifier is what makes the claim true, and to brackets 4 and 5
  it is the most credible sentence in the email.
- **Never write "validated", "proven", "peer-reviewed", "benchmarked".** `TF-71`.
- **Every factual sentence cites `[f:TF-nn]`.** If nothing fits, cut it.
- **Every personalised claim cites `[e:n]`**, verbatim from `research.md` (`web`)
  or `fit.md` (`owner`). Never paraphrase into the evidence block — the gate
  compares literally, and the paraphrase is where invention creeps in.
- **No attachment.** The design report `TF-25` is offered in a reply.
- **Two ask-families**: `call` + `react`. Never ask an educator for coverage,
  a share, or their audience — see `guardrails.md` rule 5.
- Subject under 9 words. Zero exclamation marks. en-NZ spelling.

## If there is no specific opener

`/email-research` will have said so. **Do not write a vague one.** Recommend
holding the contact — that is the base email's own instruction. A general email
that is honest beats a specific one that is hollow, and both beat an invented one.

## Then

Run the gate on every draft and report findings **grouped by check, not by file**.
Fifty drafts with the same finding is one problem — fix it once and regenerate.

Offer `/email-proof <batch>`.

## Never

- Never write a specific about an organisation that is not in `research.md` or
  `fit.md`.
- Never draft for a `lookup` contact, or one on the suppression list.
- Never shorten the email for a contact-form recipient without being asked. The
  decision was: same email, pasted in.
- Never let one bracket's phrasing drift into another because it read well.
