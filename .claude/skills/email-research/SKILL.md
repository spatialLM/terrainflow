---
name: email-research
description: Find the one verifiable specific each tailored opener needs, resolve addresses for contacts that have none, and confirm the tracker's own fit judgement. Use after importing the tracker, or when the user gives an org URL.
---

# /email-research

**The tracker already did the hard part.** `TerrainFlow_Outreach_Tracker.xlsx`
carries a Category, a Region and a hand-written *Why It's a Fit* for all 51
contacts. `queue.py --import-xlsx` files that as `fit.md` in each recipient
folder. Do not re-derive any of it — the judgement in that sheet is better than
anything signal-matching produces.

Three jobs remain.

## Read

`outreach/guardrails.md` · `outreach/config.yaml` · the recipient's
`record.yaml` and `fit.md` · the relevant `outreach/brackets/*.md`

## Job 1 — one verifiable specific for the opener

The base email's own instruction is the standard:

> *[TAILORED OPENER — 1–2 sentences. A specific, verifiable reference to their
> project, region or recent work… Name a fact only they'd recognise. If you
> can't get that specific, hold this email for a warmer contact.]*

WebFetch the URL in `record.yaml`. Home page first, then whichever of
about / projects / work / research / team looks likeliest. Two or three pages is
plenty. Write verbatim quotes with their source URLs to
`recipients/<slug>/research.md`:

```markdown
# <Org name>
fetched: <YYYY-MM-DD>

## https://example.org/projects
> "verbatim sentence or phrase from the page"

## Not found
- no mention of earthworks, swales or water retention on the pages read
```

**The `Not found` section matters as much as the quotes.** It is what stops the
next step reaching for something that is not there.

Then say plainly whether there is enough for a specific opener. If there is not,
**say so and recommend holding the contact** rather than writing a vague one.
That is Liam's own rule and it is not yours to soften.

## Job 2 — resolve the addresses that are missing

Six contacts have `route: lookup` — named academics whose address the tracker
says to find on a faculty page. They are bracket 5, the people the CAMELS-NZ work
was done for, so they are worth the effort.

Fetch the university staff or group page. If a published address is there, record
it in `record.yaml` as `email:` with the page URL in `notes:`, and change `route:`
to `direct`.

**Never guess an address.** Not `first.last@institution.edu`, not a pattern
inferred from a colleague's address, not one from a contact-lookup service. If no
address is published, leave `route: lookup` and report it as unresolved — the
gate hard-fails a `lookup` draft, which is what stops an unresolved contact
reaching a batch.

The same rule already applies to two tracker entries: Geoff Lawton's personal
address came from a lookup service and must not be used, and Brad Lancaster's
site says he rarely answers unsolicited mail.

## Job 3 — confirm or challenge the fit

Read the tracker's *Why It's a Fit* against what the site actually says. Usually
it will be right. When it is not — the organisation has changed focus, the
programme named has ended — **say so**, and say what you saw. Do not silently
re-file them.

If the bracket looks wrong, propose the change with the quote behind it. The
bracket is in `record.yaml`; `/email-learn` folds accepted corrections back into
the bracket `Signals` blocks.

## Evidence provenance

Two kinds, both verbatim-checked by the gate, against different files:

| | From | Checked against | Written as |
|---|---|---|---|
| `web` | a page you fetched | `research.md` | `e1 \| web \| https://… \| quote` |
| `owner` | the tracker's fit or notes | `fit.md` | `e2 \| owner \| tracker \| quote` |

`owner` is not a loophole. It still has to appear verbatim in the `fit.md` that
`queue.py` wrote from the sheet, so it cannot be invented either — it simply
credits Liam's own knowledge rather than pretending it came from a scrape.

## --batch

Over every contact in state `new`. Print one scan table, then set each to
`researched`:

```
org                              bracket  route    opener
Example Restoration Trust        2        direct   "planting across forty sites"
Example University               5        direct   resolved: staff page
Example Institute                1        form     <- nothing specific found
Dr Example                       5        lookup   <- no published address
```

Then say plainly, not buried under the table: how many have no specific opener,
and how many lookups stayed unresolved. Those are the user's call.

## Never

- Never write a quote you did not fetch, in either file.
- Never guess an email address, in any form.
- Never invent a specific because the fit note says one should exist.
- Never re-derive a bracket the tracker already assigned without saying why.
- Never record a person's biography beyond what is needed to address them and
  write one accurate sentence. You are researching an organisation's work.
