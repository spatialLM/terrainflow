---
name: email-base
description: Ingest the user's base email for the TerrainFlow intro video and interrogate it before anything is built on top of it. Use when the user pastes their base email, or wants the base rewritten or re-examined.
---

# /email-base

Everything downstream inherits from this file. A weak base produces five weak
templates and a hundred weak emails, and no amount of personalisation rescues it.
So the job here is not to be encouraging.

## Read

`outreach/guardrails.md` · `outreach/voice.md` · `outreach/facts.md` ·
`outreach/lexicon.yaml` · `outreach/config.yaml` · all five `outreach/brackets/*.md`

## What comes in

The user pastes their email. Save it verbatim first, before any commentary, to
`outreach/base/<YYYY-MM-DD>-<slug>.md` with `kind: base` in the frontmatter.
Verbatim matters: `/email-learn` later diffs against it, and an already-improved
"original" destroys that signal.

## The interrogation

Work through these in order and show your reasoning. Be direct. A polite critique
that changes nothing is worse than none.

**1. Is the first sentence about them or about the tool?**
Almost every base email opens on the tool. It reads as a press release and the
reader has decided by the end of it. The opening line has to describe a problem
the reader already has, in their words.

**2. Does the subject line survive a phone notification?**
Under 9 words, no colon-clause construction, and it has to make sense with no
sender context. Read it aloud as though it appeared on a lock screen.

**3. What exactly is the ask, and how many are there?**
Count the ask-families (`lexicon.yaml: cta_families`). The intended shape is two:
react, and talk for half an hour. Three or more and the gate blocks it. Also
check the ask is *decidable* — "let me know if you'd be interested in exploring
a collaboration" makes the reader do the work of guessing what you want.

**4. Which limit does it state?**
If none, it fails. `TF-70` at minimum. To an engineer the caveat is the
credential — an unqualified claim from a stranger reads as marketing.

**5. Which facts does it claim, and do they all have IDs?**
Every factual sentence maps to a `facts.md` ID. Name the ones that do not.
**If a sentence has no fact behind it, the fix is to cut the sentence, not to add
a fact.** Adding a fact to license a sentence inverts the whole mechanism.

**6. What survives translation to all five brackets?**
Read it once as each bracket's reader. The parts that only land for one bracket
belong in that template, not in the base. The base is what is true regardless of
who is reading.

**7. Would you send this to someone whose opinion you actually want?**
Ask the user directly. It is the only test that catches the failure where every
individual sentence is fine and the whole thing is forgettable.

**8. Word count.**
Under 180. If it will not fit, it is doing two jobs — say which one to cut.

## Then

- Show the specific rewrite for each finding, not a general note. "The opening is
  tool-first" is useless; the replacement line is the work.
- Run `python outreach/tools/check.py outreach/base/<file>.md` and report it.
  The base carries `kind: template` for checking purposes — it has no recipient.
- Offer `/email-variant --templates` to build the five bracket templates.

## Never

- Never rewrite the base silently. Show the diff and let the user choose. The
  rough version usually contains the one specific true detail that makes it good,
  and a rewrite is exactly where that gets smoothed away.
- Never add enthusiasm the user did not write. If it reads flat to you and true
  to them, it is true.
- Never invent a fact, a user, a result or a deadline to strengthen it.
