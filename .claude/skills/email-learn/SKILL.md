---
name: email-learn
description: Diff drafted emails against what was actually sent, fold classifier corrections back into bracket signals, and propose cited rule changes for approval. Use after a batch goes out, when replies arrive, or to absorb the user's real sent emails.
---

# /email-learn

This is the part that compounds. Everything it proposes must cite the evidence
that produced it — a proposal without a citation is an opinion, and opinions
drift the rule pack instead of sharpening it.

## Read

`outreach/outbox/*/` (`.md` and `.sent.md` pairs) · `outreach/voice.md` ·
`outreach/lexicon.yaml` · `outreach/brackets/*.md` · `outreach/performance.md` ·
`outreach/corpus/` when run with `--corpus`

## Three inputs, three outputs

| Input | Updates | Because |
|---|---|---|
| Draft vs sent diff | `lexicon.yaml`, `voice.md` | The edits are corrections you already made |
| A bracket you changed in the scan table | that bracket's `Signals` block | The classifier was wrong in a specific way |
| Replies | `performance.md` | Which openers and brackets get answered |

## 1. Draft vs sent

Diff each `<slug>.md` against `<slug>.sent.md`. Every substitution is a
correction with evidence attached.

Propose in this shape:

> Add to `lexicon.yaml` `prefer:` — `"I have built" -> "I built"`.
> You made this change in 6 of 8 drafts in batch-01.
> Cited: `outbox/batch-01/{a,b,c,d,e,f}.sent.md`

Rules that a regex can enforce go in `lexicon.yaml`. Register and structure go in
`voice.md`. If you cannot express it mechanically, it belongs in `voice.md` — do
not force it into a pattern that will misfire.

**Thresholds.** One change is a one-off. Propose a rule at **three or more
occurrences of the same substitution**, and say the count every time. Below three,
list it as "worth watching" and hold it.

**Cuts count.** A sentence the user deleted from every draft is the strongest
signal in the batch and the easiest to miss — diff for deletions, not just
replacements.

## 2. Classifier corrections

When the user overrides a bracket in the scan table, work out *what in the site
should have said so*, and propose the addition:

> Add `"catchment management plan"` to bracket 3 `Signals: strong`.
> Example Engineering was classified 2 on "stream naturalisation"; you moved it
> to 3. The phrase that distinguishes them is on their services page.

If a correction is not explainable from the site text, say so plainly rather than
inventing a signal. Some corrections come from knowledge the user has and the
website does not contain — those improve nothing mechanically, and pretending
otherwise pollutes the signal lists.

If several `unclassified` organisations cluster around a common theme, propose a
sixth bracket with the quotes that justify it. Do not create it unasked.

## 3. Replies

Append to `performance.md`, keyed by bracket and opener mechanism:

```
batch-01  bracket 3  opener: early-stage-gap   sent 12  replied 3  calls 1
batch-01  bracket 5  opener: method-statement  sent  9  replied 4  calls 2
```

**Say the honesty caveat every time.** Under about 12 sends per opener mechanism
this is directional, not significant. Never recommend dropping a bracket on one
batch of evidence, and never present a reply rate as a finding when the
denominator is single digits.

Watch for the case that matters most: a bracket with a low reply rate and a high
call-acceptance rate among those who did reply. That is a bracket that is worth
more than it looks, and a raw reply rate will say the opposite.

## `--corpus`

The user's real sent emails in `outreach/corpus/` are the voice ground truth, and
they outrank everything derived. Read them against `voice.md` and answer the open
questions listed at the bottom of that file — greeting, sign-off, contractions,
first person — with the lines that evidence each answer.

Then replace the seeded hypotheses in `voice.md` with what the corpus shows. Say
explicitly which seeded rules the corpus contradicted; those are the ones that
were quietly making every draft slightly wrong.

## The approval rule

**The user approves every proposed rule.** Never write to `voice.md`,
`lexicon.yaml` or `brackets/*.md` unasked. Those files are the authority, and an
authority that changes silently drifts.

Present proposals as a numbered list with the citation on each, and let the user
accept by number.

## Never

- Never propose a rule without the count and the files that evidence it.
- Never infer a voice rule from a single email.
- Never fold a generated draft into `corpus/` — that pulls the corpus toward the
  model's mean instead of the user's. Only genuinely sent, human-written mail
  goes in there, and a `.sent.md` only qualifies if the user rewrote it
  substantially.
- Never present performance numbers without the sample size next to them.
