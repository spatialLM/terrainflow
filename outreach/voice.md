# Voice

**This is a seed, and it is the weakest file in the repo.**

It was built from one source: the way you write in
[README.md](../README.md) and the project `CLAUDE.md` files. That is your
*documentation* register — considered, technical, written for no one in
particular. A cold email is a different act, and the gap between how someone
writes docs and how they write to a stranger is wider than it looks.

**The fix is not to describe your voice better. It is `outreach/corpus/`.** Drop
10–15 emails you have actually sent — to colleagues, to suppliers, anything that
sounds like you — into that folder (it is gitignored, nothing leaves your
machine). Then run `/email-learn --corpus`. Every rule below that survives
contact with real evidence stays; the rest gets replaced by something cited.

Until that happens, treat everything here as a hypothesis.

---

## What the documentation register already shows

These are observed in your own prose, with the line that evidences each.

**You state the limit in the same breath as the claim.**
> "It is not intended to replace detailed hydraulic modelling. It is designed to
> inform early-stage engineering judgement with real spatial data."

Two sentences, the second doing the work of the first. This is the single most
distinctive thing about how you write and it is worth protecting — most people
put the caveat at the end where it reads as a disclaimer. You put it adjacent,
where it reads as precision.

**You give the reason immediately after the fact, in the same sentence.**
> "D-infinity distributes flow fractionally between the two steepest downslope
> cells, producing smoother accumulation patterns on gentle terrain."

Never a bare capability. Always the mechanism or the consequence attached.

**You set up the absence before the answer.**
> "…for engineers and practitioners working at site or farm scale … there is a
> gap. TerrainFlow fills that gap."

Problem, then tool. Not tool, then use case.

**Plain declaratives, present tense, no hedging adverbs.** "Load a DEM and
TerrainFlow runs a complete hydrological terrain analysis." No "simply", no
"easily", no "powerful".

**Zero exclamation marks across the entire README.** Keep it at zero.

**en-NZ throughout**: licence, artefacts, modelling, metres, realised.

## Carried over as email rules

- **Short first sentence.** Under 18 words. It is read in a preview pane.
- **Contractions are fine in email, absent in docs.** "I have built" and "I've
  built" are both you; the email is the one that gets the contraction. Low
  confidence — this is a guess until the corpus says otherwise.
- **No greeting flourish.** "Kia ora <name>," then straight into it. Nothing
  between the greeting and the first real sentence.
- **Sign off with your name, role and the repo link.** Not a signature block with
  a logo. You are one person who built a thing.

## Open questions the corpus will answer

Recorded so `/email-learn` knows what to look for, not so you answer them here:

1. Kia ora, Hi, or Hello — and does it change by bracket?
2. Do you use em-dashes in email the way you do in prose, or is that a docs
   habit?
3. First person singular throughout, or do you slip into "we" about the project?
4. How do you close? "Cheers", "Thanks", "Ngā mihi", nothing?
5. Do you ask a direct question, or state the ask and let it sit?

## Never, in any register

The banned list in [lexicon.yaml](lexicon.yaml) is enforced. The ones worth
naming here because they are what a model reaches for by default:

- "I hope this email finds you well" — the single clearest tell.
- "reaching out", "circle back", "touch base".
- "I'd love to pick your brain."
- Any sentence whose subject is the software and whose verb is a benefit.
