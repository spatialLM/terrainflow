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

## Observed from your own writing to a person — 2026-09-01

**This section outranks everything above it.** The rest of this file is inferred
from documentation; this is one paragraph you actually wrote to a named human,
supplied after two rounds of drafts were rejected. One real sample beats every
hypothesis it contradicts.

The opener, verbatim:

> *"I've been following your videos for a while now along with other permaculture
> educators (thank you for the great work you do). Your videos were a big part of
> what inspired me to complete Geoff Lawtons permaculture course. Using my
> professional and academic experience I've been developing a tool that I hope
> can really improve the accuracy and speed which permaculture and water
> harvesting projects can be planned, designed and developed."*

### The rule this settles

**Write what their work did to you, not what their work is.**

Every rejected opener described the recipient: their layer ordering, their lot
size, their demonstration site. All accurate, all sourced, all rejected — because
a description of someone's own work, returned to them, carries no information
they do not have. What they cannot know is its effect on a stranger.

This is why it passes the deletion test where the others failed. Delete
*"your videos were a big part of what inspired me to complete the PDC"* and the
email loses its reason to be addressed to Millison rather than to anyone. It is
also a credential — it says *I am not a random engineer emailing about permaculture*
— without a sentence spent claiming one.

The mechanical difference: the rejected openers had the recipient as the subject.
This one has **I** as the subject and their work as the object. When the personal
sentence starts with *"You"* or *"Your work"*, it is about to fail.

### Where it does not reach

It requires having actually consumed their work. That is true for the educators
and authors on the list and false for most of the organisations. **Do not
manufacture it** — an invented *"your work inspired me"* is worse than every
rejected draft, because it is both decorative and false.

Where it is not true, there is no opener. Open with the rewilding site
(`TF-62`) and say plainly why this organisation. That was already the approach
for those, and it stands.

### Sentence-level, observed

- **Contractions throughout.** "I've", not "I have". The guess above was right;
  it is now evidence.
- **Warmth in a parenthesis**, not in a clause. *"(thank you for the great work
  you do)"* — asides carry the feeling, the main clause stays flat.
- **No pivot.** Three sentences: what their work did, what it led to, what he is
  building. The third does not double back to justify the first. Nothing in this
  paragraph exists to set anything else up, which is precisely why it does not
  read as constructed.
- **Everything specific is checkable and his**: the videos, the course, the
  developing. Nothing asserted about the recipient at all.
- **He does not name the tool in the opener.** "a tool that I hope can..." —
  hedged, unnamed, deferred to the paragraph that follows.
- **"a tool that I hope can really improve"** — he hedges his own work. Do not
  strip that into a claim; "which improves" is a different person writing.
- 3 sentences, 22/16/33 words. Longer than the 18-word rule above. That rule was
  a guess from the README and is now downgraded: the first sentence is 22 words
  and it works.

### Two edits worth making, and why they are only suggestions

- *"along with other permaculture educators"* dilutes the one sentence doing the
  work — it says *you are one of several I follow*. Cutting four words makes it
  land harder and costs nothing true.
- *"Using my professional and academic experience"* restates the paragraph
  immediately below it, which opens *"I'm a civil and environmental engineer…"*.
  The opener can hand straight over.

Neither is a voice correction. Do not touch the grammar of
*"the accuracy and speed which … can be planned"* or the missing apostrophe in
*"Geoff Lawtons"* on any pass that is not proofreading — flag them once at proof
time and let him decide. Smoothing a person's sentences into standard register is
how the last two rounds ended up sounding like nobody.

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
