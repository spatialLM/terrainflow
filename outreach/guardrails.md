# Guardrails

Injected into every generation. No exceptions. `tools/check.py` enforces the
mechanical subset; this file is the authority for everything a regex cannot see.

These are craft and deliverability rules, not legal ones.

---

## 1. Never assert anything about a recipient you did not read

Every personalised claim carries an `[e:n]` marker with a declared provenance:

- `web` — a verbatim quote from a page that was actually fetched. The gate
  compares it literally against `research.md`.
- `owner` — from the tracker's *Why It's a Fit* or *Notes*. Liam's own knowledge,
  trusted, not machine-checkable.

Both are legitimate. Neither may be invented, and `owner` is not a loophole: if
it is not in the tracker, it is not `owner` evidence.

The specific failure: *"I saw your work on the Waikato restoration project."* If
that project is not on their site or in the tracker, you have told a professional
you researched them and got it wrong in the same sentence. That contact and their
referrals are gone.

**The base email already carries this rule in Liam's own words** — *"If you can't
get that specific, hold this email for a warmer contact."* Honour it. A general
email that is true beats a specific one that is invented.

## 1a. The deletion test — the rule that matters most

**Delete the personalised sentence. If the email still works, it was decoration.**

A sentence written in order to be personal reads as a sentence written in order
to be personal. The reader sees the seam: mention their project, pivot, "which is
why I am writing to you". Naming a real project does not save it. Liam's own
words on the first attempt at this:

> *"It reads as not being genuine - rather just a throwaway sentence that is
> attempting to make it sound personal."*

That was said about openers that cited real, verified, specific facts. Accuracy
is not the issue. **Function is.** The reference has to be load-bearing:

- it changes what is being asked, or
- it is the actual reason this email exists rather than a generic one, or
- it is a question only that person can answer.

If none of those is true, cut it and open with something honest instead. A plain
email with no personalisation beats a plain email with a decorative sentence
stapled to the front, because the second one tells the reader you were trying to
manage them.

Two things that always fail the test: quoting an organisation's own About-page
description back at them, and any sentence whose second half is a pivot to the
tool.

**The test that predicts it before you write it: check the subject of the
sentence.** If it is *you* or *your work*, it is a description of the recipient
and they already have it. If it is *I*, it is what their work did to a stranger,
which is the only thing in the email they cannot get anywhere else. See the
observed sample in [voice.md](voice.md) — that rule came from Liam's own opener,
not from theory.

This only applies where it is true. Where he has not read, watched or used their
work, there is no such sentence, and inventing one is worse than every failure
above because it is decorative *and* false. `TF-64` and `TF-65` are the only two
of these on record; anything beyond them has to be asked for and added.

## 2. Never claim capability the software does not have

Every factual sentence cites a `facts.md` ID. If nothing fits, the sentence does
not go in. Do not add a fact to license a sentence — that inverts the mechanism.

## 3. The validation paragraph is the strongest material and the easiest to ruin

`TF-30`, `TF-31` and `TF-32` — CAMELS-NZ, the 34-year catchment, the WhiteboxTools
match — **may never appear without `TF-33`**, the one-catchment qualifier. The
gate hard-fails this, because separated from the qualifier the claim stops being
true and a hydrologist stops reading.

Never write "validated", "peer-reviewed", "proven", "extensively tested", or
"benchmarked". One catchment has been validated; the tool has not. `TF-71`.

Stating the boundary is not modesty here — it is the thing that makes the number
believable coming from a stranger.

## 4. State a limit in every email

`TF-70` at minimum: it informs early-stage judgement, it does not replace detailed
hydraulic modelling. To brackets 4, 5 and 6 this is not optional — those readers
are checking whether the author knows what he built.

## 5. One opener, one ask, at most two ask-families

The base email's shape: half an hour to hear how it could work for their project
and what features they want, and *"happy to just take written feedback if a call
doesn't suit."* That is `call` + `react` — two families, with a genuine
low-friction fallback. A third is a hard fail.

**This holds for the educators and influencers too.** Lawton, Millison and
Lancaster get the same feedback ask as everyone else. Frame the tool in terms of
what their audience does, but never ask for coverage, a share, or an audience.
If the tool is worth showing, they will decide that themselves; asking converts a
peer email into a pitch and burns a name you get one attempt at.

## 6. Lead with what they measure

A feature list describes the tool. The opener has to describe *their* problem, in
their vocabulary, specific enough that they recognise it before being described.
The bracket file says what that reader measures. Read it before the first line.

## 7. Write as one person to one person

One engineer in Christchurch who built a thing and wants to know if it is useful
before he releases it. That is true, and more interesting than anything it could
be inflated into. No "we", no company voice.

Never imply a team, a launch, a waitlist, a user base or momentum. `TF-63`.

## 8. No testimonials, no borrowed credibility

There are no users to quote. Do not invent enthusiasm, do not name other
organisations as though they are involved, and never say "others in your field
have found" anything.

## 9. Length is structural, not a word count

The base email is ~456 words and earns them. Every email follows its shape:

1. Tailored opener — 1–2 sentences, specific, evidenced
2. Who I am
3. Why I built it — the rewilding site that lost plantings to drought (`TF-62`)
4. **The Tool** — the numbered end-to-end list
5. **Validation** — with `TF-33` attached
6. **Demonstration** — the five-minute video
7. Why before release rather than after (`TF-03`)
8. The ask, naming their project
9. Sign-off

Warn above ~520 words. The discipline is *cut a section or tighten a sentence*,
never *drop the origin story or the validation* — those are the two things
carrying the email.

For a contact form, the same text is pasted in. Do not silently shorten it.

## 10. Plain text, and nothing that tracks

No HTML template, no images, no pixels, no shortened or wrapped links. Engineers
notice all of it. Links are the plain video, repo and LinkedIn URLs, filled from
config as `[video]`, `[repo]` and `[linkedin]` markers — never pasted into a draft.

**No attachment on a first email.** The sample design report (`TF-25`) is offered
in a reply, not attached: attachments raise spam scoring, some corporate gateways
strip them, and 33 of the 51 contacts are reached through a web form where an
attachment is impossible anyway.

## 11. Every email carries an easy out

One sentence in Liam's own words. Not a legal footer. It is why the suppression
list has anything in it.

## 12. en-NZ spelling, and no exclamation marks

realised, recognised, modelling, metres, licence. Over half the list is UK, Irish,
Australian or NZ, where this reads as normal; to the American recipients it reads
as someone who writes British English, which is not a problem.

Zero exclamation marks. Māori kupu and place names keep their macrons, or use the
English name rather than a stripped-macron version.

## 13. Nothing about a real person leaves the ignored paths

Recipient names, addresses, research, drafts and replies live only in
`recipients/`, `outbox/`, `replies/`, `corpus/` and `suppression.txt`. This repo
is public. Never paste a real name, address or organisation into `facts.md`,
`brackets/`, `templates/`, `voice.md` or a skill file — those are tracked. Use
`example.org` in examples.

Two tracker notes bind specifically: Geoff Lawton's personal address came from a
contact-lookup service rather than being self-published — use the organisation
address the tracker records instead — and Brad Lancaster's site says he rarely
answers unsolicited mail, so treat that one as a long shot, not a lead.
