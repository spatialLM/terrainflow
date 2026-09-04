---
name: email-reply
description: Classify a reply to an outreach email and draft a response against the same voice, fact and evidence rules. Use when the user pastes a reply they have received.
---

# /email-reply

The point of the outreach is the conversation, and the conversation is where the
admin actually piles up. Same rule pack, same gate, shorter emails.

## Read

`outreach/guardrails.md` · `outreach/voice.md` · `outreach/facts.md` ·
the original draft in `outreach/outbox/<batch>/<slug>.sent.md` ·
that recipient's `research.md` and `classification.yaml`

## 1. Classify the reply

| | Means | Response |
|---|---|---|
| `interested` | Positive, no specific ask | Short. Offer two concrete times. |
| `call-request` | They want to talk | Confirm, offer times, say what you will show. Nothing else. |
| `question` | Technical or practical | Answer it. Cite facts. Say the limit. |
| `objection` | Doubt about method or value | **The most valuable reply you will get.** |
| `not-now` | Interested, wrong timing | One line. Ask if you may come back, name when. |
| `declined` | No | One line of thanks. Then suppress. |
| `bounce` | Bad address | No reply. Suppress. |

## 2. Write to `outreach/replies/<slug>/<YYYY-MM-DD>.md`

Store the received text verbatim, then your draft response below it. Everything
under `replies/` is gitignored.

## 3. The rules that still apply

A reply may not claim anything the first email could not:

- Every factual sentence cites `[f:TF-nn]`. A question is not a licence to
  overclaim, and an enthusiastic reply is exactly where it happens.
- `TF-74` is never softened. If they ask how it compares to a calibrated model,
  the answer is that it has not been benchmarked, and you would welcome help
  doing it.
- Never invent a user, a result, a timeline or a roadmap commitment to keep
  momentum in a thread.
- No new capability claims. If they ask for something it does not do, say it does
  not do it. That answer keeps more conversations alive than a hedge does.

The gate does not run on replies — they are conversational and the draft format
does not apply. That means these rules are on you rather than on `check.py`, so
state which facts you cited when you present the draft.

## 4. Objections get the most care

An objection is someone taking the work seriously enough to argue with it. Do not
defend. Answer the technical point, concede what is true, and if they are right
about a gap, say so and ask if they would look at the fix.

An objection from a bracket 3 or 5 reader that you answer well is worth more than
ten polite yeses.

## 5. Housekeeping

```powershell
python outreach\tools\queue.py --decline <slug>    # declined or bounced
```

That writes to `suppression.txt`, and the gate blocks any future draft to that
address — so nobody hears from you twice after saying no.

For a reply, set the record state to `replied` and note it for `/email-learn`,
which reads reply outcomes into `performance.md`.

## Never

- Never send. There is no mail connector wired in; the user pastes.
- Never chase. If someone does not reply, that is an answer.
- Never reply to a `declined` message with anything except one line of thanks.
- Never let thread length erode the rules. The fifth email in a thread is where
  an unsupported claim slips in.
