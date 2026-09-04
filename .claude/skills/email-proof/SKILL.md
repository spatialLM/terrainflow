---
name: email-proof
description: Run the gate over a batch, present drafts for proofreading as diffs by tier, and record what was actually sent. Use before sending, or when the user says they are ready to proofread.
---

# /email-proof

The bottleneck in this whole pipeline is the user's eyes, not the generation.
Everything here exists to reduce what they have to read without reducing what
they control.

## Read

The batch's drafts in `outreach/outbox/<batch>/` · the matching
`outreach/templates/*.md` · `outreach/recipients/<slug>/record.yaml`

## 1. Gate first

```powershell
python outreach\tools\check.py outreach\outbox\<batch>\<slug>.md
```

Run it over every draft. **Present findings grouped by check, not by file.**
Forty drafts sharing one finding is one problem. Fix it in the template and
regenerate rather than fixing forty files.

Nothing is shown for proofreading until it passes. A hard finding means the user
would be proofreading text they cannot send.

## 2. Build the review sheet

```powershell
python outreach\tools\sheet.py <batch> --open
```

One local HTML page — every draft grouped by tier, with its gate findings and a
Copy button. The button puts **plain text** on the clipboard, so an ordinary
Ctrl+V into Gmail lands as plain text rather than as an HTML email. A draft with
a hard finding is shown but cannot be copied.

The page names real people. It is written into the gitignored `outbox/`, and it
**must never be published, attached, or turned into an Artifact.** If the user
asks for it to be shared, say why not.

Use the sheet for the bulk of the pass; use the presentation below when talking
through Tier A drafts in the conversation.

## 3. Present by tier

**Tier A** — full text, one at a time. These get read end to end.

**Tier B** — the whole point of the tiering. Show only what differs from the
template:

```
Example Restoration Trust   [b2, high]
  + Your riparian work along the streambanks [e:1] is the kind of thing where
    the order matters more than the total.
    e1  https://example.org/projects  "planting along the streambanks"
```

One table, forty rows, one pass. The user approves, edits, or cuts each line.
They are reading the personalisation, not the email — the email was approved once
when the template was.

**Tier C** — do not show the text again. Confirm the count per bracket and that
every one is the approved template with only name and organisation merged.

## 3. Take edits

When the user rewrites a line, apply it exactly. Do not improve it on the way in.
Their wording is the training signal for `/email-learn`, and a line you polished
teaches the pipeline your register instead of theirs.

## 4. Render for sending

```powershell
python outreach\tools\check.py --render outreach\outbox\<batch>\<slug>.md
```

This is the **only** way to produce send-ready text, and it refuses on a hard
finding. Do not hand-copy a draft around the gate, and do not reconstruct the
text yourself — the marker-stripping is what stops `[e:1]` reaching a recipient,
and the `[video]` substitution is what puts the link in.

**Before the video exists**, everything up to this point still runs: templates,
research, drafts, and the whole proofreading pass. Only `--render` blocks, and it
says so plainly. Do not work around it by pasting a URL into the drafts.

Render in send order, tier A first.

## 5. Record what was actually sent

For each one the user sends, write their final text to
`outreach/outbox/<batch>/<slug>.sent.md` and mark the record:

```powershell
python outreach\tools\queue.py --sent <slug>
```

The `.sent.md` file is not bookkeeping. It is the input to `/email-learn` — the
diff between what you drafted and what they sent is the only honest measure of
how wrong the drafts are.

## Batch discipline

Remind the user once, not every time: `config.yaml: send.batch_size` is 18, and
the reason is that a hundred cold emails from one address in a day lands in
Promotions. The second reason is that if a tenth accept, a full send is ten calls
in one week.

Never offer to send. There is no mail connector wired in, sending stays manual,
and the paste is the last human checkpoint.

## Never

- Never present a draft that has not passed the gate.
- Never improve a line the user wrote.
- Never mark something sent that the user has not said they sent.
- Never reconstruct send-ready text by hand instead of using `--render`.
