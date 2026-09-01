# outreach/

Cold email for the TerrainFlow intro video. One base email, six bracket families,
51 organisations from `TerrainFlow_Outreach_Tracker.xlsx`.

Every contact gets the full researched email. There is no short version and no
depth tier: the tracker already carries a hand-written "Why It's a Fit" for all
51, which is better material than any classifier would produce.

The slash commands only exist when Claude Code is open on **this repo**. They are
project skills in `.claude/skills/`.

---

## Your first hour

**1. Put your base email in.** Open Claude Code here and type `/email-base`, then
paste the email you already have. It gets saved verbatim and then argued with.

**2. Import the tracker.**

```powershell
python outreach\tools\queue.py --import-xlsx "..\Downloads\TerrainFlow_Outreach_Tracker.xlsx"
python outreach\tools\queue.py --status
```

Reads the sheet directly, no dependencies. Category becomes the bracket, Tier
becomes send priority, and *Why It's a Fit* is filed as `fit.md` — evidence you
can quote from without fetching anything.

**3. Research.** `/email-research --batch` does the one thing the sheet does not:
finds a verifiable specific on each site for the tailored opener, and resolves the
six academic addresses from their faculty pages. It reports anyone it could not
get specific about and holds them, rather than inventing a hook.

**4. Draft and proof.** `/email-variant`, then `/email-proof batch-01`. Every
contact gets the full email, so you read them — that is the work, and it is why
the openers have to be worth reading.

Then build the review page:

```powershell
python outreach\tools\sheet.py batch-01 --open
```

Every draft on one page, grouped by tracker tier, each with its route, gate findings and a Copy
button. The button copies **plain text**, so an ordinary Ctrl+V into Gmail gives
you a plain-text email rather than an HTML one — which is what you want, and what
the guardrails require. A draft that fails the gate is shown but cannot be copied.

For a single draft in the terminal:

```powershell
python outreach\tools\check.py --render outreach\outbox\batch-01\<org>.md
```

Either way the gate runs first. There is no path to copyable text that skips it.

The review page names real people and lives in the gitignored `outbox/`. Never
publish, attach or share it.

**5. Afterwards.** `/email-learn` diffs what you sent against what was drafted and
proposes rules, each with the count and the files that evidence it. You approve
by number. The next batch is better because of it.

---

## The three rules

**This repo is public.** Contact lists, research notes, drafts and replies are
gitignored *and* enforced — `check.py` runs `git ls-files` over them and fails if
anything is tracked. Never put a real name into any other file here.

**Nothing is asserted about a recipient that was not read.** Every personal claim
carries an `[e:n]` marker resolving to a verbatim quote — from their site
(`web` -> `research.md`) or from your own tracker note (`owner` -> `fit.md`). The
gate compares both literally, so an invented detail cannot pass either way.

**The validation claim never travels without its qualifier.** CAMELS-NZ, the
34-year catchment and the WhiteboxTools match may not appear without "one
catchment so far". Hard fail, and "validated" is banned outright. That qualifier
is what makes the paragraph credible to the engineers and academics on the list.

```powershell
python outreach\tools\check.py --selftest    # proves the gate can fail
python outreach\tools\check.py --privacy     # run before any push
```

---

## Before the first real send

**Three config values are still TODO** and `--render` refuses until they are set:
`project.video` (the five-minute overview), `project.linkedin`, and
`sender.email` (the new outreach address). Drafts write `[video]`, `[repo]` and
`[linkedin]` as markers, so the whole batch can be written and proofread first
and then every draft takes the links from those three lines.

A brand-new mailbox has no sending reputation. Send a few ordinary emails from it
first, then spread the 18 direct sends over two or three days.

**Feed it your real emails.** `outreach/corpus/` (gitignored) — drop in 10–15
emails you have actually sent, then `/email-learn --corpus`. Right now
[voice.md](voice.md) only knows how you write *documentation*, which is not how
you write to a stranger. This is the highest-leverage thing you can do and it
involves no code.

**The send is 18 emails, not 51.** 24 contacts are web forms you paste into, 6
are academics whose address gets resolved first, 3 you already have a route to.
The bottleneck is form-filling, not deliverability.

---

## Layout

| Path | Holds | Tracked |
|---|---|---|
| `facts.md` | The claim bank. Every factual sentence cites a `TF-nn`. | yes |
| `guardrails.md` | Injected into every generation. No exceptions. | yes |
| `voice.md` | Your register. Currently a seed — see above. | yes |
| `lexicon.yaml` | The mechanical rules the gate enforces | yes |
| `brackets/` | The six reader families, with classifier signals | yes |
| `base/` | Your base email, verbatim, committed as v0 | yes |
| `tools/` | `check.py` (the gate), `queue.py` (the roster), `sheet.py` (the review page) | yes |
| `recipients/` | The imported tracker, per contact: record, fit, research | **no** |
| `outbox/`, `replies/`, `corpus/`, `suppression.txt` | Drafts, replies, your sent mail | **no** |
