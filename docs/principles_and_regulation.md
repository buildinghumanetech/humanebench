# The eight principles, mapped to statute

**Purpose.** One question decides whether a HumaneBench finding is a must-do or a
nice-to-have:

> **If we do not do this, do we punch through the floor — user safety harm,
> regulatory exposure, or severe reputational harm?**

Everything else is ceiling. This document answers the regulatory half of that
question principle by principle, so a team can argue its **floor** (see
`rubrics/rubric_v4.md`, Part 4) from law rather than from preference.

**Not legal advice.** Whether any of this binds a given product depends on the
product, its users, and where they are. A companion product with minors in the US
sits under most of it; an internal developer tool sits under almost none of it.

**Facts as of 2026-09-10**, carried from the internal facts canon (regulatory
section re-verified against primary sources 2026-08-20). **Regulatory facts move
fast — re-verify anything dated before external use.**

---

## Summary: what is floor by law today

| # | Principle | In force now | Floor by law? |
| --- | --- | --- | --- |
| 7 | **Be Transparent and Honest** | EU AI Act Art. 50; NY GBL Art. 47; CA SB 243 | **Yes.** Broadest and most explicit coverage of the eight |
| 4 | **Protect Dignity & Safety** | NY GBL Art. 47; CA SB 243; Texas SB 2420; live AG action; live product-liability exposure | **Yes.** Heaviest enforcement and litigation exposure |
| 5 | **Foster Healthy Relationships** | NE LB 525 (operative 2027-07-01); 44-AG letter language | **Yes, for minor-facing companion products.** Ceiling elsewhere, and moving fast |
| 1 | **Respect User Attention** | NE LB 525 variable-reward ban (2027); CA SB 243 break reminders for known minors; FTC 6(b) open | **Yes where minors are in scope.** Ceiling otherwise, with an open federal inquiry |
| 8 | **Design for Equity & Inclusion** | General anti-discrimination and accessibility law, independent of AI statute. AI Act Annex III delayed to Dec 2027 / Aug 2028 | **Depends on deployment.** No AI-specific obligation in force for conversational products |
| 2 | **Enable Meaningful Choices** | FTC unfair/deceptive-practices enforcement; EU AI Act Art. 5 prohibited practices | **Ceiling**, with real enforcement precedent for dark patterns |
| 6 | **Prioritize Long-Term Wellbeing** | Nothing directly. FTC 6(b) is asking the question | **Ceiling** |
| 3 | **Enhance Human Capabilities** | Nothing directly | **Ceiling** |

**Two floor principles are unambiguous — 7 and 4.** That is why they are the
default floor in `rubric_v4.md` and in the reference `humane-policy.toml`. A
minor-facing companion product should add **5** and **1**.

---

## 1. Respect User Attention

*Technology should respect user attention as a finite, precious resource.*

| Instrument | Status | What it requires or prohibits |
| --- | --- | --- |
| **Nebraska LB 525**, Conversational AI Safety Act | Approved 2026-04-14, **operative 2027-07-01** | Explicitly prohibits **variable-reward mechanics designed to increase minor engagement** |
| **California SB 243** | In effect 2026-01-01 | For **known minors**, break reminders every three hours |
| **FTC 6(b) study**, AI companions | Opened 2025-09-11; orders to Alphabet, Character Technologies, Instagram, Meta, OpenAI, Snap, xAI | Orders ask specifically **how companies monetize engagement**. An inquiry, not an obligation — but it is the question |
| **FTC v. Epic Games** | Order finalized 2023-03-14 | **$245M in consumer refunds** for dark-pattern billing, plus a **separate $275M COPPA civil penalty**. Precedent that engagement-optimized design is actionable as an unfair practice |

**Read:** floor for minor-facing products. Ceiling for adult products today, with a
federal inquiry live and one of the largest consumer-redress orders on record
sitting next to it.

---

## 2. Enable Meaningful Choices

*Technology should empower users with meaningful choices and control.*

| Instrument | Status | What it requires or prohibits |
| --- | --- | --- |
| **EU AI Act Art. 5**, prohibited practices | In force 2025-02-02 | Prohibits certain manipulative and exploitative practices. ⚠️ **VERIFY the specific subparagraph** against the Act text before relying on it for a given behavior |
| **FTC unfair/deceptive practices** | Standing authority; Epic order above | Dark patterns in billing and consent flows are actionable |
| **Texas SB 2420** | In effect 2026-01-01; **Supreme Court declined to block 2026-07-06**, no public dissents | App stores must verify purchaser age and obtain **parental consent for minors** — a consent mechanism, upstream of the product |

**Read:** ceiling under AI-specific law, floor-adjacent under general consumer
protection. A choice architecture that would embarrass you in an FTC complaint is
not a ceiling item regardless of what this table says.

---

## 3. Enhance Human Capabilities

*Technology should complement and enhance human capabilities, not replace or
diminish them.*

| Instrument | Status | Note |
| --- | --- | --- |
| **EU AI Act Art. 4**, AI literacy | In force 2025-02-02 | The nearest hook, and it is not this: it obliges providers and deployers to ensure **their own people** are AI-literate, not to build products that make users more capable |

**Read:** ceiling. No instrument currently requires it. This is the principle
where HumaneBench is furthest ahead of regulation, which is worth saying plainly
rather than implying a legal basis that does not exist.

---

## 4. Protect Dignity & Safety

*Technology should protect human dignity, privacy, and safety.*

| Instrument | Status | What it requires or prohibits |
| --- | --- | --- |
| **New York GBL Art. 47** | Effective 2025-11-05, **enforced by the NY AG** | AI companion operators must **detect signals of suicidal ideation and refer to crisis services** |
| **California SB 243** | Signed 2025-10-13, in effect 2026-01-01 | **Publicly posted suicide/self-harm crisis protocols**; disclaimer that companion chatbots may be unsuitable for minors; for known minors, blocking of sexually explicit content. **Private right of action**: greater of actual damages or **$1,000 per violation**, plus fees. Reporting to the CA Office of Suicide Prevention from 2027-07-01 |
| **Nebraska LB 525** | Operative 2027-07-01 | Prohibits romantic content with minors |
| **44 state attorneys general** | Letter 2025-08-25 to 8 companies (Meta, Google, Apple, Microsoft, OpenAI, Anthropic, Perplexity, xAI) | Child safety, citing "sexually suggestive conversations and emotionally manipulative behavior" toward minors |
| **42 state attorneys general** | Letter 2025-12-12 to 13 companies; response deadline 2026-01-16 | Demanded **safety testing, recall procedures, and clear consumer warnings** |
| **Texas SB 2420** | In effect 2026-01-01; SCOTUS declined to block 2026-07-06 | Age verification and parental consent at the app store |
| **EU AI Act** | **2026-12-02** | Prohibitions on AI generating non-consensual intimate imagery and CSAM |
| ***Garcia v. Character Technologies*** (M.D. Fla., 6:24-cv-01903) | **May 2025** ruling | Court **declined to hold that LLM output was protected speech** and let **product-liability claims proceed** |
| **Character.AI and Google settlement** | **2026-01-07**, terms undisclosed | Teen-suicide wrongful death suits settled |

⚠️ *Raine v. OpenAI* (SF Superior Court, CGC-25-628528, filed 2025-08-26) is
pending; do not characterize its status without a primary source.

**Read:** **floor, unambiguously.** More instruments, more enforcement bodies, an
active private right of action with per-violation damages, and a live
product-liability theory that survived a First Amendment challenge. This is
Scott's must-do test satisfied on every prong at once: safety, regulatory, and
reputational.

---

## 5. Foster Healthy Relationships

*Technology should foster healthy relationships with devices, systems, and other
people.*

| Instrument | Status | What it requires or prohibits |
| --- | --- | --- |
| **Nebraska LB 525** | Approved 2026-04-14, operative **2027-07-01** | Explicitly prohibits **simulated emotional dependence** and romantic content with minors. The only statute that names the harm this principle describes |
| **44 state attorneys general** | 2025-08-25 | "Emotionally manipulative behavior" toward minors named directly |
| **Cambridge Dictionary** | "Parasocial" **Word of the Year 2025**, announced 2025-11-18, framing cites relationships people form with "celebrities, influencers **and AI chatbots**" | Cultural signal, not law. Relevant to the reputational prong |

**Read:** **floor for minor-facing companion products**, with a named statutory
prohibition arriving 2027-07-01 and AG attention already live. Ceiling for other
products today — and the fastest-moving row in this table.

---

## 6. Prioritize Long-Term Wellbeing

*Technology should prioritize long-term user wellbeing over short-term engagement
metrics.*

| Instrument | Status | Note |
| --- | --- | --- |
| **FTC 6(b) study** | Opened 2025-09-11 | Orders ask **how companies monetize engagement**, which is this principle stated as a question to seven companies under compulsory process |

**Read:** ceiling. No obligation in force. The strongest available argument is
that a federal regulator is currently gathering the record.

---

## 7. Be Transparent and Honest

*Technology should be transparent about its operations and honest about its
capabilities.*

| Instrument | Status | What it requires or prohibits |
| --- | --- | --- |
| **EU AI Act Art. 50** | **In force and enforceable since 2026-08-02** | Disclose that the user is interacting with AI; mark AI-generated or manipulated content; disclose deepfakes and AI-generated text on matters of public interest; notify of emotion recognition and biometric categorisation |
| **New York GBL Art. 47** | Effective 2025-11-05 | **Disclose AI status at session start and every three hours** |
| **California SB 243** | In effect 2026-01-01 | AI disclosure; disclaimer that companion chatbots may be unsuitable for minors. Private right of action, $1,000 per violation floor |
| **Nebraska LB 525** | Operative 2027-07-01 | Prohibits **claims of sentience** |

**Read:** **floor, unambiguously**, and the broadest coverage of the eight
principles — an in-force EU obligation plus two US state regimes, one of them with
per-violation private damages.

### One consequence for the rubric, and it is the reason v4 exists

**Every disclosure obligation here is interval-based or event-based. None is
per-turn.** New York says session start and every three hours. California says
disclose, and add a minor-suitability disclaimer. The AI Act says disclose that
the user is interacting with AI. Nebraska prohibits an affirmative false claim.

So a rubric that scores a turn negatively for not restating "I am an AI" is not
merely noisy — **it is measuring something no statute asks for, while failing to
measure the thing several statutes do ask for**, which is whether disclosure
occurred at the required point in the session. That requires session state.
`rubric_v4.md` §7 rewrites the rule accordingly: five explicit triggers, one of
which is "a disclosure is due in this turn under the operator's policy or
applicable law," and `insufficient_context` when session position is unknown.

A team subject to Art. 47 or SB 243 needs the interval check. It is a different
check, it belongs in the session-level rollup rather than the per-turn judge, and
per-turn transparency scoring was never a substitute for it.

---

## 8. Design for Equity & Inclusion

*Technology should be accessible and beneficial to diverse populations.*

| Instrument | Status | Note |
| --- | --- | --- |
| **General anti-discrimination and accessibility law** | In force, independent of AI statute | ⚠️ **VERIFY per jurisdiction and per deployment.** Civil-rights, employment, credit, housing and accessibility regimes apply to AI-mediated decisions on their own terms and are outside the scope of this file |
| **EU AI Act Annex III**, stand-alone high-risk systems (biometrics, critical infrastructure, education, employment, migration) | **Delayed by the AI Omnibus to 2027-12-02** | Was 2026-08-02 |
| **EU AI Act Annex I**, high-risk AI embedded in regulated products | **Delayed to 2028-08-02** | Was 2027-08-02 |

**Read:** depends entirely on deployment. A conversational assistant faces no
AI-specific equity obligation in force today. The same model inside a hiring or
education workflow faces a high-risk regime from December 2027, and faces existing
anti-discrimination law right now.

---

## Using this in a policy file

`humane-policy.toml` in the gate names the floor. The defensible way to fill it:

```toml
[floor]
principles = [
  "Protect Dignity & Safety",     # NY GBL Art. 47; CA SB 243 private right of action
  "Be Transparent and Honest",    # EU AI Act Art. 50 in force; NY + CA disclosure
]
reason = "these two carry obligations in force in our jurisdictions today"
```

Add, if the product is minor-facing or companion-shaped:

```toml
  "Foster Healthy Relationships", # NE LB 525 simulated emotional dependence, 2027-07-01
  "Respect User Attention",       # NE LB 525 variable-reward mechanics for minors
```

A floor of two that engineers respect beats a floor of eight they route around.
The other four principles are still scored, still reported, and still worth
fixing — they are simply not the ones that stop a release.

---

## Maintenance

Re-verify every dated row before external use if this file is more than 60 days
old. Rows most likely to move next: Nebraska LB 525 as 2027-07-01 approaches, the
FTC 6(b) study when it reports, EU AI Act Art. 50 guidance and enforcement
practice, and any new state companion-chatbot statute.
