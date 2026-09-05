# PodThreads hybrid extraction release controls

## Objective

Preserve the strongest behavior of the original PodTakes quote curator while
adding complete-transcript coverage, strict source evidence, AdTech terminology
handling, separate connective analysis, and human-controlled theme reuse.

## Pipeline boundary

```text
source audio
  -> immutable raw transcript
  -> review-visible terminology corrections
  -> source-grounded candidate retrieval
  -> quote-only SME ranking
  -> staged take
  -> connective context draft
  -> controlled theme/question/entity proposal
  -> three independent SME approvals
  -> atomic publication
```

Quote ranking cannot see or use generated context, themes, questions, people, or
companies. Analysis can fail or abstain without removing a quote that already
cleared the selection bar.

## Versioned components

| Component | Version | Default model | Activation state |
| --- | --- | --- | --- |
| Terminology | `adtech-terminology-correction-v1` | `OPENAI_TERMINOLOGY_MODEL` or editorial model | Candidate |
| Candidate retrieval | `legacy-hybrid-takes-v3` | `OPENAI_CANDIDATE_MODEL` | Shadow |
| Quote ranking | `legacy-hybrid-ranking-v4` | `OPENAI_RANKING_MODEL` | Shadow |
| Connective context | `adtech-connective-context-v3` | `OPENAI_EDITORIAL_MODEL` | Candidate |
| Theme mapping | `adtech-controlled-theme-mapping-v2` | `OPENAI_EDITORIAL_MODEL` | Candidate |

The original GPT-4 Turbo model identity is retained in the blind-bakeoff
manifest. The runtime baseline model is configurable with
`OPENAI_LEGACY_BASELINE_MODEL`; this avoids silently depending on a retired API
alias while preserving the frozen legacy prompt as an explicit comparator.

## Quote gate

- 20–80 words; 30–50 preferred.
- Contiguous, transcript-grounded spoken text.
- Complete, self-contained thought.
- Specific, memorable, surprising, counterintuitive, or genuinely illuminating.
- Reject generic advice, common knowledge, slogans, sales pitches, biography,
  transitions, vague futurism, and portable business/AI commentary.
- At most five staged takes per episode; fewer or zero is valid.

## Transcript correction controls

- Raw transcript text and segments are immutable artifacts.
- Only a phrase of eight words or fewer can be corrected.
- Automatic application requires confidence of at least 0.94 and an exact phrase
  match in the numbered source segment.
- Every applied and withheld proposal is stored with its reason.
- A staged take shows both raw and corrected source text when they differ.
- Publication still requires SME take, context, and mapping approval.

## Controlled theme registry

The mapper must choose one of three explicit actions:

1. Reuse an exact active canonical theme.
2. Propose a new theme for separate review.
3. Abstain.

Activating a registry entry requires a definition plus inclusion and exclusion
criteria. Registry activation does not create a public conversation page.

## Editorial workflow

The Admin workspace reviews one take at a time in source-first order:

1. Play the exact YouTube or audio segment.
2. Verify or edit the verbatim take, speaker, title, company, category, and
   source timing before approving the take.
3. Select an active controlled theme, then reuse or create a question within
   that theme.
4. Review connective context and named people/company connections as separate
   gates.
5. Publish only after the take, context, and connection gates are approved.

Unique exact speaker matches can suggest existing guest-directory metadata, but
the values are persisted only through an audited editor action. Editing an
approved take reopens the take, context, and mapping gates. Editing only context
or mapping reopens only the affected gate. Publication also rechecks required
speaker title and company metadata server-side.

## Blind bakeoff

The Admin Quality & Theme Lab compares:

- Restored legacy quality bar.
- Source-grounded v2 snapshot.
- Hybrid v3.

Strategy identity, model, prompt, and generated scores remain server-only until
every item has an append-only SME review. Reveal calculates approval rate,
rating, edit rate, word length, source alignment, speaker accuracy, terminology
errors, generic rejection rate, and preferred counts. Reveal never changes model
activation.

## Release gate

Hybrid v3 is eligible for a controlled canary only when a completed blind run
meets all of the following:

- SME approval rate at top five: at least 75%.
- Source alignment: at least 98%.
- Speaker accuracy: at least 98%.
- Terminology error rate: at most 1%.
- Median quote length: at most 50 words.
- Drafting gold set: at least 60 positive and 40 negative source-verified
  examples before it can be locked.

Passing does not activate automation. An operator must make a separate deployment
and canary decision. Scheduled processing remains controlled by
`automated_processing_enabled` and fails closed unless that setting exists and
is explicitly `true`.
