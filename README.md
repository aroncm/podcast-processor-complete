# PodTakes podcast processor

The backend for PodTakes: complete-episode ingestion, timestamped transcription,
literal take extraction, adtech SME ranking and contextual-analysis proposals,
and an auditable human publication workflow.

## Editorial contract

- Candidate extraction reads the complete transcript in overlapping chunks and
  may abstain. It must not rewrite a speaker's words.
- Direct evidence is tied to transcript segment IDs. Domain inference is labeled
  separately from transcript-backed claims.
- A high score does not publish anything. The take and its contextual analysis
  require separate SME approvals.
- Rejections capture a reason and the reviewer's expertise lens. Decisions are
  append-only and usable for held-out evaluation.
- Promotion is an atomic service-only database operation and refuses incomplete
  source, speaker, category, or context data.

## Runtime

- Modal app: `podcast-processor-full`
- Modal workspace: `aron-personal`
- Public health: `GET /health`
- Protected API: Supabase administrator JWT in `Authorization: Bearer ...`
- Schedule: daily at `00:00 UTC`, subject to the database automation switch
- Pipeline: `podtakes-sme-v1`

The runtime image and Python dependencies are pinned in
`modal_app/full_processor.py` and `requirements.txt`. Credentials live in the
Modal secret `podtakes-secrets`; they are not included in the image or browser.

## Local verification and deployment

```bash
venv/bin/python -m py_compile modal_app/full_processor.py
venv/bin/python -m unittest tests.test_pipeline_quality -v
venv/bin/modal profile current
venv/bin/modal run modal_app/full_processor.py --action health
venv/bin/modal deploy modal_app/full_processor.py
```

Run a bounded, durable operator smoke test only when an episode-processing cost
is acceptable:

```bash
venv/bin/modal run modal_app/full_processor.py \
  --action process --max-episodes 1 --days-back 30
```

The CLI processing action creates a `processing_jobs` record before it invokes
the worker. See [OPERATIONS.md](OPERATIONS.md) for release order, monitoring,
quality gates, and incident controls.
