# PodTakes operations and audit runbook

## Release order

1. Run backend unit tests and frontend lint/build.
2. Apply reviewed Supabase migrations. DDL belongs in migrations, not ad-hoc
   dashboard SQL.
3. Run Supabase security and performance advisors. Resolve security lints before
   releasing; document any intentionally deferred infrastructure notice.
4. Deploy Edge Functions, then verify rejection cases.
5. Confirm `modal profile current` reports `aron-personal` and deploy the Modal
   app.
6. Run the Modal health function and HTTP health/unauthorized checks.
7. Run one bounded episode only when a live content/cost smoke test is warranted.
8. Inspect every staged sample for source fidelity and editorial specificity.
9. Test guest, mobile, exact-take, and admin authorization flows before frontend
   production deployment.

## Required secrets and configuration

`podtakes-secrets` must contain `SUPABASE_URL`, `SUPABASE_KEY`, and
`OPENAI_API_KEY`. Optional runtime controls include:

- `OPENAI_CANDIDATE_MODEL` (default `gpt-5.6-terra`)
- `OPENAI_EDITORIAL_MODEL` (default `gpt-5.6-sol`)
- `MIN_EDITORIAL_QUALITY` (default `0.78`)
- `MIN_CONTEXT_CONFIDENCE` (default `0.72`)
- `MAX_EPISODES_PER_RUN` (default `3`)
- `SCHEDULED_MAX_EPISODES` (default `2`)
- `ALLOWED_ORIGINS` (comma-separated frontend origins)

Administrator authority must be assigned in signed Supabase `app_metadata` as
`{"role":"admin"}`. Never authorize from `user_metadata` or an email hardcode.
An operator must refresh the session or sign in again after the claim changes.

## Job monitoring

`processing_jobs` is the durable source of truth. Normal state progression is:

`queued -> claimed -> downloading -> transcribing -> extracting -> ranking -> mapping -> staging -> succeeded`

Every in-progress job updates `heartbeat_at`, `progress`, and the current episode.
A failed job records `error_code`, bounded `error_message`, result details, and
timestamps. Do not restart a job only because the console is quiet; check its
heartbeat first. Scheduled runs use a daily idempotency key.

Useful read-only check:

```sql
select id, source, state, progress, error_code, error_message,
       heartbeat_at, created_at, completed_at
from public.processing_jobs
order by created_at desc
limit 20;
```

## Editorial acceptance checklist

Before approving a take, the SME should confirm:

- the quote is literal and its segment/timestamp bounds contain the complete take;
- the take is specific enough to change an industry assumption or decision;
- the context connects the take to a real industry mechanism, stakeholder tension,
  prior idea, or adjacent debate without announcing its importance;
- transcript evidence and domain inference are labeled correctly;
- the category and speaker identity are defensible;
- the analysis does not add an unsupported fact or imply false certainty.
- the proposed theme is durable enough to connect multiple episodes rather than
  acting as a category or episode summary;
- the question is open, legible, and genuinely answered by the take;
- every person and company connection has a recorded source or explicit
  editorial basis; inferred employment or partnership is never accepted.

Take, context, and conversation-mapping approvals are separate. Promotion remains
disabled until all three are complete, then the production quote and conversation
graph are written atomically. A reject decision requires a reason. Edits and undo
actions append new `curation_decisions`; they do not rewrite history.

## Model evaluation gate

Run a held-out evaluation after at least six approved and six rejected
source-backed decisions exist. The balanced evaluation excludes labeled examples
from its prompt and records per-item predictions. Current thresholds are:

- precision at least `0.75`;
- approved-take recall at least `0.60`;
- rejected-take exclusion at least `0.75`.

A failed run is recorded and does not silently change the configured model.
Human review remains mandatory even after a passing evaluation.

## Incident controls

- Disable automated processing through the authenticated admin automation
  control; verify the change in `automation_settings` and `automation_logs`.
- New processing is capped per run. Do not raise caps during an incident.
- Protected Modal routes must return `401` without a JWT and `403` without the
  administrator app claim.
- Browser roles have no direct production-content mutation privilege. Promotion,
  feed configuration, and curation writes cross the authenticated Modal boundary.
- Anonymous support accepts only bounded IDs and approved origins, stores a
  keyed HMAC rather than an IP address, and calls a service-only atomic RPC.
- Supabase migrations are forward-only. Remediate with a new migration; do not
  rewrite applied migration history.
- Data repairs are limited to deterministic relationships and record before/after
  values in `data_quality_issues`. Speaker identity and topic taxonomy remain
  human decisions.

## Rollback strategy

- Modal: redeploy the last verified commit; the app deployment history remains
  available in the `aron-personal` dashboard.
- Frontend: redeploy the last verified artifact/commit.
- Database: restore from the Supabase backup only for a material incident. For a
  normal defect, ship a corrective migration. Deterministic repair before-values
  are retained in `data_quality_issues` for targeted reversal.
- Content: never delete an audit decision to undo publication. Record an undo or
  removal decision and preserve its reason.
