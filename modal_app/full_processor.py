"""Full processor with real transcription and GPT quote extraction - Quality-focused version"""

import modal
import os
import json
import hashlib
import uuid
from datetime import datetime, timedelta, timezone

app = modal.App("podcast-processor-full")

# Enhanced image with ffmpeg for audio processing
image = modal.Image.debian_slim() \
    .pip_install(
        "supabase==2.31.0",
        "openai==3.3.1",
        "feedparser==6.0.14",
        "pydub==0.25.1",
        "fastapi==0.141.1",
        "youtube-transcript-api==1.2.4",
        "thefuzz==0.22.1",
        "yt-dlp==2026.8.19",
        "requests==2.34.2"
    ) \
    .apt_install("ffmpeg")

# Keep runtime credentials in Modal rather than baking a local .env snapshot into
# the deployed app definition. Create/update this with:
#   modal secret create podtakes-secrets --from-dotenv .env
my_secret = modal.Secret.from_name("podtakes-secrets")

PIPELINE_VERSION = "podtakes-sme-v1"
EXTRACTION_PROMPT_VERSION = "take-candidates-v2"
RANKING_PROMPT_VERSION = "adtech-sme-ranking-v2"
CONTEXT_PROMPT_VERSION = "adtech-sme-context-v1"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_processing_job(supabase, job_id: str | None, state: str, **fields) -> None:
    """Best-effort audit update that never hides the underlying pipeline error."""
    if not job_id:
        return
    payload = {
        "state": state,
        "heartbeat_at": utcnow_iso(),
        "updated_at": utcnow_iso(),
        **fields,
    }
    try:
        supabase.table("processing_jobs").update(payload).eq("id", job_id).execute()
    except Exception as exc:
        print(f"AUDIT_WARNING job={job_id} state={state} update_failed={exc}")


def update_processing_job_from_env(job_id: str | None, state: str, **fields) -> None:
    if not job_id:
        return
    try:
        from supabase import create_client
        supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        update_processing_job(supabase, job_id, state, **fields)
    except Exception as exc:
        print(f"AUDIT_WARNING job={job_id} state={state} client_failed={exc}")


def _process_episode_with_ai_impl(
    feed_ids: list = None,
    start_date: str = None,
    end_date: str = None,
    max_episodes: int = None,
    job_id: str = None,
):
    """Process full episode with quality-focused quote extraction. Supports manual date/feed filtering."""
    
    import feedparser
    import subprocess
    import tempfile
    import time
    from datetime import datetime
    from supabase import create_client
    from openai import OpenAI
    
    print(f"🚀 Starting AI-powered processing... (Manual Filter: {bool(feed_ids or start_date)})")
    
    # Initialize clients
    supabase = create_client(
        os.environ['SUPABASE_URL'],
        os.environ['SUPABASE_KEY']
    )
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    update_processing_job(
        supabase,
        job_id,
        "claimed",
        claimed_at=utcnow_iso(),
        started_at=utcnow_iso(),
        attempt_count=1,
    )
    
    # Automated/manual-all runs should respect the active-feed control. An
    # explicitly selected feed remains callable for targeted diagnostics.
    feeds_query = supabase.table('test_podcast_feeds').select('*')
    if not feed_ids:
        feeds_query = feeds_query.eq('active', True)
    feeds = feeds_query.execute()
    if not feeds.data:
        result = {"success": False, "error": "No active podcast feeds found"}
        update_processing_job(
            supabase,
            job_id,
            "failed",
            result=result,
            error_code="no_active_feeds",
            error_message=result["error"],
            completed_at=utcnow_iso(),
        )
        return result
    
    # Filter by specific feed_ids if provided
    if feed_ids:
        feeds.data = [f for f in feeds.data if f['id'] in feed_ids]
        if not feeds.data:
            result = {"success": False, "error": "Selected podcast feeds were not found"}
            update_processing_job(
                supabase,
                job_id,
                "failed",
                result=result,
                error_code="selected_feeds_not_found",
                error_message=result["error"],
                completed_at=utcnow_iso(),
            )
            return result
        
    all_results = []
    attempted_episodes = 0
    effective_max_episodes = max_episodes or int(os.environ.get("MAX_EPISODES_PER_RUN", "3"))
    
    # Parse Date Filters
    start_dt = datetime.fromisoformat(start_date) if start_date else None
    end_dt = datetime.fromisoformat(end_date) if end_date else None
    
    for feed in feeds.data:
        if attempted_episodes >= effective_max_episodes:
            break
        print(f"\n📡 Processing Feed: {feed['name']}")
        
        try:
            parsed = feedparser.parse(feed['rss_url'])
            
            # Find ALL unprocessed episodes in this feed
            new_episodes = []
            
            # Limit processing to avoid timeouts (Cron catches up)
            MAX_EPISODES_PER_FEED = 2
            
            # If manual mode, check deeper
            check_depth = 50 if (start_date or end_date) else 10
            
            for entry in parsed.entries[:check_depth]:
                # Date Filter Check (New Feature)
                if start_dt or end_dt:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])
                        if start_dt and pub_date < start_dt:
                            continue
                        if end_dt and pub_date > end_dt:
                            continue
                        print(f"  📅 Match Date: {pub_date.date()} ({entry.title[:30]}...)")
                    except:
                        pass # Skip date check if parsing fails

                # 1. Check GUID (Best)
                is_duplicate = False
                if hasattr(entry, 'id'):
                    res = supabase.table('test_quotes').select('id').eq('episode_guid', entry.id).limit(1).execute()
                    if res.data:
                        is_duplicate = True
                
                # 2. Check Truncated Title (Fallback - DB stores max 100 chars)
                if not is_duplicate:
                    truncated_title = entry.title[:100]
                    res = supabase.table('test_quotes').select('id').eq('episode_name', truncated_title).limit(1).execute()
                    if res.data:
                        is_duplicate = True
                
                if not is_duplicate:
                    print(f"  📎 Found new episode: {entry.title}")
                    new_episodes.append(entry)
                else:
                    print(f"  ⏭️  Skipping existing: {entry.title[:30]}...")
                    pass # Don't break loop in manual mode, might find older unchecked ones
            
            if not new_episodes:
                print(f"  ✅ No new episodes for {feed['name']}")
                continue
            
            # Reduce batch size to prevent Timeout (Unless manual mode)
            is_manual = bool(start_date or end_date or feed_ids)
            if not is_manual and len(new_episodes) > MAX_EPISODES_PER_FEED:
                print(f"  ⚠️ Limiting to {MAX_EPISODES_PER_FEED} episodes (from {len(new_episodes)}) to prevent timeout.")
                new_episodes = new_episodes[:MAX_EPISODES_PER_FEED]

            remaining = effective_max_episodes - attempted_episodes
            new_episodes = new_episodes[:max(0, remaining)]
                
            print(f"  ✨ Processing {len(new_episodes)} new episodes for {feed['name']}...")
            
            # Process each new episode
            for episode in new_episodes:
                episode_guid = getattr(episode, "id", None)
                update_processing_job(
                    supabase,
                    job_id,
                    "downloading",
                    current_episode_guid=episode_guid,
                    progress={
                        "attempted_episodes": attempted_episodes,
                        "current_podcast": feed["name"],
                        "current_episode": episode.title,
                    },
                )
                result = process_single_episode_logic(
                    episode,
                    feed,
                    client,
                    supabase,
                    job_id=job_id,
                )
                all_results.append(result)
                attempted_episodes += 1
                
        except Exception as e:
            print(f"❌ Error processing feed {feed['name']}: {str(e)}")
            continue
        
    failed_results = [item for item in all_results if isinstance(item, dict) and item.get("error")]
    successful_results = [item for item in all_results if isinstance(item, dict) and not item.get("error")]
    result = {
        "success": len(failed_results) == 0,
        "partial_success": bool(failed_results and successful_results),
        "processed_count": len(successful_results),
        "failed_count": len(failed_results),
        "details": all_results,
    }
    final_state = "failed" if failed_results and not successful_results else "succeeded"
    update_processing_job(
        supabase,
        job_id,
        final_state,
        result=result,
        progress={"attempted_episodes": attempted_episodes},
        error_code="episode_processing_failed" if final_state == "failed" else None,
        error_message=(
            "; ".join(str(item.get("error")) for item in failed_results)[:4000]
            if final_state == "failed" else None
        ),
        completed_at=utcnow_iso(),
    )
    return result


@app.function(
    image=image,
    secrets=[my_secret],
    timeout=1800,
    cpu=2,
)
def process_episode_with_ai(
    feed_ids: list = None,
    start_date: str = None,
    end_date: str = None,
    max_episodes: int = None,
    job_id: str = None,
):
    """Audited Modal entrypoint. The job row remains the durable source of truth."""
    try:
        return _process_episode_with_ai_impl(
            feed_ids=feed_ids,
            start_date=start_date,
            end_date=end_date,
            max_episodes=max_episodes,
            job_id=job_id,
        )
    except Exception as exc:
        update_processing_job_from_env(
            job_id,
            "failed",
            error_code=type(exc).__name__,
            error_message=str(exc)[:4000],
            completed_at=utcnow_iso(),
        )
        raise

# BOILERPLATE MOCK Implementation for Missing Apps 
# (Real implementation would duplicate logic, for now we restore the stubs/functions 
# so the Modal dashboard looks correct and they can be expanded)

def slugify(text: str) -> str:
    """Simple slugify for ID generation"""
    if not text: return "unknown"
    import re
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

def normalize_text(text: str) -> str:
    """Rigorous text normalization for caption matching."""
    if not text:
        return ""
    import re
    # Lowercase
    text = text.lower()
    # Normalize quotes, dashes, etc
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", " ").replace("–", " ").replace("-", " ")
    text = text.replace("'", "")
    # Remove punctuation for matching form
    text = re.sub(r"[^\w\s]", ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# YouTube Caption Timestamp Alignment
# ─────────────────────────────────────────────────────────────────────────────

# Module-level caption cache: {youtube_id: [{text, start, duration}, ...]}
_caption_cache: dict = {}

def get_yt_captions(youtube_id: str) -> list | None:
    """Fetch and parse YouTube captions using yt-dlp (json3 format).
    Results are cached in-memory per video to avoid redundant API calls."""
    import yt_dlp
    import requests
    import json
    
    if youtube_id in _caption_cache:
        return _caption_cache[youtube_id]
    
    try:
        print(f"  🎬 Fetching YouTube captions for {youtube_id} via yt-dlp...")
        ydl_opts = {
            'skip_download': True,
            'writeautosubs': True,
            'subtitleslangs': ['en.*'],
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_id, download=False)
            
            # Find the best English subtitle (auto or manual)
            sub_url = None
            
            # Try manual subtitles first
            if 'subtitles' in info and 'en' in info['subtitles']:
                for fmt in info['subtitles']['en']:
                    if fmt.get('ext') == 'json3':
                        sub_url = fmt['url']
                        break
            
            # Fallback to automatic captions
            if not sub_url and 'automatic_captions' in info:
                # Often 'en', 'en-us', etc.
                en_keys = [k for k in info['automatic_captions'].keys() if k.startswith('en')]
                if en_keys:
                    # Pick first one, look for json3
                    for fmt in info['automatic_captions'][en_keys[0]]:
                        if fmt.get('ext') == 'json3':
                            sub_url = fmt['url']
                            break
            
            if not sub_url:
                print(f"  ⚠️  No English json3 captions found for {youtube_id}")
                _caption_cache[youtube_id] = None
                return None
            
            # Fetch the actual json3 data
            resp = requests.get(sub_url)
            if resp.status_code != 200:
                print(f"  ⚠️  Failed to download captions from {sub_url[:50]}...")
                _caption_cache[youtube_id] = None
                return None
            
            data = resp.json()
            if 'events' not in data:
                print(f"  ⚠️  Unrecognized caption format for {youtube_id}")
                _caption_cache[youtube_id] = None
                return None
            
            # 2. Parse caption events into {start, end, raw_text, norm_text, word_count}
            processed = []
            for event in data['events']:
                if 'segs' not in event: continue
                
                start_ms = event.get('tStartMs', 0)
                duration_ms = event.get('dDurationMs', 0)
                
                # Concatenate segments
                text = "".join([s.get('utf8', '') for s in event['segs']]).strip()
                if not text: continue
                
                processed.append({
                    'start': start_ms / 1000.0,
                    'end': (start_ms + duration_ms) / 1000.0,
                    'raw_text': text,
                    'norm_text': normalize_text(text),
                    'word_count': len(text.split())
                })
            
            print(f"  ✅ Parsed {len(processed)} caption events for {youtube_id}")
            _caption_cache[youtube_id] = processed
            return processed
            
    except Exception as e:
        print(f"  ⚠️  Caption fetch error for {youtube_id}: {e}")
        _caption_cache[youtube_id] = None
        return None

def align_timestamps_to_youtube_captions(
    quote_text: str,
    youtube_id: str,
    whisper_start: int,
    whisper_end: int
) -> dict | None:
    """Deterministic caption-alignment pipeline with strict confidence gating."""
    from difflib import SequenceMatcher
    
    captions = get_yt_captions(youtube_id)
    if not captions:
        return None

    # Step 2: Normalize quote text
    norm_quote = normalize_text(quote_text)
    quote_words = norm_quote.split()
    quote_word_count = len(quote_words)
    if quote_word_count < 5:
        return None # Too short for reliable alignment

    # Step 3: Find quote location via sliding-window matching
    best_candidate = None
    best_composite_score = 0
    second_best_composite = 0
    
    # Candidate size scales with quote length
    min_win = max(1, int(quote_word_count * 0.55))
    max_win = int(quote_word_count * 2.2)
    
    # To optimize, we search in a +/- 5 minute window around the whisper timestamp
    SEARCH_RADIUS = 300 # seconds
    search_start_time = max(0, whisper_start - SEARCH_RADIUS)
    search_end_time = whisper_end + SEARCH_RADIUS
    
    # Find relevant caption indices
    start_idx = 0
    end_idx = len(captions)
    for i, c in enumerate(captions):
        if c['start'] < search_start_time: start_idx = i
        if c['start'] > search_end_time: 
            end_idx = i
            break
            
    # Sliding window over indices
    for i in range(start_idx, end_idx):
        for win_size in range(1, 15): # Most quotes span < 15 caption events
            if i + win_size > len(captions): break
            
            window_events = captions[i : i+win_size]
            window_text = " ".join([e['norm_text'] for e in window_events])
            window_words = window_text.split()
            
            # Length guardrails
            if len(window_words) < min_win: continue
            if len(window_words) > max_win: break
            
            # Scoring
            sm_ratio = SequenceMatcher(None, norm_quote, window_text).ratio()
            
            # Token metrics
            needle_set = set(quote_words)
            candidate_set = set(window_words)
            common = needle_set & candidate_set
            
            f1 = 0
            recall = 0
            if common:
                prec = len(common) / len(candidate_set)
                rec = len(common) / len(needle_set)
                f1 = 2 * (prec * rec) / (prec + rec)
                recall = rec
            
            # Composite Score: 0.6 * SM + 0.25 * F1 + 0.15 * Recall
            composite = (0.6 * sm_ratio) + (0.25 * f1) + (0.15 * recall)
            
            if composite > best_composite_score:
                # Check for overlap with existing best to track ambiguity
                is_overlapping = False
                if best_candidate:
                    if not (i + win_size <= best_candidate['start_idx'] or i >= best_candidate['end_idx']):
                        is_overlapping = True
                
                if not is_overlapping:
                    second_best_composite = best_composite_score
                
                best_composite_score = composite
                best_candidate = {
                    'start_idx': i,
                    'end_idx': i + win_size,
                    'start_time': window_events[0]['start'],
                    'end_time': window_events[-1]['end']
                }
            elif composite > second_best_composite:
                # Track non-overlapping rivals for ambiguity check
                is_overlapping = (not (i + win_size <= best_candidate['start_idx'] or i >= best_candidate['end_idx']))
                if not is_overlapping:
                    second_best_composite = composite

    # Step 4: Confidence/ambiguity gates
    if not best_candidate:
        return None
        
    # Thresholds (Tuned based on user policy)
    MIN_COMPOSITE = 0.75
    MIN_MARGIN = 0.04 # Strict margin over second best
    
    if quote_word_count > 30:
        MIN_COMPOSITE = 0.70 # Looser for very long quotes
        MIN_MARGIN = 0.03
        
    is_ambiguous = (best_composite_score - second_best_composite) < MIN_MARGIN
    
    if best_composite_score < MIN_COMPOSITE or is_ambiguous:
        reason = "low_confidence" if best_composite_score < MIN_COMPOSITE else "ambiguous_match"
        print(f"  ⚠️ YT Alignment rejected ({reason}): score={best_composite_score:.2f}, margin={best_composite_score-second_best_composite:.3f}")
        return None

    # Step 5: Context window expansion (±30s) + sentence-safe alignment
    exact_start = best_candidate['start_time']
    exact_end = best_candidate['end_time']
    
    padded_start = max(0, int(exact_start - 30))
    padded_end = int(exact_end + 30)
    
    # Sentence boundary alignment logic
    # We look for punctuation (.!?) or gaps > 1.5s as sentence boundaries
    final_start = padded_start
    final_end = padded_end
    
    # Find nearest sentence start at/before padded_start
    # Search within 15s of the padded start
    for i in range(len(captions)-1, -1, -1):
        c = captions[i]
        if c['start'] <= padded_start:
            # Check if this or previous event ended a sentence
            is_sentence_boundary = False
            if i > 0:
                prev = captions[i-1]
                if any(p in prev['raw_text'] for p in ['.', '!', '?']):
                    is_sentence_boundary = True
                elif c['start'] - prev['end'] > 1.5:
                    is_sentence_boundary = True
            
            if is_sentence_boundary:
                final_start = int(c['start'])
                break
        if c['start'] < padded_start - 15: break

    # Find nearest sentence end at/after padded_end
    for i in range(len(captions)):
        c = captions[i]
        if c['end'] >= padded_end:
            if any(p in c['raw_text'] for p in ['.', '!', '?']):
                final_end = int(c['end'])
                break
            # Or if next event starts after a big gap
            if i + 1 < len(captions):
                if captions[i+1]['start'] - c['end'] > 1.5:
                    final_end = int(c['end'])
                    break
        if c['end'] > padded_end + 15: break
            
    confidence = round(best_composite_score, 3)
    print(f"  🎯 YT Deterministic Match: {final_start}s–{final_end}s (conf={confidence}, margin={best_composite_score-second_best_composite:.3f})")
    
    return {
        'start': final_start,
        'end': final_end,
        'confidence': confidence
    }


@app.function(image=image, secrets=[my_secret], timeout=600)
def promote_quote_to_production(quote_id: str, reviewer_id: str = None):
    """Atomically promote an SME-approved take and approved context."""
    print(f"🚀 Promoting curated quote {quote_id}...")
    from supabase import create_client
    supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    try:
        result = supabase.rpc(
            "promote_curated_quote",
            {"p_quote_id": quote_id, "p_reviewer_id": reviewer_id},
        ).execute()
        production_quote_id = result.data
        print(f"✅ Atomic promotion complete: {production_quote_id}")
        return {"success": True, "production_quote_id": production_quote_id}
    except Exception as exc:
        print(f"❌ Promotion failed: {exc}")
        return {"success": False, "error": str(exc)}

@app.function(image=image, secrets=[my_secret], timeout=1800)
def batch_process_episodes(days_back: int = 7):
    """Process all episodes from last N days"""
    from datetime import datetime, timedelta
    start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
    return process_episode_with_ai.remote(start_date=start_date)

@app.function(image=image, secrets=[my_secret], timeout=600)
def backfill_processing_costs():
    """Backfill cost calculation for old episodes"""
    print("💰 Backfilling costs...")
    # Mock implementation
    return {"status": "completed", "updated": 0}

@app.function(image=image, secrets=[my_secret], timeout=120)
def health_check():
    """Verify that the deployed image, required secrets, and database are reachable."""
    from supabase import create_client

    required = ('SUPABASE_URL', 'SUPABASE_KEY', 'OPENAI_API_KEY')
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return {"ok": False, "missing_secrets": missing}

    supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    feeds = supabase.table('test_podcast_feeds').select('id').limit(1).execute()
    return {
        "ok": True,
        "database_reachable": True,
        "feed_table_readable": feeds.data is not None,
    }

@app.function(image=image, secrets=[my_secret], timeout=1800)
def trigger_manual_processor(max_episodes: int = 1, days_back: int = 7):
    """Create an auditable job before an operator-initiated CLI run."""
    from supabase import create_client

    bounded_max = max(1, min(max_episodes, 3))
    start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    job = supabase.table("processing_jobs").insert({
        "idempotency_key": f"operator:{uuid.uuid4()}",
        "job_type": "episode_batch",
        "source": "admin",
        "parameters": {
            "max_episodes": bounded_max,
            "start_date": start_date,
            "operator_surface": "modal_cli",
        },
    }).execute()
    job_id = job.data[0]["id"]
    result = process_episode_with_ai.remote(
        start_date=start_date,
        max_episodes=bounded_max,
        job_id=job_id,
    )
    return {"job_id": job_id, **result}

@app.function(image=image, secrets=[my_secret], timeout=600)
def trigger_scheduled_processor():
    """Legacy trigger for scheduled"""
    return scheduled_processor.remote()

def fetch_curation_examples(supabase) -> str:
    """Load balanced SME examples; never learn from approvals alone."""
    try:
        approved_res = (
            supabase.table("test_quotes")
            .select("quote_text, editorial_context, ranking_reason")
            .in_("approval_status", ["approved", "promoted"])
            .eq("used_for_training", True)
            .order("updated_at", desc=True)
            .limit(8)
            .execute()
        )
        rejected_res = (
            supabase.table("test_quotes")
            .select("quote_text, rejection_reason")
            .eq("approval_status", "rejected")
            .order("updated_at", desc=True)
            .limit(8)
            .execute()
        )

        approved = approved_res.data or []
        rejected = rejected_res.data or []
        if not approved and not rejected:
            return ""

        sections = ["SME preference examples. Infer principles; do not copy wording."]
        if approved:
            sections.append("APPROVED HIGH-SIGNAL TAKES:")
            for row in approved:
                sections.append(
                    f"+ {row.get('quote_text', '')}\n"
                    f"  Editorial reason: {row.get('ranking_reason') or 'SME approved'}"
                )
        if rejected:
            sections.append("REJECTED OR GENERIC TAKES:")
            for row in rejected:
                sections.append(
                    f"- {row.get('quote_text', '')}\n"
                    f"  Rejection reason: {row.get('rejection_reason') or 'Low signal or generic'}"
                )
        return "\n".join(sections)
    except Exception as e:
        print(f"⚠️ Failed to fetch balanced curation examples: {e}")
        return ""


def transcribe_audio_in_chunks(temp_path, client, supabase, job_id=None):
    """Transcribe every 20-minute chunk and retain absolute segment offsets."""
    import glob
    import shutil
    import subprocess
    import tempfile

    chunk_dir = tempfile.mkdtemp(prefix="podtakes-transcript-")
    chunk_pattern = os.path.join(chunk_dir, "chunk-%03d.mp3")
    try:
        split_result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", temp_path,
                "-f", "segment", "-segment_time", "1200",
                "-reset_timestamps", "1", "-c", "copy", "-y", chunk_pattern,
            ],
            capture_output=True,
            text=True,
        )
        if split_result.returncode != 0:
            raise RuntimeError(f"Audio chunking failed: {split_result.stderr[-1000:]}")

        chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "chunk-*.mp3")))
        if not chunk_paths:
            raise RuntimeError("Audio chunking produced no files")

        absolute_offset = 0.0
        transcript_parts = []
        absolute_segments = []
        transcript_model = os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")

        for chunk_index, chunk_path in enumerate(chunk_paths):
            update_processing_job(
                supabase,
                job_id,
                "transcribing",
                progress={
                    "transcript_chunk": chunk_index + 1,
                    "transcript_chunks": len(chunk_paths),
                },
            )
            with open(chunk_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=transcript_model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            chunk_text = getattr(transcript, "text", "") or ""
            transcript_parts.append(chunk_text)
            raw_segments = getattr(transcript, "segments", None) or []
            max_end = 0.0
            for raw_segment in raw_segments:
                if isinstance(raw_segment, dict):
                    text = raw_segment.get("text", "")
                    start = float(raw_segment.get("start", 0))
                    end = float(raw_segment.get("end", start))
                else:
                    text = getattr(raw_segment, "text", "")
                    start = float(getattr(raw_segment, "start", 0))
                    end = float(getattr(raw_segment, "end", start))
                max_end = max(max_end, end)
                absolute_segments.append({
                    "id": len(absolute_segments),
                    "text": text.strip(),
                    "start": round(start + absolute_offset, 3),
                    "end": round(end + absolute_offset, 3),
                    "chunk_index": chunk_index,
                })

            if max_end <= 0:
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", chunk_path,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                max_end = float(probe.stdout.strip())
            absolute_offset += max_end

        return {
            "text": "\n".join(transcript_parts).strip(),
            "segments": absolute_segments,
            "model": transcript_model,
        }
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


def build_extraction_chunks(segments, max_chars=18000, overlap_segments=3):
    """Create complete, overlapping chunks while preserving global segment IDs."""
    chunks = []
    current_lines = []
    current_size = 0
    for segment in segments:
        line = f"[{segment['id']}] {segment['text']}"
        if current_lines and current_size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current_lines))
            current_lines = current_lines[-overlap_segments:]
            current_size = sum(len(value) + 1 for value in current_lines)
        current_lines.append(line)
        current_size += len(line) + 1
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def deduplicate_candidates(candidates):
    """Remove overlapping or near-identical extraction candidates deterministically."""
    from difflib import SequenceMatcher

    ordered = sorted(
        candidates,
        key=lambda q: (
            float(q.get("domain_specificity", 0)),
            float(q.get("novelty", 0)),
            float(q.get("provocation", 0)),
            float(q.get("evidence_quality", 0)),
        ),
        reverse=True,
    )
    kept = []
    for candidate in ordered:
        normalized = normalize_text(candidate.get("text", ""))
        if not normalized:
            continue
        duplicate = False
        for existing in kept:
            similarity = SequenceMatcher(
                None,
                normalized,
                normalize_text(existing.get("text", "")),
            ).ratio()
            overlaps = not (
                int(candidate.get("end_segment_id", -1)) < int(existing.get("start_segment_id", -1))
                or int(candidate.get("start_segment_id", -1)) > int(existing.get("end_segment_id", -1))
            )
            if similarity >= 0.88 or (overlaps and similarity >= 0.72):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def context_evidence_is_source_bounded(evidence_items, start_segment, end_segment):
    """Direct evidence must point inside the exact transcript span supporting the take."""
    for evidence in evidence_items:
        if evidence.get("evidence_type") != "direct_transcript":
            continue
        evidence_segments = evidence.get("segment_ids") or []
        if not evidence_segments or any(
            int(segment_id) < int(start_segment) or int(segment_id) > int(end_segment)
            for segment_id in evidence_segments
        ):
            return False
    return True


def process_single_episode_logic(episode, feed, client, supabase, job_id=None):
    """Refactored logic for processing a single episode"""
    import subprocess
    import tempfile
    import time
    
    # Balanced preference context from SME approvals and rejections.
    curation_examples = fetch_curation_examples(supabase)
    if curation_examples:
        print("  🧠 Loaded balanced SME preference examples")

    
    try:
        # Extract YouTube ID (Fixed Regex + Search Scope)
        print("🔍 Searching for YouTube ID...")
        search_sources = []
        search_sources.append(f"Summary ({len(getattr(episode, 'summary', ''))} chars)")
        search_sources.append(f"Description ({len(getattr(episode, 'description', ''))} chars)")
        
        search_text = getattr(episode, 'summary', '') + " " + getattr(episode, 'description', '')
        
        if hasattr(episode, 'content'):
            for i, content in enumerate(episode.content):
                search_text += " " + content.value
                search_sources.append(f"Content[{i}] ({len(content.value)} chars)")
        
        # Added: Check main link and links array
        link_val = getattr(episode, 'link', '')
        search_text += " " + link_val
        search_sources.append(f"Link ({len(link_val)} chars)")
        
        if hasattr(episode, 'links'):
             for i, link in enumerate(episode.links):
                 href = getattr(link, 'href', '')
                 search_text += " " + href
                 search_sources.append(f"Links[{i}] ({len(href)} chars)")
                 
        print(f"  📄 Search Sources: {', '.join(search_sources)}")
        print(f"  📄 Total Search Text Length: {len(search_text)} chars")
                
        youtube_id = extract_youtube_id(search_text)
        if youtube_id:
            print(f"📺 FOUND YouTube ID (from RSS text): {youtube_id}")
        else:
            print("❌ No YouTube ID found in RSS. Initiating fallback search...")
            # Try searching with just the episode title first (often more successful for long titles)
            youtube_id = search_youtube_for_episode(episode.title)
            
            if not youtube_id:
                print("  ⚠️ Search with title only failed. Trying [Podcast Name] + [Episode Title]...")
                youtube_id = search_youtube_for_episode(f"{feed['name']} {episode.title}")
                
            if youtube_id:
                print(f"📺 FOUND YouTube ID (via search): {youtube_id}")
            else:
                print("❌ No matching full-length YouTube video found in search.")
        
        # Get audio URL
        audio_url = episode.enclosures[0].get('href') if episode.enclosures else None
        if not audio_url:
            return {"episode": episode.title, "error": "No audio URL"}
        
        print(f"🎙️ Processing: {episode.title}")
        
        # Download FULL episode
        temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_path = temp_audio.name
        temp_audio.close()
        
        print("⬇️ Downloading full episode audio...")
        cmd = [
            'ffmpeg', '-v', 'error', '-reconnect', '1',
            '-reconnect_streamed', '1', '-reconnect_delay_max', '5',
            '-i', audio_url,
            '-acodec', 'mp3',
            '-ar', '16000',
            '-ac', '1',
            '-y', temp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Full episode download failed: {result.stderr[-1000:]}")
        
        # Use the media duration, not compressed file size, for duration/cost.
        probe = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', temp_path
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration_minutes = float(probe.stdout.strip()) / 60
        processing_cost = duration_minutes * 0.006 
        
        print(f"📊 Episode duration: ~{duration_minutes:.1f} minutes")
        print(f"💰 Estimated cost: ${duration_minutes * 0.006:.2f}")
        
        episode_guid = getattr(episode, 'id', None) or hashlib.sha256(
            f"{feed['name']}|{episode.title}|{audio_url}".encode("utf-8")
        ).hexdigest()

        # Transcribe every bounded audio chunk and preserve absolute timestamps.
        print("🎤 Transcribing complete episode in bounded chunks...")
        transcription = transcribe_audio_in_chunks(
            temp_path,
            client,
            supabase,
            job_id=job_id,
        )
        transcript_text = transcription["text"]
        segments = transcription["segments"]
        print(f"✅ Transcription complete: {len(transcript_text)} characters, {len(segments)} segments")

        artifact_payload = {
            "processing_job_id": job_id,
            "episode_guid": episode_guid,
            "podcast_name": feed["name"],
            "episode_name": episode.title,
            "source_audio_url": audio_url,
            "transcript_text": transcript_text,
            "transcript_segments": segments,
            "transcript_model": transcription["model"],
            "transcript_duration_seconds": round(duration_minutes * 60, 3),
            "transcription_cost_usd": round(processing_cost, 4),
            "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
            "ranking_prompt_version": RANKING_PROMPT_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "artifact_status": "complete",
            "updated_at": utcnow_iso(),
        }
        try:
            supabase.table("episode_processing_artifacts").upsert(
                artifact_payload,
                on_conflict="episode_guid,pipeline_version",
            ).execute()
        except Exception as exc:
            print(f"AUDIT_WARNING transcript artifact persistence failed: {exc}")

        # Generate candidates from every transcript chunk; each chunk may abstain.
        print("🧠 Extracting source-grounded candidates from the complete transcript...")
        formatted_chunks = build_extraction_chunks(segments)
        all_candidates = []
        update_processing_job(
            supabase,
            job_id,
            "extracting",
            progress={"extraction_chunks": len(formatted_chunks), "extraction_chunk": 0},
        )

        for chunk_index, chunk_text in enumerate(formatted_chunks):
            time.sleep(0.25)
            update_processing_job(
                supabase,
                job_id,
                "extracting",
                progress={
                    "extraction_chunks": len(formatted_chunks),
                    "extraction_chunk": chunk_index + 1,
                },
            )
            candidates = extract_quotes(
                chunk_text,
                feed['name'],
                episode.title,
                client,
                chunk_num=chunk_index + 1,
                curation_examples=curation_examples,
            )

            for candidate in candidates:
                try:
                    start_id = int(candidate.get("start_segment_id"))
                    end_id = int(candidate.get("end_segment_id"))
                    if start_id < 0 or end_id < start_id or end_id >= len(segments):
                        raise ValueError("segment range outside transcript")

                    source_excerpt = " ".join(
                        segment["text"] for segment in segments[start_id:end_id + 1]
                    ).strip()
                    candidate_text = candidate.get("text", "").strip()
                    source_normalized = normalize_text(source_excerpt)
                    candidate_normalized = normalize_text(candidate_text)
                    if not candidate_normalized or candidate_normalized not in source_normalized:
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(
                            None,
                            candidate_normalized,
                            source_normalized,
                        ).ratio()
                        if similarity < 0.62:
                            raise ValueError(f"quote not grounded in source (similarity={similarity:.2f})")

                    start_time = max(0, float(segments[start_id]["start"]) - 0.5)
                    end_time = float(segments[end_id]["end"]) + 0.5
                    if end_time - start_time < 15:
                        end_time = start_time + 15

                    candidate.update({
                        "clip_start": int(start_time),
                        "clip_end": int(end_time),
                        "clip_duration": int(end_time - start_time),
                        "start_seg": start_id,
                        "end_seg": end_id,
                        "source_transcript_excerpt": source_excerpt,
                    })
                    all_candidates.append(candidate)
                except Exception as exc:
                    print(f"⚠️ Rejected ungrounded candidate: {exc}")

        all_candidates = deduplicate_candidates(all_candidates)
        print(f"🔎 {len(all_candidates)} unique, transcript-grounded candidates")

        update_processing_job(
            supabase,
            job_id,
            "ranking",
            progress={"grounded_candidates": len(all_candidates)},
        )
        all_quotes = rank_and_contextualize_quotes(
            all_candidates[:20],
            feed['name'],
            episode.title,
            client,
            curation_examples=curation_examples,
        )[:8]
        
        print(f"💎 Extracted {len(all_quotes)} high-quality takes")
        
        if len(all_quotes) < 3:
            print("⚠️ Episode had fewer than 3 exceptional quotes")
        
        # Parse Correct Date
        try:
            date_published = datetime(*episode.published_parsed[:6]).isoformat()
        except:
            date_published = datetime.now().isoformat()
            
        # Save to database. The UI sums cost per quote, so allocate the episode
        # cost across its quote rows instead of repeating the full cost.
        saved = []
        per_quote_cost = processing_cost / max(len(all_quotes), 1)
        candidate_set_id = str(uuid.uuid4())
        for i, quote in enumerate(all_quotes):
            whisper_start = int(quote.get('clip_start', i * 60))
            whisper_end = int(quote.get('clip_end', (i + 1) * 60))

            # ── YouTube Caption Timestamp Alignment ──────────────────────────
            # Try to replace Whisper timestamps with YouTube-native ones.
            # Falls back silently to Whisper if captions are unavailable or
            # the match confidence is below threshold.
            yt_alignment = None
            if youtube_id:
                yt_alignment = align_timestamps_to_youtube_captions(
                    quote['text'], youtube_id, whisper_start, whisper_end
                )
            
            final_start = yt_alignment['start'] if yt_alignment else whisper_start
            final_end   = yt_alignment['end']   if yt_alignment else whisper_end
            yt_confidence = yt_alignment['confidence'] if yt_alignment else None
            # ─────────────────────────────────────────────────────────────────

            record = {
                'podcast_name': feed['name'],
                'episode_name': episode.title[:100],
                'speaker_name': quote.get('speaker', 'Unknown'),
                'category': quote.get('category', 'Other'),
                'quote_text': quote['text'],
                'date_published': date_published,
                'audio_clip_url': audio_url,
                'episode_audio_url': audio_url,
                'timestamp_start': final_start,
                'timestamp_end': final_end,
                'approval_status': 'pending',
                'test_run': True,
                'youtube_id': youtube_id,
                'duration_minutes': round(duration_minutes, 1),
                'processing_cost': round(per_quote_cost, 4),
                'episode_guid': episode_guid,
                'quality_score': round(quote.get('quality_score', 0.0), 3),
                'extraction_model': quote.get(
                    'extraction_model',
                    os.environ.get('OPENAI_EDITORIAL_MODEL', 'gpt-5.6-sol'),
                ),
                'yt_timestamp_confidence': yt_confidence, # NULL if failed, signals local bridge
                'processing_job_id': job_id,
                'candidate_fingerprint': hashlib.sha256(
                    f"{episode_guid}|{normalize_text(quote['text'])}".encode("utf-8")
                ).hexdigest(),
                'candidate_set_id': candidate_set_id,
                'candidate_rank': i + 1,
                'ranking_reason': quote.get('ranking_reason'),
                'pipeline_version': PIPELINE_VERSION,
                'extraction_prompt_version': EXTRACTION_PROMPT_VERSION,
                'ranking_prompt_version': RANKING_PROMPT_VERSION,
                'original_quote_text': quote['text'],
                'source_transcript_excerpt': quote.get('source_transcript_excerpt'),
                'source_start_segment': quote.get('start_seg'),
                'source_end_segment': quote.get('end_seg'),
                'editorial_context': quote.get('editorial_context'),
                'context_evidence': quote.get('context_evidence', []),
                'context_confidence': quote.get('context_confidence'),
                'context_model': os.environ.get('OPENAI_EDITORIAL_MODEL', 'gpt-5.6-sol'),
                'context_prompt_version': CONTEXT_PROMPT_VERSION,
                # Context is never public until an SME approves it explicitly.
                'context_review_status': 'unreviewed',
            }
            
            print(f"🚀 Attempting to save quote to Supabase: {quote['text'][:50]}...")
            try:
                db_res = supabase.table('test_quotes').insert(record).execute()
                if db_res.data:
                    print(f"✅ Saved successfully: ID {db_res.data[0]['id']}")
                    saved.append(quote['text'][:80])
                else:
                    print(f"⚠️ Insert failed (no data returned): {db_res}")
            except Exception as e:
                if "23505" in str(e) or "duplicate key" in str(e).lower():
                    print("⏭️ Candidate already staged; idempotent retry skipped")
                else:
                    print(f"❌ Supabase Insert Error: {e}")
        
        os.remove(temp_path)
        
        return {
            "episode": episode.title,
            "quotes": len(saved),
            "youtube_id": youtube_id,
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Error processing {episode.title}: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return {"episode": episode.title, "error": str(e)}

def call_openai_structured(
    client,
    *,
    model,
    system_prompt,
    user_prompt,
    schema_name,
    schema,
    reasoning_effort,
    max_output_tokens=6000,
):
    """Call the Responses API with a strict, versioned output contract."""
    import time

    max_retries = 4
    base_delay = 4
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                reasoning={"effort": reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                    "verbosity": "low",
                },
                max_output_tokens=max_output_tokens,
                store=False,
                metadata={
                    "pipeline_version": PIPELINE_VERSION,
                    "schema_name": schema_name,
                },
            )
            if getattr(response, "status", None) == "incomplete":
                details = getattr(response, "incomplete_details", None)
                raise RuntimeError(f"OpenAI response incomplete: {details}")
            output_text = getattr(response, "output_text", "")
            if not output_text:
                raise RuntimeError("OpenAI returned no structured output")
            return json.loads(output_text)
        except Exception as exc:
            retryable = any(
                marker in str(exc).lower()
                for marker in ("rate_limit", "429", "timeout", "temporarily", "500", "502", "503")
            )
            if retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️ OpenAI transient error; retrying in {delay}s: {exc}")
                time.sleep(delay)
                continue
            raise


def extract_quotes(text, podcast, episode, client, chunk_num=0, curation_examples=""):
    """Generate zero to three literal, transcript-grounded candidates per chunk."""
    model = os.environ.get("OPENAI_CANDIDATE_MODEL", "gpt-5.6-terra")
    reasoning_effort = os.environ.get("OPENAI_CANDIDATE_REASONING", "low")
    candidate_schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "start_segment_id": {"type": "integer"},
                        "end_segment_id": {"type": "integer"},
                        "speaker": {"type": "string"},
                        "category": {"type": "string"},
                        "specific_claim": {"type": "string"},
                        "consensus_challenged": {"type": "string"},
                        "causal_mechanism": {"type": "string"},
                        "novelty": {"type": "number"},
                        "provocation": {"type": "number"},
                        "domain_specificity": {"type": "number"},
                        "evidence_quality": {"type": "number"},
                        "genericness_risk": {"type": "number"},
                        "extraction_reason": {"type": "string"},
                    },
                    "required": [
                        "text", "start_segment_id", "end_segment_id", "speaker",
                        "category", "specific_claim", "consensus_challenged",
                        "causal_mechanism", "novelty", "provocation",
                        "domain_specificity", "evidence_quality", "genericness_risk",
                        "extraction_reason"
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }

    user_prompt = f"""
Podcast: {podcast}
Episode: {episode}
Transcript section: {chunk_num}

Select zero to three candidate takes from the transcript below. Zero is the
correct answer when this section contains no genuinely high-signal take.

Hard requirements:
- `text` must be copied verbatim from contiguous transcript segments.
- Segment IDs must exactly bound the quoted source.
- Prefer a specific prediction, causal claim, economic tradeoff, market-structure
  argument, counter-position, or reusable framework.
- A candidate should matter to an adtech operator, publisher, marketer, agency,
  platform, investor, or regulator because it changes a decision or assumption.
- Penalize vague futurism, slogans, product pitches, biography, scene-setting,
  summaries, and advice a smart generalist could give in any industry.
- Scores are numbers from 0 to 1. `genericness_risk` is higher when the take is
  interchangeable with generic business or AI commentary.
- Do not manufacture controversy. Do not rewrite or improve the speaker's words.

{curation_examples}

TRANSCRIPT WITH GLOBAL SEGMENT IDS:
{text}
"""
    system_prompt = """
You are the candidate-retrieval layer for PodTakes. You understand adtech market
structure and terminology, but this step is extractive, not generative. Recall
matters, yet literal source fidelity is mandatory. Abstain instead of filling a
quota. Return only candidates that a senior industry editor would plausibly
consider; final judgment happens in a separate SME-ranking stage.
"""
    data = call_openai_structured(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="podtakes_candidate_set",
        schema=candidate_schema,
        reasoning_effort=reasoning_effort,
        max_output_tokens=5000,
    )
    candidates = data.get("candidates", [])[:3]
    for candidate in candidates:
        for key in (
            "novelty", "provocation", "domain_specificity",
            "evidence_quality", "genericness_risk",
        ):
            candidate[key] = max(0.0, min(1.0, float(candidate.get(key, 0))))
        candidate["extraction_model"] = model
    return candidates


def rank_and_contextualize_quotes(
    candidates,
    podcast,
    episode,
    client,
    curation_examples="",
):
    """Apply an adtech-specific editorial rubric and draft evidence-linked context."""
    if not candidates:
        return []

    model = os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol")
    reasoning_effort = os.environ.get("OPENAI_EDITORIAL_REASONING", "high")
    selection_schema = {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {"type": "integer"},
                        "quality_score": {"type": "number"},
                        "ranking_reason": {"type": "string"},
                        "editorial_context": {"type": "string"},
                        "context_confidence": {"type": "number"},
                        "why_it_matters": {"type": "string"},
                        "stakeholders": {"type": "array", "items": {"type": "string"}},
                        "counterpoint": {"type": "string"},
                        "genericness_check": {"type": "string", "enum": ["pass", "fail"]},
                        "context_evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim": {"type": "string"},
                                    "support": {"type": "string"},
                                    "evidence_type": {
                                        "type": "string",
                                        "enum": [
                                            "direct_transcript",
                                            "domain_inference",
                                            "editorial_judgment"
                                        ],
                                    },
                                    "segment_ids": {"type": "array", "items": {"type": "integer"}},
                                },
                                "required": ["claim", "support", "evidence_type", "segment_ids"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "candidate_index", "quality_score", "ranking_reason",
                        "editorial_context", "context_confidence", "why_it_matters",
                        "stakeholders", "counterpoint", "genericness_check",
                        "context_evidence"
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["selections"],
        "additionalProperties": False,
    }

    compact_candidates = []
    for index, candidate in enumerate(candidates):
        compact_candidates.append({
            "candidate_index": index,
            "quote": candidate.get("text"),
            "speaker": candidate.get("speaker"),
            "category": candidate.get("category"),
            "specific_claim": candidate.get("specific_claim"),
            "consensus_challenged": candidate.get("consensus_challenged"),
            "causal_mechanism": candidate.get("causal_mechanism"),
            "source_segment_ids": [candidate.get("start_seg"), candidate.get("end_seg")],
            "source_excerpt": candidate.get("source_transcript_excerpt"),
            "retrieval_scores": {
                key: candidate.get(key)
                for key in (
                    "novelty", "provocation", "domain_specificity",
                    "evidence_quality", "genericness_risk",
                )
            },
        })

    user_prompt = f"""
Podcast: {podcast}
Episode: {episode}

Rank up to eight takes. Select none when the candidates do not clear the bar.

Editorial standard:
1. The take makes a specific claim and exposes a real causal mechanism,
   incentive, tradeoff, prediction, or non-obvious market implication.
2. The analysis demonstrates adtech fluency where relevant: auction mechanics,
   identity/addressability, measurement and incrementality, privacy, supply-path
   economics, publisher monetization, agency/brand incentives, CTV, retail media,
   walled gardens, or AI's effect on media and advertising economics.
3. `editorial_context` must explain why this exact take matters in 60-110 words.
   Name the mechanism, affected stakeholder, and practical tension. Do not merely
   paraphrase the quote.
4. Distinguish transcript facts from domain inference in `context_evidence`.
   Never invent a company fact, market statistic, event, or speaker intent.
5. `genericness_check` must be `fail` if the analysis could be attached to an
   unrelated business quote with only noun substitutions.
6. Avoid generic AI prose such as "in today's rapidly evolving landscape",
   "underscores the importance", "game changer", or "businesses must adapt".
7. Acknowledge the strongest reasonable counterpoint rather than presenting
   provocation as settled fact.
8. Scores are from 0 to 1. Reserve 0.90+ for unusually specific, consequential,
   source-grounded insight.

{curation_examples}

CANDIDATES:
{json.dumps(compact_candidates, ensure_ascii=False)}
"""
    system_prompt = """
You are PodTakes' senior industry editor. Your standard is an expert adtech
publication, not an AI summary product. Your job is to identify decision-relevant
insight and draft rigorous context for SME review. Do not optimize for quantity,
engagement bait, or superficial controversy. Treat every factual statement as a
claim that needs either direct transcript support or an explicit inference label.
"""
    data = call_openai_structured(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="podtakes_editorial_selection",
        schema=selection_schema,
        reasoning_effort=reasoning_effort,
        max_output_tokens=9000,
    )

    minimum_quality = float(os.environ.get("MIN_EDITORIAL_QUALITY", "0.78"))
    minimum_context_confidence = float(os.environ.get("MIN_CONTEXT_CONFIDENCE", "0.72"))
    selected = []
    used_indices = set()
    for selection in data.get("selections", []):
        index = int(selection.get("candidate_index", -1))
        if index < 0 or index >= len(candidates) or index in used_indices:
            continue
        quality = max(0.0, min(1.0, float(selection.get("quality_score", 0))))
        confidence = max(0.0, min(1.0, float(selection.get("context_confidence", 0))))
        if quality < minimum_quality:
            continue
        if confidence < minimum_context_confidence:
            continue
        if selection.get("genericness_check") != "pass":
            continue
        editorial_context = str(selection.get("editorial_context", "")).strip()
        if len(editorial_context.split()) < 35:
            continue
        evidence_items = selection.get("context_evidence", [])
        candidate_start = int(candidates[index].get("start_seg", -1))
        candidate_end = int(candidates[index].get("end_seg", -1))
        if not context_evidence_is_source_bounded(
            evidence_items,
            candidate_start,
            candidate_end,
        ):
            continue

        candidate = dict(candidates[index])
        candidate.update({
            "quality_score": quality,
            "ranking_reason": selection.get("ranking_reason"),
            "editorial_context": editorial_context,
            "context_confidence": confidence,
            "context_evidence": evidence_items,
            "why_it_matters": selection.get("why_it_matters"),
            "stakeholders": selection.get("stakeholders", []),
            "counterpoint": selection.get("counterpoint"),
            "extraction_model": model,
        })
        selected.append(candidate)
        used_indices.add(index)

    selected.sort(key=lambda item: item.get("quality_score", 0), reverse=True)
    return selected


@app.function(image=image, secrets=[my_secret], timeout=1800, cpu=2)
def run_editorial_evaluation(sample_limit: int = 40, job_id: str = None):
    """Evaluate the active editorial gate against balanced, source-backed SME decisions."""
    from openai import OpenAI
    from supabase import create_client

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    update_processing_job(
        supabase,
        job_id,
        "claimed",
        claimed_at=utcnow_iso(),
        started_at=utcnow_iso(),
        progress={"phase": "loading_evaluation_set"},
    )
    try:
        rows_result = (
            supabase.table("test_quotes")
            .select(
                "id,approval_status,quote_text,speaker_name,category,"
                "source_transcript_excerpt,source_start_segment,source_end_segment"
            )
            .in_("approval_status", ["approved", "promoted", "rejected"])
            .order("updated_at", desc=True)
            .limit(400)
            .execute()
        )
        source_backed = [
            row for row in (rows_result.data or [])
            if row.get("source_transcript_excerpt")
            and row.get("source_start_segment") is not None
            and row.get("source_end_segment") is not None
        ]
        positives = [row for row in source_backed if row["approval_status"] in ("approved", "promoted")]
        negatives = [row for row in source_backed if row["approval_status"] == "rejected"]
        per_class = min(max(1, sample_limit // 2), len(positives), len(negatives))
        if per_class < 6:
            raise RuntimeError(
                "Evaluation requires at least six approved and six rejected source-backed decisions"
            )
        evaluation_rows = positives[:per_class] + negatives[:per_class]

        model = os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol")
        model_version_id = f"{model}:{RANKING_PROMPT_VERSION}:{PIPELINE_VERSION}"
        supabase.table("model_versions").upsert({
            "id": model_version_id,
            "component": "ranking",
            "provider": "openai",
            "model_name": model,
            "prompt_version": RANKING_PROMPT_VERSION,
            "rubric_version": "sme-rubric-v1",
            "status": "active",
            "configuration": {
                "reasoning_effort": os.environ.get("OPENAI_EDITORIAL_REASONING", "high"),
                "minimum_quality": float(os.environ.get("MIN_EDITORIAL_QUALITY", "0.78")),
                "minimum_context_confidence": float(os.environ.get("MIN_CONTEXT_CONFIDENCE", "0.72")),
            },
            "deployed_at": utcnow_iso(),
        }, on_conflict="id").execute()

        thresholds = {
            "precision": 0.75,
            "positive_recall": 0.60,
            "negative_exclusion": 0.75,
        }
        run_insert = supabase.table("model_evaluation_runs").insert({
            "model_version_id": model_version_id,
            "dataset_version": f"source-backed-curation:{datetime.now(timezone.utc).date().isoformat()}",
            "status": "running",
            "thresholds": thresholds,
            "notes": "Balanced recent SME decisions; evaluation prompt receives no labeled examples.",
        }).execute()
        run_id = run_insert.data[0]["id"]

        predictions = {}
        for start in range(0, len(evaluation_rows), 20):
            batch_rows = evaluation_rows[start:start + 20]
            candidates = []
            for row in batch_rows:
                candidates.append({
                    "id": row["id"],
                    "text": row["quote_text"],
                    "speaker": row.get("speaker_name") or "Unknown",
                    "category": row.get("category") or "Other",
                    "specific_claim": "Held-out candidate for evaluation",
                    "consensus_challenged": "Assess from quote and source only",
                    "causal_mechanism": "Assess from quote and source only",
                    "source_transcript_excerpt": row["source_transcript_excerpt"],
                    "start_seg": row["source_start_segment"],
                    "end_seg": row["source_end_segment"],
                    "novelty": 0.5,
                    "provocation": 0.5,
                    "domain_specificity": 0.5,
                    "evidence_quality": 0.5,
                    "genericness_risk": 0.5,
                })
            selections = rank_and_contextualize_quotes(
                candidates,
                "Held-out evaluation set",
                "Mixed source-backed SME decisions",
                client,
                curation_examples="",
            )
            for rank, selection in enumerate(selections, start=1):
                predictions[selection["id"]] = {
                    "score": selection.get("quality_score", 0),
                    "rank": rank,
                    "ranking_reason": selection.get("ranking_reason"),
                }
            update_processing_job(
                supabase,
                job_id,
                "ranking",
                progress={"evaluated": min(start + 20, len(evaluation_rows)), "total": len(evaluation_rows)},
            )

        true_positives = sum(row["id"] in predictions for row in positives[:per_class])
        false_negatives = per_class - true_positives
        false_positives = sum(row["id"] in predictions for row in negatives[:per_class])
        true_negatives = per_class - false_positives
        precision = true_positives / max(true_positives + false_positives, 1)
        positive_recall = true_positives / max(true_positives + false_negatives, 1)
        negative_exclusion = true_negatives / max(true_negatives + false_positives, 1)
        metrics = {
            "sample_size": per_class * 2,
            "positive_count": per_class,
            "negative_count": per_class,
            "precision": round(precision, 4),
            "positive_recall": round(positive_recall, 4),
            "negative_exclusion": round(negative_exclusion, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        }
        passed = all(metrics[name] >= threshold for name, threshold in thresholds.items())
        items = []
        for row in evaluation_rows:
            prediction = predictions.get(row["id"])
            expected_positive = row["approval_status"] in ("approved", "promoted")
            predicted_positive = prediction is not None
            items.append({
                "evaluation_run_id": run_id,
                "quote_id": row["id"],
                "expected_decision": "approve" if expected_positive else "reject",
                "predicted_score": prediction.get("score") if prediction else 0,
                "predicted_rank": prediction.get("rank") if prediction else None,
                "passed": expected_positive == predicted_positive,
                "evidence": {"ranking_reason": prediction.get("ranking_reason") if prediction else None},
            })
        supabase.table("model_evaluation_items").insert(items).execute()
        supabase.table("model_evaluation_runs").update({
            "status": "passed" if passed else "failed",
            "metrics": metrics,
            "completed_at": utcnow_iso(),
        }).eq("id", run_id).execute()
        result = {"success": passed, "evaluation_run_id": run_id, "metrics": metrics, "thresholds": thresholds}
        update_processing_job(
            supabase,
            job_id,
            "succeeded" if passed else "failed",
            result=result,
            error_code=None if passed else "evaluation_threshold_failed",
            error_message=None if passed else "Editorial model failed one or more activation thresholds",
            completed_at=utcnow_iso(),
        )
        return result
    except Exception as exc:
        update_processing_job(
            supabase,
            job_id,
            "failed",
            error_code=type(exc).__name__,
            error_message=str(exc)[:4000],
            completed_at=utcnow_iso(),
        )
        raise

def search_youtube_for_episode(query: str) -> str | None:
    """Uses yt-dlp to search YouTube for the episode and pick the best full-length match."""
    import yt_dlp
    
    ydl_opts = {
        'format': 'best',
        'noplaylist': True,
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'default_search': 'ytsearch3'
    }
    
    print(f"  🔍 Searching YouTube for: '{query}'")
    
    # Explicitly force a search so yt-dlp doesn't mistakenly treat dots/patterns in the title as a valid URL
    search_query = f"ytsearch3:{query}"
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_query, download=False)
            if 'entries' in info:
                for entry in info['entries']:
                    duration = entry.get('duration', 0)
                    if not duration: duration = 0
                    
                    # Exclude shorts (< 5 minutes / 300 seconds)
                    if duration > 300:
                        print(f"  ✅ Picked search result: {entry.get('title')[:60]}... ({duration}s)")
                        return entry.get('id')
                    else:
                        print(f"  ⏭️ Ignored short search result: {entry.get('title')[:30]}... ({duration}s)")
        except Exception as e:
            print(f"  ❌ YouTube search failed: {e}")
            
    return None

def extract_youtube_id(text):
    """Extract YouTube ID with improved regex and logging"""
    if not text:
        return None
    
    import re
    # Patterns covering standard, shortened, and embed URLs
    patterns = [
        r'youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',       # Standard
        r'youtu\.be\/([a-zA-Z0-9_-]{11})',                   # Shortened
        r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',         # Embed
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',             # V path
        r'youtube\.com\/.*[?&]v=([a-zA-Z0-9_-]{11})'         # Catch-all query param
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if match:
            print(f"  ✅ Matched Regex #{i+1}: {match.group(0)}")
            return match.group(1)
            
    return None

def find_clip_boundaries_fixed(quote, segments):
    """Find natural sentence boundaries for clips using Robust Fuzzy Matching"""
    import difflib
    
    quote_text = quote['text'].lower()
    search_chunks = [quote_text[:50]] # First 50 chars
    
    # Also try the middle and end if it's long, in case the start was hallucinated/changed
    if len(quote_text) > 100:
        search_chunks.append(quote_text[-50:])
    
    best_overall_score = 0.0
    best_overall_idx = -1
    
    print(f"🔎 Looking for timestamp for: '{quote_text[:30]}...'")
    
    for chunk in search_chunks:
        for i, seg in enumerate(segments):
            seg_text = seg.text if hasattr(seg, 'text') else str(seg)
            # Check similarity
            ratio = difflib.SequenceMatcher(None, chunk, seg_text.lower()).ratio()
            
            if ratio > best_overall_score:
                best_overall_score = ratio
                best_overall_idx = i
                
    # Strict Threshold: 0.65 (0.4-0.5 was producing garbage matches)
    confidence_threshold = 0.65
    
    if best_overall_idx == -1 or best_overall_score < confidence_threshold: 
        print(f"⚠️ LOW CONFIDENCE (Best: {best_overall_score:.2f}). Rejected to avoid bad clip.")
        # Return default 0-60s so prompt engineer knows it failed, or return None?
        # Returning default 0-60 allows manual fixing in UI.
        return {}
    
    matched_seg_text = segments[best_overall_idx].text if hasattr(segments[best_overall_idx], 'text') else str(segments[best_overall_idx])
    print(f"✅ Found fuzzy match (Score: {best_overall_score:.2f}) at segment {best_overall_idx}")
    print(f"   Match: '{matched_seg_text[:50]}...'")
    
    start_idx = max(0, best_overall_idx - 1)
    
    start_time = segments[start_idx].start if hasattr(segments[start_idx], 'start') else 0
    end_time = start_time + 45 # Default duration
    
    return {
        'clip_start': int(start_time),
        'clip_end': int(end_time),
        'clip_duration': int(end_time - start_time)
    }

@app.function(
    image=image,
    secrets=[my_secret],
    timeout=300,
)
def create_audio_clip(quote_id: str):
    """Create audio clip for an approved quote"""
    
    import subprocess
    import tempfile
    from supabase import create_client
    from pydub import AudioSegment
    
    print(f"🎬 Creating audio clip for quote {quote_id}")
    
    supabase = create_client(
        os.environ['SUPABASE_URL'],
        os.environ['SUPABASE_KEY']
    )
    
    result = supabase.table('test_quotes') \
        .select('*') \
        .eq('id', quote_id) \
        .single() \
        .execute()
    
    if not result.data:
        return {"error": f"Quote {quote_id} not found"}
    
    quote = result.data
    
    start_sec = max(0, quote['timestamp_start'] - 10)
    end_sec = quote['timestamp_end'] + 10
    duration = end_sec - start_sec
    
    print(f"📊 Clip duration: {duration} seconds ({start_sec} to {end_sec})")
    
    temp_clip = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    temp_path = temp_clip.name
    temp_clip.close()
    
    print(f"⬇️ Extracting clip from episode...")
    source_audio_url = quote.get('episode_audio_url') or quote.get('audio_clip_url')
    if not source_audio_url:
        return {"error": "Quote has no source episode audio URL"}

    cmd = [
        'ffmpeg', '-i', source_audio_url,
        '-ss', str(start_sec),
        '-t', str(duration),
        '-acodec', 'mp3',
        '-ar', '44100',
        '-ab', '128k',
        '-y', temp_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FFmpeg error: {result.stderr}")
        return {"error": "Failed to create clip"}
    
    print("✨ Adding fade effects...")
    audio = AudioSegment.from_mp3(temp_path)
    audio = audio.fade_in(500).fade_out(500)
    audio.export(temp_path, format="mp3", bitrate="128k")
    
    print("☁️ Uploading to storage...")
    clip_filename = f"clips/{quote_id}.mp3"
    
    with open(temp_path, 'rb') as f:
        upload_result = supabase.storage \
            .from_('audio-clips') \
            .upload(clip_filename, f.read(), {
                'content-type': 'audio/mpeg',
                'upsert': 'true'
            })
    
    if hasattr(upload_result, 'error') and upload_result.error:
        print(f"❌ Upload error: {upload_result.error}")
        return {"error": "Failed to upload clip"}
    
    clip_url = supabase.storage \
        .from_('audio-clips') \
        .get_public_url(clip_filename)
    
    supabase.table('test_quotes') \
        .update({'audio_clip_url': clip_url}) \
        .eq('id', quote_id) \
        .execute()
    
    os.remove(temp_path)
    print(f"✅ Clip created successfully: {clip_url}")
    
    return {
        "success": True,
        "quote_id": quote_id,
        "clip_url": clip_url,
        "duration": duration
    }

@app.function(
    image=image,
    secrets=[my_secret],
    timeout=1800,
    schedule=modal.Cron("0 0 * * *", timezone="UTC"),
)
def scheduled_processor():
    """Process a bounded daily batch when automation is enabled."""
    from supabase import create_client

    started_at = datetime.now(timezone.utc).isoformat()
    supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

    setting = (
        supabase.table('automation_settings')
        .select('value')
        .eq('key', 'automated_processing_enabled')
        .limit(1)
        .execute()
    )
    if setting.data and not setting.data[0].get('value', False):
        print("⏸️ Automated processing is disabled")
        return {"success": True, "status": "disabled", "processed_count": 0}

    log = supabase.table('automation_logs').insert({
        'run_type': 'scheduled',
        'status': 'running',
        'started_at': started_at,
        'episodes_processed': 0,
        'quotes_extracted': 0,
    }).execute()
    log_id = log.data[0]['id'] if log.data else None

    try:
        max_episodes = int(os.environ.get('SCHEDULED_MAX_EPISODES', '2'))
        idempotency_key = f"scheduled:{datetime.now(timezone.utc).date().isoformat()}"
        existing_job = (
            supabase.table("processing_jobs")
            .select("id,state,result")
            .eq("idempotency_key", idempotency_key)
            .limit(1)
            .execute()
        )
        if existing_job.data and existing_job.data[0].get("state") in {
            "queued", "claimed", "downloading", "transcribing", "extracting",
            "ranking", "staging", "succeeded"
        }:
            result = existing_job.data[0].get("result") or {
                "success": True,
                "status": existing_job.data[0].get("state"),
                "processed_count": 0,
            }
        else:
            if existing_job.data:
                job_id = existing_job.data[0]["id"]
                supabase.table("processing_jobs").update({
                    "state": "queued",
                    "attempt_count": int(existing_job.data[0].get("attempt_count", 0)) + 1,
                    "error_code": None,
                    "error_message": None,
                    "completed_at": None,
                    "updated_at": utcnow_iso(),
                }).eq("id", job_id).execute()
            else:
                job_insert = supabase.table("processing_jobs").insert({
                    "idempotency_key": idempotency_key,
                    "job_type": "episode_batch",
                    "source": "scheduled",
                    "parameters": {"max_episodes": max_episodes},
                }).execute()
                job_id = job_insert.data[0]["id"]
            result = process_episode_with_ai.remote(
                max_episodes=max_episodes,
                job_id=job_id,
            )
        episodes_processed = int(result.get('processed_count', 0))
        quotes_extracted = sum(
            int(item.get('quotes', 0))
            for item in result.get('details', [])
            if isinstance(item, dict)
        )
        if log_id:
            supabase.table('automation_logs').update({
                'status': 'success',
                'result': result,
                'episodes_processed': episodes_processed,
                'quotes_extracted': quotes_extracted,
                'completed_at': datetime.now(timezone.utc).isoformat(),
            }).eq('id', log_id).execute()
        print(f"Scheduled run result: {result}")
        return result
    except Exception as exc:
        if log_id:
            supabase.table('automation_logs').update({
                'status': 'failed',
                'error_message': str(exc),
                'completed_at': datetime.now(timezone.utc).isoformat(),
            }).eq('id', log_id).execute()
        raise

# ==========================================
# FastAPI Web Endpoints (Nested to avoid local deps)
# ==========================================
@app.function(
    image=image,
    secrets=[my_secret],
    timeout=1800,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request, Depends, HTTPException, status
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
    from supabase import create_client

    web_app = FastAPI(title="PodTakes Admin API", docs_url=None, redoc_url=None)
    bearer_scheme = HTTPBearer(auto_error=False)

    allowed_origins = [
        value.strip()
        for value in os.environ.get(
            "ALLOWED_ORIGINS",
            "https://podtakes.com,https://www.podtakes.com,http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if value.strip()
    ]
    
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
    )

    class ProcessRequest(BaseModel):
        feed_ids: list[str] | None = None
        start_date: str = None
        end_date: str = None
        max_episodes: int | None = Field(default=None, ge=1, le=10)
        idempotency_key: str | None = Field(default=None, min_length=8, max_length=200)

    class ClipRequest(BaseModel):
        quote_id: str

    class ReviewRequest(BaseModel):
        quote_id: str
        action: str = "approve"
        quote_text: str | None = None
        editorial_context: str | None = None
        speaker_name: str | None = None
        category: str | None = None
        speaker_title: str | None = None
        speaker_company: str | None = None
        speaker_linkedin: str | None = None
        youtube_id: str | None = None
        podcast_name: str | None = None
        episode_name: str | None = None
        timestamp_start: float | None = Field(default=None, ge=0)
        timestamp_end: float | None = Field(default=None, ge=0)
        youtube_offset: float | None = None
        reason_code: str | None = None
        reason_detail: str | None = None
        reviewer_expertise: list[str] = Field(default_factory=list)
        target_decision_id: str | None = None

    class AutomationRequest(BaseModel):
        enabled: bool

    class FeedStateRequest(BaseModel):
        active: bool

    class FeedCreateRequest(BaseModel):
        name: str = Field(min_length=2, max_length=200)
        rss_url: str = Field(min_length=8, max_length=2000)

    class EvaluationRequest(BaseModel):
        sample_limit: int = Field(default=40, ge=12, le=100)

    def service_client():
        return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    async def require_admin(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ):
        if not credentials or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Supabase access token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            auth_result = service_client().auth.get_user(credentials.credentials)
            user = auth_result.user
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired access token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        app_metadata = getattr(user, "app_metadata", None) or {}
        if app_metadata.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator role required",
            )
        return {"id": str(user.id), "email": getattr(user, "email", None)}

    @web_app.get("/health")
    async def health_endpoint():
        return {
            "ok": True,
            "pipeline_version": PIPELINE_VERSION,
            "auth": "supabase-admin-jwt",
        }

    @web_app.post("/process-episode")
    async def process_episode_endpoint(
        req: ProcessRequest,
        request: Request,
        admin=Depends(require_admin),
    ):
        """Queue processing and return immediately with a durable job ID."""
        supabase = service_client()
        idempotency_key = (
            request.headers.get("Idempotency-Key")
            or req.idempotency_key
            or f"manual:{uuid.uuid4()}"
        )
        try:
            existing = (
                supabase.table("processing_jobs")
                .select("id,state,result,error_message,created_at")
                .eq("idempotency_key", idempotency_key)
                .limit(1)
                .execute()
            )
            if existing.data:
                return JSONResponse(
                    status_code=200,
                    content={"success": True, "duplicate": True, "job": existing.data[0]},
                )

            parameters = {
                "feed_ids": req.feed_ids,
                "start_date": req.start_date,
                "end_date": req.end_date,
                "max_episodes": req.max_episodes,
            }
            inserted = supabase.table("processing_jobs").insert({
                "idempotency_key": idempotency_key,
                "job_type": "episode_batch",
                "source": "admin",
                "requested_by": admin["id"],
                "parameters": parameters,
            }).execute()
            job_id = inserted.data[0]["id"]
            function_call = await process_episode_with_ai.spawn.aio(
                feed_ids=req.feed_ids,
                start_date=req.start_date,
                end_date=req.end_date,
                max_episodes=req.max_episodes,
                job_id=job_id,
            )
            supabase.table("processing_jobs").update({
                "modal_call_id": function_call.object_id,
                "updated_at": utcnow_iso(),
            }).eq("id", job_id).execute()
            return JSONResponse(
                status_code=202,
                content={
                    "success": True,
                    "job_id": job_id,
                    "state": "queued",
                    "status_url": f"/jobs/{job_id}",
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})

    @web_app.get("/jobs/{job_id}")
    async def job_status_endpoint(job_id: str, admin=Depends(require_admin)):
        result = (
            service_client().table("processing_jobs")
            .select("id,job_type,source,state,parameters,progress,result,error_code,error_message,attempt_count,queued_at,started_at,completed_at,heartbeat_at")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"success": True, "job": result.data[0]}

    @web_app.post("/evaluations/run")
    async def run_evaluation_endpoint(req: EvaluationRequest, admin=Depends(require_admin)):
        supabase = service_client()
        inserted = supabase.table("processing_jobs").insert({
            "idempotency_key": f"evaluation:{uuid.uuid4()}",
            "job_type": "model_evaluation",
            "source": "admin",
            "requested_by": admin["id"],
            "parameters": {"sample_limit": req.sample_limit},
        }).execute()
        job_id = inserted.data[0]["id"]
        function_call = await run_editorial_evaluation.spawn.aio(
            sample_limit=req.sample_limit,
            job_id=job_id,
        )
        supabase.table("processing_jobs").update({
            "modal_call_id": function_call.object_id,
            "updated_at": utcnow_iso(),
        }).eq("id", job_id).execute()
        return JSONResponse(
            status_code=202,
            content={"success": True, "job_id": job_id, "state": "queued"},
        )

    @web_app.post("/create-clip")
    async def create_clip_endpoint(req: ClipRequest, admin=Depends(require_admin)):
        try:
            result = await create_audio_clip.remote.aio(req.quote_id)
            if not result.get("success"):
                return JSONResponse(status_code=422, content=result)
            return result
        except Exception as exc:
            return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})

    @web_app.post("/review-quote")
    async def review_quote_endpoint(req: ReviewRequest, admin=Depends(require_admin)):
        allowed_actions = {
            "approve", "reject", "edit", "approve_context",
            "reject_context", "undo",
        }
        if req.action not in allowed_actions:
            raise HTTPException(status_code=422, detail="Unsupported review action")
        if req.action in {"approve", "reject", "approve_context", "reject_context"} and not req.reviewer_expertise:
            raise HTTPException(status_code=422, detail="Reviewer expertise is required for an editorial decision")

        supabase = service_client()
        staged_result = (
            supabase.table("test_quotes").select("*").eq("id", req.quote_id).single().execute()
        )
        if not staged_result.data:
            raise HTTPException(status_code=404, detail="Quote not found")
        before = staged_result.data
        updates = {}

        if req.action == "undo":
            if not req.target_decision_id:
                raise HTTPException(status_code=422, detail="target_decision_id is required")
            decision = (
                supabase.table("curation_decisions")
                .select("metadata")
                .eq("id", req.target_decision_id)
                .single()
                .execute()
            )
            restore = ((decision.data or {}).get("metadata") or {}).get("before") or {}
            allowed_restore = {
                "approval_status", "quote_text", "editorial_context",
                "context_review_status", "speaker_name", "category",
                "speaker_title", "speaker_company", "speaker_linkedin",
                "youtube_id", "podcast_name", "episode_name",
                "timestamp_start", "timestamp_end", "youtube_offset",
                "rejection_reason", "editorial_notes",
            }
            updates = {key: value for key, value in restore.items() if key in allowed_restore}
            if not updates:
                raise HTTPException(status_code=422, detail="Decision has no restorable state")
        else:
            editable_values = {
                "quote_text": req.quote_text,
                "editorial_context": req.editorial_context,
                "speaker_name": req.speaker_name,
                "category": req.category,
                "speaker_title": req.speaker_title,
                "speaker_company": req.speaker_company,
                "speaker_linkedin": req.speaker_linkedin,
                "youtube_id": req.youtube_id,
                "podcast_name": req.podcast_name,
                "episode_name": req.episode_name,
                "timestamp_start": req.timestamp_start,
                "timestamp_end": req.timestamp_end,
                "youtube_offset": req.youtube_offset,
            }
            updates.update({
                key: value.strip() if isinstance(value, str) else value
                for key, value in editable_values.items()
                if value is not None
            })
            next_start = updates.get("timestamp_start", before.get("timestamp_start"))
            next_end = updates.get("timestamp_end", before.get("timestamp_end"))
            if next_start is not None and next_end is not None and next_end <= next_start:
                raise HTTPException(status_code=422, detail="Quote end time must be after start time")
            if req.action == "approve":
                updates["approval_status"] = "approved"
                updates["rejection_reason"] = None
            elif req.action == "reject":
                if not req.reason_code and not req.reason_detail:
                    raise HTTPException(status_code=422, detail="A rejection reason is required")
                updates["approval_status"] = "rejected"
                updates["rejection_reason"] = req.reason_detail or req.reason_code
            elif req.action == "approve_context":
                context_value = updates.get("editorial_context", before.get("editorial_context"))
                if not context_value or len(context_value.split()) < 25:
                    raise HTTPException(status_code=422, detail="Context is too thin for SME approval")
                updates.update({
                    "context_review_status": "approved",
                    "context_reviewed_by": admin["id"],
                    "context_reviewed_at": utcnow_iso(),
                })
            elif req.action == "reject_context":
                if not req.reason_code and not req.reason_detail:
                    raise HTTPException(status_code=422, detail="A context rejection reason is required")
                updates["context_review_status"] = "rejected"

        updates["updated_at"] = utcnow_iso()
        updated = supabase.table("test_quotes").update(updates).eq("id", req.quote_id).execute()
        after = updated.data[0] if updated.data else {**before, **updates}
        audit_metadata = {
            "before": {
                key: before.get(key)
                for key in (
                    "approval_status", "quote_text", "editorial_context",
                    "context_review_status", "speaker_name", "category",
                    "speaker_title", "speaker_company", "speaker_linkedin",
                    "youtube_id", "podcast_name", "episode_name",
                    "timestamp_start", "timestamp_end", "youtube_offset",
                    "rejection_reason", "editorial_notes",
                )
            },
            "after": updates,
            "target_decision_id": req.target_decision_id,
        }
        decision = supabase.table("curation_decisions").insert({
            "quote_id": req.quote_id,
            "reviewer_id": admin["id"],
            "decision": req.action,
            "original_quote_text": before.get("quote_text"),
            "edited_quote_text": after.get("quote_text"),
            "original_context": before.get("editorial_context"),
            "edited_context": after.get("editorial_context"),
            "reason_code": req.reason_code,
            "reason_detail": req.reason_detail,
            "reviewer_expertise": req.reviewer_expertise,
            "candidate_rank": before.get("candidate_rank"),
            "candidate_set_id": before.get("candidate_set_id"),
            "model_version": before.get("extraction_model"),
            "prompt_version": before.get("context_prompt_version"),
            "metadata": audit_metadata,
        }).execute()
        return {
            "success": True,
            "quote": after,
            "decision_id": decision.data[0]["id"] if decision.data else None,
        }

    @web_app.post("/approve-quote")
    async def approve_quote_compat(req: ReviewRequest, admin=Depends(require_admin)):
        req.action = "approve"
        return await review_quote_endpoint(req, admin)

    @web_app.post("/promote-quote")
    async def promote_quote_endpoint(req: ClipRequest, admin=Depends(require_admin)):
        try:
            result = await promote_quote_to_production.remote.aio(req.quote_id, admin["id"])
            if not result.get("success"):
                return JSONResponse(status_code=422, content=result)
            return result
        except Exception as exc:
            return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})

    @web_app.post("/automation")
    async def automation_endpoint(req: AutomationRequest, admin=Depends(require_admin)):
        supabase = service_client()
        updated = (
            supabase.table("automation_settings")
            .update({"value": req.enabled, "updated_at": utcnow_iso()})
            .eq("key", "automated_processing_enabled")
            .execute()
        )
        supabase.table("automation_logs").insert({
            "run_type": "configuration",
            "status": "success",
            "result": {"enabled": req.enabled, "changed_by": admin["id"]},
            "completed_at": utcnow_iso(),
        }).execute()
        return {"success": True, "enabled": req.enabled, "data": updated.data}

    @web_app.post("/feeds/{feed_id}/state")
    async def feed_state_endpoint(
        feed_id: str,
        req: FeedStateRequest,
        admin=Depends(require_admin),
    ):
        updated = (
            service_client().table("test_podcast_feeds")
            .update({"active": req.active})
            .eq("id", feed_id)
            .execute()
        )
        if not updated.data:
            raise HTTPException(status_code=404, detail="Feed not found")
        service_client().table("automation_logs").insert({
            "run_type": "feed_configuration",
            "status": "success",
            "result": {
                "action": "set_state",
                "feed_id": feed_id,
                "active": req.active,
                "changed_by": admin["id"],
            },
            "completed_at": utcnow_iso(),
        }).execute()
        return {"success": True, "feed": updated.data[0]}

    @web_app.post("/feeds")
    async def create_feed_endpoint(req: FeedCreateRequest, admin=Depends(require_admin)):
        if not req.rss_url.lower().startswith(("https://", "http://")):
            raise HTTPException(status_code=422, detail="RSS feed URL must use http or https")
        supabase = service_client()
        existing = (
            supabase.table("test_podcast_feeds")
            .select("id")
            .eq("rss_url", req.rss_url.strip())
            .limit(1)
            .execute()
        )
        if existing.data:
            raise HTTPException(status_code=409, detail="This RSS feed is already configured")
        inserted = supabase.table("test_podcast_feeds").insert({
            "name": req.name.strip(),
            "rss_url": req.rss_url.strip(),
            "active": True,
        }).execute()
        feed = inserted.data[0]
        supabase.table("automation_logs").insert({
            "run_type": "feed_configuration",
            "status": "success",
            "result": {"action": "create", "feed_id": feed["id"], "changed_by": admin["id"]},
            "completed_at": utcnow_iso(),
        }).execute()
        return {"success": True, "feed": feed}

    @web_app.delete("/feeds/{feed_id}")
    async def delete_feed_endpoint(feed_id: str, admin=Depends(require_admin)):
        supabase = service_client()
        existing = (
            supabase.table("test_podcast_feeds")
            .select("id,name,rss_url")
            .eq("id", feed_id)
            .limit(1)
            .execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Feed not found")
        supabase.table("test_podcast_feeds").delete().eq("id", feed_id).execute()
        supabase.table("automation_logs").insert({
            "run_type": "feed_configuration",
            "status": "success",
            "result": {"action": "delete", "feed": existing.data[0], "changed_by": admin["id"]},
            "completed_at": utcnow_iso(),
        }).execute()
        return {"success": True, "deleted_feed_id": feed_id}

    return web_app


@app.local_entrypoint()
def main(action: str = "health", max_episodes: int = 1, days_back: int = 7):
    """Operator-only CLI entrypoint for audited smoke checks and bounded runs."""
    import json

    if action == "health":
        result = health_check.remote()
    elif action == "process":
        result = trigger_manual_processor.remote(
            max_episodes=max(1, min(max_episodes, 3)),
            days_back=days_back,
        )
    elif action == "scheduled-check":
        result = scheduled_processor.remote()
    else:
        raise ValueError("action must be health, process, or scheduled-check")

    print(json.dumps(result, indent=2, default=str))
