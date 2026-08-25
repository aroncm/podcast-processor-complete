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

PIPELINE_VERSION = "podthreads-hybrid-v3"
TRANSCRIPT_CORRECTION_PROMPT_VERSION = "adtech-terminology-correction-v1"
EXTRACTION_PROMPT_VERSION = "legacy-hybrid-takes-v3"
RANKING_PROMPT_VERSION = "legacy-hybrid-ranking-v4"
CONTEXT_PROMPT_VERSION = "adtech-connective-context-v3"
MAPPING_PROMPT_VERSION = "adtech-controlled-theme-mapping-v2"
HISTORICAL_MAPPING_PROMPT_VERSION = "adtech-historical-conversation-mapping-v1"
EDITORIAL_RUBRIC_VERSION = "podthreads-operator-take-rubric-v2"
MIN_QUOTE_WORDS = 20
IDEAL_QUOTE_WORDS_MIN = 30
IDEAL_QUOTE_WORDS_MAX = 50
MAX_QUOTE_WORDS = 80


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def quote_word_count(text: str) -> int:
    return len(str(text or "").strip().split())


def candidate_has_publishable_length(text: str) -> bool:
    """Keep the quote readable while allowing a complete spoken thought."""
    word_count = quote_word_count(text)
    return MIN_QUOTE_WORDS <= word_count <= MAX_QUOTE_WORDS


def apply_transcript_corrections(segments, proposed_corrections, minimum_confidence=0.94):
    """Apply only narrow, high-confidence term fixes while preserving raw text.

    The returned segments retain ``raw_text`` and an audit list. Corrections are
    display/extraction aids until an SME approves the staged take; they never
    replace the immutable raw transcript artifact.
    """
    import re

    corrected_segments = [{**segment, "raw_text": segment.get("text", "")} for segment in segments]
    applied = []
    rejected = []
    for proposal in proposed_corrections or []:
        try:
            segment_id = int(proposal.get("segment_id"))
            original = str(proposal.get("original_phrase") or "").strip()
            replacement = str(proposal.get("corrected_phrase") or "").strip()
            confidence = float(proposal.get("confidence", 0))
            if segment_id < 0 or segment_id >= len(corrected_segments):
                raise ValueError("segment_out_of_range")
            if confidence < minimum_confidence:
                raise ValueError("below_confidence_gate")
            if not original or not replacement or original.casefold() == replacement.casefold():
                raise ValueError("empty_or_unchanged")
            if len(original.split()) > 8 or len(replacement.split()) > 8:
                raise ValueError("replacement_too_broad")

            current_text = corrected_segments[segment_id].get("text", "")
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            if not pattern.search(current_text):
                raise ValueError("phrase_not_found")
            corrected_text = pattern.sub(replacement, current_text, count=1)
            corrected_segments[segment_id]["text"] = corrected_text
            audit_row = {
                **proposal,
                "segment_id": segment_id,
                "confidence": round(confidence, 4),
                "raw_segment_text": corrected_segments[segment_id]["raw_text"],
                "corrected_segment_text": corrected_text,
                "status": "applied_for_sme_review",
            }
            applied.append(audit_row)
        except Exception as exc:
            rejected.append({**proposal, "status": "not_applied", "reason": str(exc)})
    return corrected_segments, applied, rejected


def corrections_for_segment_range(corrections, start_segment, end_segment):
    return [
        correction for correction in (corrections or [])
        if int(start_segment) <= int(correction.get("segment_id", -1)) <= int(end_segment)
    ]


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
    register_pipeline_model_versions(supabase)
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
    """Fetch captions through the transcript API, then yt-dlp as a fallback.
    Results are cached in-memory per video to avoid redundant API calls."""
    import yt_dlp
    import requests
    import json
    import html
    from youtube_transcript_api import YouTubeTranscriptApi
    
    if youtube_id in _caption_cache:
        return _caption_cache[youtube_id]

    try:
        print(f"  🎬 Fetching YouTube captions for {youtube_id} via transcript API...")
        transcript = YouTubeTranscriptApi().fetch(youtube_id, languages=['en'])
        processed = []
        for snippet in transcript:
            text = html.unescape(str(getattr(snippet, 'text', '') or '')).strip()
            if not text:
                continue
            start = float(getattr(snippet, 'start', 0) or 0)
            duration = float(getattr(snippet, 'duration', 0) or 0)
            processed.append({
                'start': start,
                'end': start + duration,
                'raw_text': text,
                'norm_text': normalize_text(text),
                'word_count': len(text.split()),
            })
        if processed:
            print(f"  ✅ Parsed {len(processed)} transcript API events for {youtube_id}")
            _caption_cache[youtube_id] = processed
            return processed
    except Exception as exc:
        print(f"  ⚠️  Transcript API unavailable for {youtube_id}: {exc}")
    
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


@app.function(image=image, secrets=[my_secret], timeout=180)
def caption_source_check(youtube_id: str):
    """Read-only operator check for caption availability from Modal."""
    captions = get_yt_captions(youtube_id)
    if not captions:
        return {"ok": False, "youtube_id": youtube_id, "events": 0}
    return {
        "ok": True,
        "youtube_id": youtube_id,
        "events": len(captions),
        "first_start": captions[0]["start"],
        "last_end": captions[-1]["end"],
    }

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
        staged_result = (
            supabase.table("test_quotes")
            .select(
                "quote_text,speaker_name,speaker_title,speaker_company,category,"
                "approval_status,context_review_status,mapping_review_status"
            )
            .eq("id", quote_id)
            .single()
            .execute()
        )
        staged = staged_result.data or {}
        missing = missing_take_verification_fields(staged)
        if missing:
            return {
                "success": False,
                "error": f"Verify {', '.join(missing)} before publication",
            }
        if staged.get("approval_status") != "approved":
            return {"success": False, "error": "The take must be approved before publication"}
        if staged.get("context_review_status") != "approved":
            return {"success": False, "error": "Context must be approved before publication"}
        if staged.get("mapping_review_status") != "approved":
            return {"success": False, "error": "Connections must be approved before publication"}
        result = supabase.rpc(
            "promote_curated_quote_with_conversation",
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
        try:
            locked_set = (
                supabase.table("editorial_gold_sets")
                .select("id,version")
                .eq("status", "locked")
                .order("locked_at", desc=True)
                .limit(1)
                .execute()
            )
            if locked_set.data:
                gold_items = (
                    supabase.table("editorial_gold_set_items")
                    .select("label,preferred_quote_text,rationale")
                    .eq("gold_set_id", locked_set.data[0]["id"])
                    .order("created_at", desc=True)
                    .limit(24)
                    .execute()
                )
                positives = [row for row in (gold_items.data or []) if row.get("label") == "positive"][:8]
                negatives = [row for row in (gold_items.data or []) if row.get("label") == "negative"][:8]
                if positives and negatives:
                    sections = [
                        f"LOCKED SME GOLD SET: {locked_set.data[0]['version']}",
                        "Infer the editorial principles; never copy wording.",
                        "APPROVED HIGH-SIGNAL TAKES:",
                    ]
                    sections.extend(
                        f"+ {row.get('preferred_quote_text', '')}\n  Editorial reason: {row.get('rationale', '')}"
                        for row in positives
                    )
                    sections.append("REJECTED OR GENERIC TAKES:")
                    sections.extend(
                        f"- {row.get('preferred_quote_text', '')}\n  Rejection reason: {row.get('rationale', '')}"
                        for row in negatives
                    )
                    return "\n".join(sections)
        except Exception as gold_exc:
            print(f"⚠️ Locked gold set unavailable: {gold_exc}")

        approved_res = (
            supabase.table("test_quotes")
            .select("quote_text, editorial_context, ranking_reason")
            .in_("approval_status", ["approved", "promoted"])
            .eq("used_for_training", True)
            .order("updated_at", desc=True)
            .limit(40)
            .execute()
        )
        rejected_res = (
            supabase.table("test_quotes")
            .select("quote_text, rejection_reason")
            .eq("approval_status", "rejected")
            .order("updated_at", desc=True)
            .limit(80)
            .execute()
        )

        approved = (approved_res.data or [])[:8]
        rejected = [
            row for row in (rejected_res.data or [])
            if str(row.get("rejection_reason") or "").strip()
        ][:8]
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


def register_pipeline_model_versions(supabase) -> None:
    """Record stage-specific provenance without marking the hybrid as active."""
    versions = [
        {
            "id": (
                f"{os.environ.get('OPENAI_TERMINOLOGY_MODEL', os.environ.get('OPENAI_EDITORIAL_MODEL', 'gpt-5.6-sol'))}:"
                f"{TRANSCRIPT_CORRECTION_PROMPT_VERSION}:{PIPELINE_VERSION}"
            ),
            "component": "terminology",
            "provider": "openai",
            "model_name": os.environ.get(
                "OPENAI_TERMINOLOGY_MODEL",
                os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol"),
            ),
            "prompt_version": TRANSCRIPT_CORRECTION_PROMPT_VERSION,
            "rubric_version": EDITORIAL_RUBRIC_VERSION,
            "status": "candidate",
            "configuration": {
                "minimum_confidence": 0.94,
                "raw_transcript_immutable": True,
            },
        },
        {
            "id": f"{os.environ.get('OPENAI_CANDIDATE_MODEL', 'gpt-5.6-terra')}:{EXTRACTION_PROMPT_VERSION}:{PIPELINE_VERSION}",
            "component": "extraction",
            "provider": "openai",
            "model_name": os.environ.get("OPENAI_CANDIDATE_MODEL", "gpt-5.6-terra"),
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "rubric_version": EDITORIAL_RUBRIC_VERSION,
            "status": "shadow",
            "configuration": {
                "minimum_words": MIN_QUOTE_WORDS,
                "ideal_words": [IDEAL_QUOTE_WORDS_MIN, IDEAL_QUOTE_WORDS_MAX],
                "maximum_words": MAX_QUOTE_WORDS,
                "complete_transcript": True,
                "source_grounding_required": True,
            },
        },
        {
            "id": (
                f"{os.environ.get('OPENAI_RANKING_MODEL', os.environ.get('OPENAI_EDITORIAL_MODEL', 'gpt-5.6-sol'))}:"
                f"{RANKING_PROMPT_VERSION}:{PIPELINE_VERSION}"
            ),
            "component": "ranking",
            "provider": "openai",
            "model_name": os.environ.get(
                "OPENAI_RANKING_MODEL",
                os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol"),
            ),
            "prompt_version": RANKING_PROMPT_VERSION,
            "rubric_version": EDITORIAL_RUBRIC_VERSION,
            "status": "shadow",
            "configuration": {
                "selection_isolated_from_analysis": True,
                "minimum_quality": float(os.environ.get("MIN_QUOTE_QUALITY", "0.74")),
            },
        },
        {
            "id": f"{os.environ.get('OPENAI_EDITORIAL_MODEL', 'gpt-5.6-sol')}:{CONTEXT_PROMPT_VERSION}:{PIPELINE_VERSION}",
            "component": "context",
            "provider": "openai",
            "model_name": os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol"),
            "prompt_version": CONTEXT_PROMPT_VERSION,
            "rubric_version": EDITORIAL_RUBRIC_VERSION,
            "status": "candidate",
            "configuration": {"selection_can_be_changed": False},
        },
        {
            "id": f"{os.environ.get('OPENAI_EDITORIAL_MODEL', 'gpt-5.6-sol')}:{MAPPING_PROMPT_VERSION}:{PIPELINE_VERSION}",
            "component": "mapping",
            "provider": "openai",
            "model_name": os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol"),
            "prompt_version": MAPPING_PROMPT_VERSION,
            "rubric_version": EDITORIAL_RUBRIC_VERSION,
            "status": "candidate",
            "configuration": {
                "controlled_theme_registry": True,
                "auto_publish": False,
            },
        },
    ]
    try:
        supabase.table("model_versions").upsert(versions, on_conflict="id").execute()
    except Exception as exc:
        print(f"AUDIT_WARNING model provenance registration failed: {exc}")


def fetch_conversation_taxonomy(supabase) -> str:
    """Give mapping a reviewed vocabulary so equivalent ideas converge over time."""
    try:
        registry = []
        try:
            registry_result = (
                supabase.table("adtech_theme_registry")
                .select(
                    "canonical_name,definition,aliases,inclusion_criteria,"
                    "exclusion_criteria,positive_examples,counter_examples"
                )
                .eq("status", "active")
                .order("canonical_name")
                .limit(80)
                .execute()
            )
            registry = registry_result.data or []
        except Exception as registry_exc:
            print(f"⚠️ Controlled theme registry unavailable: {registry_exc}")

        themes_result = (
            supabase.table("conversation_themes")
            .select("id,name,summary")
            .eq("publication_status", "published")
            .order("updated_at", desc=True)
            .limit(80)
            .execute()
        )
        questions_result = (
            supabase.table("conversation_questions")
            .select("theme_id,question_text,summary")
            .eq("publication_status", "published")
            .limit(240)
            .execute()
        )
        entities_result = (
            supabase.table("conversation_entities")
            .select("entity_type,name,description")
            .eq("publication_status", "published")
            .order("name")
            .limit(400)
            .execute()
        )

        themes = themes_result.data or []
        questions = questions_result.data or []
        entities = entities_result.data or []
        if not registry and not themes and not questions and not entities:
            return ""

        theme_names = {row["id"]: row.get("name") for row in themes}
        reviewed_graph = {
            "active_theme_registry": registry,
            "themes": [
                {"name": row.get("name"), "summary": row.get("summary")}
                for row in themes
            ],
            "questions": [
                {
                    "theme": theme_names.get(row.get("theme_id")),
                    "question": row.get("question_text"),
                    "summary": row.get("summary"),
                }
                for row in questions
            ],
            "entities": [
                {
                    "type": row.get("entity_type"),
                    "name": row.get("name"),
                    "description": row.get("description"),
                }
                for row in entities
            ],
        }
        return json.dumps(reviewed_graph, ensure_ascii=False)
    except Exception as exc:
        # The v2 migration may not yet be applied during a controlled rollout.
        print(f"⚠️ Conversation vocabulary unavailable: {exc}")
        return ""


def fetch_terminology_glossary(supabase, podcast: str, episode: str) -> str:
    """Load reviewed names and AdTech language to improve transcript display."""
    glossary = {
        "podcast": podcast,
        "episode": episode,
        "themes": [],
        "entities": [],
    }
    try:
        registry = (
            supabase.table("adtech_theme_registry")
            .select("canonical_name,aliases")
            .eq("status", "active")
            .order("canonical_name")
            .limit(80)
            .execute()
        )
        glossary["themes"] = registry.data or []
    except Exception as exc:
        print(f"⚠️ Theme glossary unavailable: {exc}")
    try:
        entities = (
            supabase.table("conversation_entities")
            .select("entity_type,name")
            .eq("publication_status", "published")
            .order("name")
            .limit(400)
            .execute()
        )
        glossary["entities"] = entities.data or []
    except Exception as exc:
        print(f"⚠️ Entity glossary unavailable: {exc}")
    return json.dumps(glossary, ensure_ascii=False)


def propose_transcript_corrections(
    segments,
    podcast,
    episode,
    client,
    terminology_glossary="",
):
    """Propose narrow terminology fixes; no prose rewriting is permitted."""
    if not segments:
        return []
    model = os.environ.get(
        "OPENAI_TERMINOLOGY_MODEL",
        os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol"),
    )
    reasoning_effort = os.environ.get("OPENAI_TERMINOLOGY_REASONING", "medium")
    correction_schema = {
        "type": "object",
        "properties": {
            "corrections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "integer"},
                        "original_phrase": {"type": "string"},
                        "corrected_phrase": {"type": "string"},
                        "correction_type": {
                            "type": "string",
                            "enum": ["industry_term", "person", "company", "product"],
                        },
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "segment_id", "original_phrase", "corrected_phrase",
                        "correction_type", "confidence", "rationale",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["corrections"],
        "additionalProperties": False,
    }
    proposals = []
    for chunk_index, chunk_text in enumerate(
        build_extraction_chunks(segments, max_chars=15000, overlap_segments=0),
        start=1,
    ):
        data = call_openai_structured(
            client,
            model=model,
            system_prompt=(
                "You are a conservative transcript terminology editor for an expert AdTech "
                "publication. Correct only unmistakable mistranscriptions of industry terms, "
                "named people, companies, or products. Never improve grammar, paraphrase, "
                "change a claim, or infer a term from weak context. Abstain when uncertain."
            ),
            user_prompt=f"""
Podcast: {podcast}
Episode: {episode}
Transcript section: {chunk_index}

Reviewed terminology and entity glossary:
{terminology_glossary or "No reviewed glossary entries are available."}

Return only high-confidence phrase replacements. Each `original_phrase` must
appear exactly in its numbered segment. Confidence below 0.94 should be omitted.
Do not rewrite a full sentence. Keep each phrase to eight words or fewer.

TRANSCRIPT WITH GLOBAL SEGMENT IDS:
{chunk_text}
""",
            schema_name="podthreads_transcript_corrections",
            schema=correction_schema,
            reasoning_effort=reasoning_effort,
            max_output_tokens=4000,
        )
        proposals.extend(data.get("corrections", []))
    return proposals


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
    direct_evidence = [
        evidence for evidence in (evidence_items or [])
        if evidence.get("evidence_type") == "direct_transcript"
    ]
    if not direct_evidence:
        return False
    for evidence in direct_evidence:
        evidence_segments = evidence.get("segment_ids") or []
        if not evidence_segments or any(
            int(segment_id) < int(start_segment) or int(segment_id) > int(end_segment)
            for segment_id in evidence_segments
        ):
            return False
    return True


def conversation_mapping_is_reviewable(selection, start_segment, end_segment):
    """Reject malformed connection proposals without rejecting the underlying take."""
    required_text = (
        "theme_name", "theme_summary", "question_text",
        "question_summary", "connection_context",
    )
    if any(not str(selection.get(field, "")).strip() for field in required_text):
        return False

    people = selection.get("related_people") or []
    companies = selection.get("related_companies") or []
    if not isinstance(people, list) or not isinstance(companies, list):
        return False

    allowed_evidence_types = {
        "direct_transcript", "episode_metadata", "speaker_identity",
        "editorial_connection",
    }
    for entity in people + companies:
        if not isinstance(entity, dict):
            return False
        if any(not str(entity.get(field, "")).strip() for field in ("name", "relationship", "evidence")):
            return False
        if entity.get("evidence_type") not in allowed_evidence_types:
            return False
        segment_ids = entity.get("segment_ids") or []
        if entity.get("evidence_type") == "direct_transcript" and (
            not segment_ids
            or any(
                int(segment_id) < int(start_segment) or int(segment_id) > int(end_segment)
                for segment_id in segment_ids
            )
        ):
            return False
    return True


TAKE_RECORD_FIELDS = {
    "quote_text", "speaker_name", "speaker_title", "speaker_company",
    "speaker_linkedin", "category", "podcast_name", "episode_name",
    "youtube_id", "timestamp_start", "timestamp_end", "youtube_offset",
}
CONTEXT_RECORD_FIELDS = {"editorial_context"}
MAPPING_RECORD_FIELDS = {
    "proposed_theme_name", "proposed_theme_summary",
    "proposed_question_text", "proposed_question_summary",
    "proposed_people", "proposed_companies", "connection_context",
    "theme_match_action",
}


def missing_take_verification_fields(record):
    """Return the human-readable metadata still required for take approval."""
    required = {
        "quote_text": "take",
        "speaker_name": "speaker",
        "speaker_title": "speaker title",
        "speaker_company": "speaker company",
        "category": "category",
    }
    return [
        label for field, label in required.items()
        if not str((record or {}).get(field) or "").strip()
    ]


def editorial_gate_invalidations(before, updates):
    """Reopen only the approval gates affected by an audited edit."""
    before = before or {}
    updates = updates or {}
    changed_take = any(
        field in updates and updates[field] != before.get(field)
        for field in TAKE_RECORD_FIELDS
    )
    changed_context = any(
        field in updates and updates[field] != before.get(field)
        for field in CONTEXT_RECORD_FIELDS
    )
    changed_mapping = any(
        field in updates and updates[field] != before.get(field)
        for field in MAPPING_RECORD_FIELDS
    )
    invalidations = {}
    if changed_take and before.get("approval_status") == "approved":
        invalidations.update({
            "approval_status": "pending",
            "context_review_status": "unreviewed",
            "context_reviewed_by": None,
            "context_reviewed_at": None,
            "mapping_review_status": "unreviewed",
            "mapping_reviewed_by": None,
            "mapping_reviewed_at": None,
        })
    elif changed_context:
        invalidations.update({
            "context_review_status": "unreviewed",
            "context_reviewed_by": None,
            "context_reviewed_at": None,
        })
    if changed_mapping:
        invalidations.update({
            "mapping_review_status": "unreviewed",
            "mapping_reviewed_by": None,
            "mapping_reviewed_at": None,
        })
    return invalidations


def build_caption_evidence(captions, start_time, end_time, padding_seconds=45, max_events=120):
    """Build a bounded, numbered caption window for historical mapping review."""
    if not captions or end_time <= start_time:
        return None
    window_start = max(0.0, float(start_time) - float(padding_seconds))
    window_end = float(end_time) + float(padding_seconds)
    selected = []
    for source_index, caption in enumerate(captions):
        caption_start = float(caption.get("start", 0))
        caption_end = float(caption.get("end", caption_start))
        if caption_end < window_start or caption_start > window_end:
            continue
        text = str(caption.get("raw_text") or "").strip()
        if not text:
            continue
        selected.append({
            "id": source_index,
            "start": caption_start,
            "end": caption_end,
            "text": text,
        })
        if len(selected) >= max_events:
            break
    if not selected:
        return None
    return {
        "segments": selected,
        "start_segment": selected[0]["id"],
        "end_segment": selected[-1]["id"],
        "excerpt": "\n".join(f"[{row['id']}] {row['text']}" for row in selected),
    }


def align_quote_to_segments(quote_text, segments, expected_start, expected_end):
    """Strictly align a published quote to timestamped transcript segments."""
    from difflib import SequenceMatcher

    normalized_quote = normalize_text(quote_text)
    quote_words = normalized_quote.split()
    if len(quote_words) < 5 or not segments:
        return None
    search_start = max(0.0, float(expected_start) - 300)
    search_end = float(expected_end) + 300
    relevant = [
        (index, segment)
        for index, segment in enumerate(segments)
        if float(segment.get("end", 0)) >= search_start
        and float(segment.get("start", 0)) <= search_end
    ]
    minimum_words = max(1, int(len(quote_words) * 0.55))
    maximum_words = int(len(quote_words) * 2.2)
    best = None
    runner_up = 0.0
    quote_word_set = set(quote_words)
    for position, (source_index, _segment) in enumerate(relevant):
        for window_size in range(1, 16):
            window = relevant[position:position + window_size]
            if len(window) != window_size:
                break
            window_words = " ".join(
                normalize_text(row.get("raw_text") or row.get("text") or "")
                for _, row in window
            ).split()
            if len(window_words) < minimum_words:
                continue
            if len(window_words) > maximum_words:
                break
            window_text = " ".join(window_words)
            sequence_score = SequenceMatcher(None, normalized_quote, window_text).ratio()
            common = quote_word_set & set(window_words)
            f1 = 0.0
            recall = 0.0
            if common:
                precision = len(common) / len(set(window_words))
                recall = len(common) / len(quote_word_set)
                f1 = 2 * precision * recall / (precision + recall)
            score = (0.6 * sequence_score) + (0.25 * f1) + (0.15 * recall)
            candidate = {
                "score": score,
                "start_index": source_index,
                "end_index": window[-1][0],
                "start": float(window[0][1].get("start", 0)),
                "end": float(window[-1][1].get("end", 0)),
            }
            if best is None or score > best["score"]:
                if best is not None and (
                    candidate["start_index"] > best["end_index"]
                    or candidate["end_index"] < best["start_index"]
                ):
                    runner_up = max(runner_up, best["score"])
                best = candidate
            elif best is not None and (
                candidate["start_index"] > best["end_index"]
                or candidate["end_index"] < best["start_index"]
            ):
                runner_up = max(runner_up, score)
    if not best:
        return None
    minimum_score = 0.70 if len(quote_words) > 30 else 0.75
    minimum_margin = 0.03 if len(quote_words) > 30 else 0.04
    if best["score"] < minimum_score or best["score"] - runner_up < minimum_margin:
        return None
    return {
        "start": best["start"],
        "end": best["end"],
        "confidence": round(best["score"], 3),
    }


def resolve_rss_audio_source(podcast_name, episode_name, feed_rows):
    """Resolve a legacy episode to its current RSS enclosure without mutating data."""
    import feedparser
    from difflib import SequenceMatcher

    normalized_podcast = normalize_text(podcast_name or "")
    feed_row = max(
        feed_rows,
        key=lambda row: SequenceMatcher(
            None, normalized_podcast, normalize_text(row.get("name") or "")
        ).ratio(),
        default=None,
    )
    if not feed_row or not feed_row.get("rss_url"):
        return None
    if SequenceMatcher(
        None, normalized_podcast, normalize_text(feed_row.get("name") or "")
    ).ratio() < 0.80:
        return None
    parsed = feedparser.parse(feed_row["rss_url"])
    normalized_episode = normalize_text(episode_name or "")
    best_entry = None
    best_score = 0.0
    for entry in parsed.entries:
        score = SequenceMatcher(
            None,
            normalized_episode,
            normalize_text(getattr(entry, "title", "")),
        ).ratio()
        if score > best_score:
            best_score = score
            best_entry = entry
    if not best_entry or best_score < 0.76:
        return None
    audio_url = None
    for enclosure in getattr(best_entry, "enclosures", []) or []:
        href = enclosure.get("href") or enclosure.get("url")
        content_type = str(enclosure.get("type") or "")
        if href and (content_type.startswith("audio/") or not audio_url):
            audio_url = href
            if content_type.startswith("audio/"):
                break
    if not audio_url:
        for link in getattr(best_entry, "links", []) or []:
            if str(link.get("type") or "").startswith("audio/") and link.get("href"):
                audio_url = link["href"]
                break
    if not audio_url:
        return None
    return {
        "audio_url": audio_url,
        "feed_url": feed_row["rss_url"],
        "episode_match_confidence": round(best_score, 3),
        "episode_title": getattr(best_entry, "title", None),
    }


def transcribe_remote_audio_window(audio_url, expected_start, expected_end, client):
    """Transcribe only the bounded RSS window needed to support one legacy take."""
    import subprocess
    import tempfile

    clip_start = max(0.0, float(expected_start) - 120.0)
    requested_duration = max(240.0, float(expected_end) - float(expected_start) + 240.0)
    clip_duration = min(requested_duration, 600.0)
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_path = temp_audio.name
    temp_audio.close()
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-ss", str(clip_start), "-i", audio_url,
                "-t", str(clip_duration), "-ac", "1", "-ar", "16000",
                "-b:a", "64k", "-y", temp_path,
            ],
            capture_output=True,
            text=True,
            timeout=360,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"RSS audio window extraction failed: {completed.stderr[-800:]}")
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "whisper-1"),
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        processed = []
        for index, segment in enumerate(getattr(transcript, "segments", None) or []):
            if isinstance(segment, dict):
                text = segment.get("text", "")
                start = float(segment.get("start", 0))
                end = float(segment.get("end", start))
            else:
                text = getattr(segment, "text", "")
                start = float(getattr(segment, "start", 0))
                end = float(getattr(segment, "end", start))
            text = str(text or "").strip()
            if not text:
                continue
            processed.append({
                "id": index,
                "start": start + clip_start,
                "end": end + clip_start,
                "raw_text": text,
                "norm_text": normalize_text(text),
                "word_count": len(text.split()),
            })
        return processed
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def historical_mapping_is_reviewable(mapping, start_segment, end_segment):
    """Apply the normal evidence gate plus historical-workflow quality floors."""
    if mapping.get("abstain"):
        return False
    if not conversation_mapping_is_reviewable(mapping, start_segment, end_segment):
        return False
    if len(str(mapping.get("connection_context") or "").split()) < 25:
        return False
    try:
        confidence = float(mapping.get("mapping_confidence", 0))
    except (TypeError, ValueError):
        return False
    return confidence >= float(os.environ.get("MIN_HISTORICAL_MAPPING_CONFIDENCE", "0.72"))


def propose_historical_conversation_mapping(
    quote,
    source_evidence,
    client,
    conversation_taxonomy="",
):
    """Propose only conversation placement; never rewrite a published take."""
    model = os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol")
    reasoning_effort = os.environ.get("OPENAI_EDITORIAL_REASONING", "high")
    entity_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "relationship": {"type": "string"},
            "description": {"type": "string"},
            "evidence_type": {
                "type": "string",
                "enum": [
                    "direct_transcript", "episode_metadata",
                    "speaker_identity", "editorial_connection",
                ],
            },
            "evidence": {"type": "string"},
            "segment_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": [
            "name", "relationship", "description", "evidence_type",
            "evidence", "segment_ids",
        ],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "abstain": {"type": "boolean"},
            "abstention_reason": {"type": "string"},
            "theme_name": {"type": "string"},
            "theme_summary": {"type": "string"},
            "question_text": {"type": "string"},
            "question_summary": {"type": "string"},
            "relationship_label": {"type": "string"},
            "connection_context": {"type": "string"},
            "mapping_confidence": {"type": "number"},
            "related_people": {"type": "array", "items": entity_schema},
            "related_companies": {"type": "array", "items": entity_schema},
        },
        "required": [
            "abstain", "abstention_reason", "theme_name", "theme_summary",
            "question_text", "question_summary", "relationship_label",
            "connection_context", "mapping_confidence", "related_people",
            "related_companies",
        ],
        "additionalProperties": False,
    }
    user_prompt = f"""
Published take (immutable): {quote.get('text')}
Speaker: {quote.get('speaker_name') or 'Unknown'}
Speaker title: {quote.get('speaker_title') or 'Unknown'}
Speaker company: {quote.get('speaker_company') or 'Unknown'}
Podcast: {quote.get('podcast_name') or 'Unknown'}
Episode: {quote.get('episode_name') or 'Unknown'}

Map this existing take into the ongoing industry discourse. This is not a fact
check, a summary, or an opportunity to embellish the quote.

Editorial standard:
1. Start with a durable industry theme broad enough to connect multiple people,
   companies, episodes, and questions. A theme is not the take's category.
2. Place the take under one open question inside that theme. The question should
   expose a live operating, market-structure, measurement, incentive, or policy
   tension that multiple industry participants could answer differently.
3. `connection_context` is 45-90 words. Explain how this take advances or
   complicates that conversation using concrete adtech mechanisms and
   stakeholder incentives. Do not praise the take, paraphrase it, verify a
   claim, or use claim/consequence language.
4. Write from inside the industry. Use relevant precision about auctions,
   identity/addressability, incrementality, privacy, supply paths, publisher
   economics, agency/brand incentives, CTV, retail media, walled gardens, or AI,
   but only where the source actually supports the connection.
5. Add named people and companies when supported by speaker identity, episode
   metadata, the numbered source segments, or a clearly labeled editorial
   connection. Do not infer employment, partnerships, or interviews.
6. `direct_transcript` evidence must cite segment IDs inside the supplied source
   boundary. Other evidence types use an empty segment list.
7. Reuse an exact approved theme, question, or entity name below when the idea is
   substantively the same. Do not force a quote into Performance TV simply
   because it is the only current theme.
8. Abstain when the source is too thin, the quote is too generic, or a defensible
   placement would require facts outside the supplied evidence.
9. Avoid generic AI prose: no "underscores the importance", "rapidly evolving
   landscape", "game changer", "key takeaway", or "businesses must adapt".

EXISTING SME-APPROVED CONVERSATION GRAPH:
{conversation_taxonomy or "No approved graph vocabulary exists yet."}

SOURCE-BOUNDED CAPTIONS:
{source_evidence['excerpt']}
"""
    system_prompt = """
You are PodThreads' senior adtech editor. You connect podcast moments into
durable industry conversations with the judgment of an experienced operator.
Your output is a private proposal for SME review, never a published conclusion.
Prefer abstention to generic analysis or unsupported connections.
"""
    return call_openai_structured(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="podthreads_historical_mapping",
        schema=schema,
        reasoning_effort=reasoning_effort,
        max_output_tokens=4500,
    )


def process_single_episode_logic(episode, feed, client, supabase, job_id=None):
    """Refactored logic for processing a single episode"""
    import subprocess
    import tempfile
    import time
    
    # Balanced preference context from SME approvals and rejections.
    curation_examples = fetch_curation_examples(supabase)
    if curation_examples:
        print("  🧠 Loaded balanced SME preference examples")
    conversation_taxonomy = fetch_conversation_taxonomy(supabase)
    if conversation_taxonomy:
        print("  🕸️ Loaded the SME-approved conversation vocabulary")
    terminology_glossary = fetch_terminology_glossary(
        supabase,
        feed.get("name", ""),
        getattr(episode, "title", ""),
    )

    
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
        raw_transcript_text = transcription["text"]
        raw_segments = transcription["segments"]
        print(f"✅ Transcription complete: {len(raw_transcript_text)} characters, {len(raw_segments)} segments")

        # Correct only unmistakable AdTech terms and named entities. The raw
        # transcript remains immutable and is stored alongside every correction.
        update_processing_job(
            supabase,
            job_id,
            "transcribing",
            progress={"phase": "terminology_correction"},
        )
        correction_proposals = propose_transcript_corrections(
            raw_segments,
            feed["name"],
            episode.title,
            client,
            terminology_glossary=terminology_glossary,
        )
        segments, applied_corrections, rejected_corrections = apply_transcript_corrections(
            raw_segments,
            correction_proposals,
        )
        corrected_transcript_text = "\n".join(
            segment.get("text", "") for segment in segments
        ).strip()
        print(
            f"📝 Terminology pass: {len(applied_corrections)} applied for review; "
            f"{len(rejected_corrections)} withheld"
        )

        artifact_payload = {
            "processing_job_id": job_id,
            "episode_guid": episode_guid,
            "podcast_name": feed["name"],
            "episode_name": episode.title,
            "source_audio_url": audio_url,
            "transcript_text": raw_transcript_text,
            "transcript_segments": raw_segments,
            "corrected_transcript_text": corrected_transcript_text,
            "corrected_transcript_segments": segments,
            "transcript_corrections": applied_corrections,
            "rejected_transcript_corrections": rejected_corrections,
            "transcript_model": transcription["model"],
            "terminology_model": os.environ.get(
                "OPENAI_TERMINOLOGY_MODEL",
                os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol"),
            ),
            "terminology_prompt_version": TRANSCRIPT_CORRECTION_PROMPT_VERSION,
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
                    raw_source_excerpt = " ".join(
                        segment["text"] for segment in raw_segments[start_id:end_id + 1]
                    ).strip()
                    candidate_text = candidate.get("text", "").strip()
                    if not candidate_has_publishable_length(candidate_text):
                        raise ValueError(
                            f"quote length outside {MIN_QUOTE_WORDS}-{MAX_QUOTE_WORDS} words "
                            f"({quote_word_count(candidate_text)} words)"
                        )
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
                        "raw_source_transcript_excerpt": raw_source_excerpt,
                        "transcript_corrections": corrections_for_segment_range(
                            applied_corrections,
                            start_id,
                            end_id,
                        ),
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
        ranked_quotes = rank_quote_candidates(
            all_candidates[:30],
            feed['name'],
            episode.title,
            client,
            curation_examples=curation_examples,
        )[:5]
        all_quotes = contextualize_and_map_quotes(
            ranked_quotes,
            feed['name'],
            episode.title,
            client,
            conversation_taxonomy=conversation_taxonomy,
        )

        update_processing_job(
            supabase,
            job_id,
            "mapping",
            progress={
                "grounded_candidates": len(all_candidates),
                "mapped_candidates": len(all_quotes),
            },
        )
        
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
        update_processing_job(
            supabase,
            job_id,
            "staging",
            progress={"candidates_ready_for_sme_review": len(all_quotes)},
        )
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
                'speaker_title': quote.get('speaker_title'),
                'speaker_company': quote.get('speaker_company'),
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
                    os.environ.get('OPENAI_CANDIDATE_MODEL', 'gpt-5.6-terra'),
                ),
                'ranking_model': quote.get(
                    'ranking_model',
                    os.environ.get('OPENAI_RANKING_MODEL', 'gpt-5.6-sol'),
                ),
                'yt_timestamp_confidence': yt_confidence, # NULL if failed, signals local bridge
                'processing_job_id': job_id,
                'candidate_fingerprint': hashlib.sha256(
                    f"{episode_guid}|{normalize_text(quote['text'])}".encode("utf-8")
                ).hexdigest(),
                'candidate_set_id': candidate_set_id,
                'candidate_rank': i + 1,
                'ranking_reason': quote.get('ranking_reason'),
                'quote_word_count': quote_word_count(quote['text']),
                'pipeline_version': PIPELINE_VERSION,
                'extraction_prompt_version': EXTRACTION_PROMPT_VERSION,
                'ranking_prompt_version': RANKING_PROMPT_VERSION,
                'original_quote_text': quote['text'],
                'source_transcript_excerpt': quote.get('source_transcript_excerpt'),
                'raw_source_transcript_excerpt': quote.get('raw_source_transcript_excerpt'),
                'transcript_corrections': quote.get('transcript_corrections', []),
                'terminology_model': os.environ.get(
                    'OPENAI_TERMINOLOGY_MODEL',
                    os.environ.get('OPENAI_EDITORIAL_MODEL', 'gpt-5.6-sol'),
                ),
                'terminology_prompt_version': TRANSCRIPT_CORRECTION_PROMPT_VERSION,
                'source_start_segment': quote.get('start_seg'),
                'source_end_segment': quote.get('end_seg'),
                'editorial_context': quote.get('editorial_context'),
                'context_evidence': quote.get('context_evidence', []),
                'context_confidence': quote.get('context_confidence'),
                'context_model': quote.get(
                    'context_model',
                    os.environ.get('OPENAI_EDITORIAL_MODEL', 'gpt-5.6-sol'),
                ),
                'context_prompt_version': CONTEXT_PROMPT_VERSION,
                # Context is never public until an SME approves it explicitly.
                'context_review_status': 'unreviewed',
                'proposed_theme_name': quote.get('theme_name'),
                'proposed_theme_summary': quote.get('theme_summary'),
                'proposed_question_text': quote.get('question_text'),
                'proposed_question_summary': quote.get('question_summary'),
                'proposed_people': quote.get('related_people', []),
                'proposed_companies': quote.get('related_companies', []),
                'connection_context': quote.get('connection_context'),
                'mapping_confidence': quote.get('mapping_confidence'),
                'theme_match_action': quote.get('theme_match_action', 'abstain'),
                'mapping_model': quote.get(
                    'mapping_model',
                    os.environ.get('OPENAI_EDITORIAL_MODEL', 'gpt-5.6-sol'),
                ),
                'mapping_prompt_version': MAPPING_PROMPT_VERSION,
                'analysis_review_flags': quote.get('analysis_review_flags', {}),
                # Themes, questions, and entities have their own SME gate.
                'mapping_review_status': 'unreviewed',
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
    """Retrieve readable, literal candidates using the proven legacy taste bar."""
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

Select zero to four candidate takes from the transcript below. Zero is the
correct answer when this section contains no genuinely high-signal take.

Hard requirements:
- `text` must be copied verbatim from contiguous transcript segments.
- Segment IDs must exactly bound the quoted source.
- The quote must be 20-80 words; 30-50 words is ideal. Choose the shortest
  contiguous span that preserves the speaker's complete thought.
- The quote must be self-contained enough to understand without a generated
  explanation. Do not start or end mid-thought.
- It must be at least one of: counterintuitive, convention-challenging,
  specific and memorable, genuinely thought-provoking, or surprising.
- Prefer a specific prediction, causal claim, economic tradeoff, market-structure
  argument, counter-position, vivid example, or reusable framework.
- A candidate should matter to an adtech operator, publisher, marketer, agency,
  platform, investor, or regulator because it changes a decision or assumption.
- Reject generic advice, common knowledge, motivational platitudes, interview
  transitions, incomplete thoughts, vague philosophy, scene-setting, biography,
  sales pitches, slogans, and commentary a smart generalist could give in any
  industry.
- Scores are numbers from 0 to 1. `genericness_risk` is higher when the take is
  interchangeable with generic business or AI commentary.
- Do not manufacture controversy. Do not rewrite or improve the speaker's words.
- Quality over quantity. Never fill a quota.

{curation_examples}

TRANSCRIPT WITH GLOBAL SEGMENT IDS:
{text}
"""
    system_prompt = """
You are the candidate-retrieval layer for PodThreads. Preserve the editorial
instinct of the original PodTakes curator: find the statements that challenge,
surprise, or deeply illuminate, while rejecting generic business fluff. You
understand AdTech market structure and terminology, but this step is extractive,
not generative. Literal source fidelity and readable quote packaging are both
mandatory. Final judgment happens in a separate SME-ranking stage.
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
    candidates = data.get("candidates", [])[:4]
    for candidate in candidates:
        for key in (
            "novelty", "provocation", "domain_specificity",
            "evidence_quality", "genericness_risk",
        ):
            candidate[key] = max(0.0, min(1.0, float(candidate.get(key, 0))))
        candidate["extraction_model"] = model
        candidate["extraction_prompt_version"] = EXTRACTION_PROMPT_VERSION
        candidate["quote_word_count"] = quote_word_count(candidate.get("text", ""))
    return [
        candidate for candidate in candidates
        if candidate_has_publishable_length(candidate.get("text", ""))
    ]


def rank_quote_candidates(
    candidates,
    podcast,
    episode,
    client,
    curation_examples="",
):
    """Choose takes without allowing downstream analysis to influence selection."""
    if not candidates:
        return []

    model = os.environ.get(
        "OPENAI_RANKING_MODEL",
        os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol"),
    )
    reasoning_effort = os.environ.get(
        "OPENAI_RANKING_REASONING",
        os.environ.get("OPENAI_EDITORIAL_REASONING", "high"),
    )
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
                        "genericness_check": {"type": "string", "enum": ["pass", "fail"]},
                        "self_contained_check": {"type": "string", "enum": ["pass", "fail"]},
                        "word_count_check": {"type": "string", "enum": ["pass", "fail"]},
                    },
                    "required": [
                        "candidate_index", "quality_score", "ranking_reason",
                        "genericness_check", "self_contained_check", "word_count_check"
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
            "word_count": quote_word_count(candidate.get("text", "")),
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

Rank up to five takes. Select none when the candidates do not clear the bar.

Editorial standard:
1. Preserve the original PodTakes standard: the quote should challenge,
   surprise, or deeply illuminate. It must be specific and memorable rather
   than merely competent commentary.
2. The take changes an AdTech operator's understanding of a decision,
   incentive, tradeoff, prediction, market structure, or causal mechanism.
3. It must be understandable as a standalone spoken moment. Fail fragments,
   pronoun-dependent excerpts, and quotes that require generated context to make
   sense.
4. It must contain 20-80 words; 30-50 is the preferred editorial package.
5. Fail generic advice, common knowledge, motivational language, sales pitches,
   vague futurism, summaries, scene-setting, and claims portable to any industry.
6. Do not reward controversy for its own sake. Provocation is useful only when
   grounded in a concrete industry idea.
7. Scores are from 0 to 1. Reserve 0.90+ for unusually specific, memorable,
   source-grounded insight. Quality over quantity.

{curation_examples}

CANDIDATES:
{json.dumps(compact_candidates, ensure_ascii=False)}
"""
    system_prompt = """
You are PodThreads' senior quote editor. Your only task in this step is choosing
the strongest spoken takes. Do not write context, infer a theme, or reward a quote
because it would be easy to analyze. Apply the proven original PodTakes taste:
specific, memorable, counterintuitive, surprising, or genuinely illuminating.
Reject polished generic business commentary and incomplete transcript fragments.
"""
    data = call_openai_structured(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="podthreads_quote_ranking",
        schema=selection_schema,
        reasoning_effort=reasoning_effort,
        max_output_tokens=5000,
    )

    minimum_quality = float(os.environ.get("MIN_QUOTE_QUALITY", "0.74"))
    selected = []
    used_indices = set()
    for selection in data.get("selections", []):
        index = int(selection.get("candidate_index", -1))
        if index < 0 or index >= len(candidates) or index in used_indices:
            continue
        quality = max(0.0, min(1.0, float(selection.get("quality_score", 0))))
        if quality < minimum_quality:
            continue
        if selection.get("genericness_check") != "pass":
            continue
        if selection.get("self_contained_check") != "pass":
            continue
        if selection.get("word_count_check") != "pass":
            continue
        if not candidate_has_publishable_length(candidates[index].get("text", "")):
            continue

        candidate = dict(candidates[index])
        candidate.update({
            "quality_score": quality,
            "ranking_reason": selection.get("ranking_reason"),
            "ranking_model": model,
            "ranking_prompt_version": RANKING_PROMPT_VERSION,
        })
        selected.append(candidate)
        used_indices.add(index)

    selected.sort(key=lambda item: item.get("quality_score", 0), reverse=True)
    return selected


def _taxonomy_theme_names(conversation_taxonomy: str) -> set[str]:
    if not conversation_taxonomy:
        return set()
    try:
        taxonomy = json.loads(conversation_taxonomy)
    except (TypeError, json.JSONDecodeError):
        return set()
    names = {
        str(item.get("canonical_name") or "").strip()
        for item in taxonomy.get("active_theme_registry", [])
    }
    names.update(
        str(item.get("name") or "").strip()
        for item in taxonomy.get("themes", [])
    )
    return {name for name in names if name}


def theme_match_is_controlled(action: str, theme_name: str, conversation_taxonomy: str) -> bool:
    known = {name.casefold() for name in _taxonomy_theme_names(conversation_taxonomy)}
    normalized = str(theme_name or "").strip().casefold()
    if action == "existing_theme":
        return bool(normalized and normalized in known)
    if action == "propose_new":
        return bool(normalized and normalized not in known)
    return action == "abstain" and not normalized


def contextualize_and_map_quotes(
    selected_candidates,
    podcast,
    episode,
    client,
    conversation_taxonomy="",
):
    """Draft connective analysis after quote selection; never change the selection."""
    if not selected_candidates:
        return []

    model = os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol")
    reasoning_effort = os.environ.get("OPENAI_EDITORIAL_REASONING", "high")
    entity_connection = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "relationship": {"type": "string"},
            "description": {"type": "string"},
            "evidence_type": {
                "type": "string",
                "enum": [
                    "direct_transcript", "episode_metadata",
                    "speaker_identity", "editorial_connection",
                ],
            },
            "evidence": {"type": "string"},
            "segment_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": [
            "name", "relationship", "description", "evidence_type",
            "evidence", "segment_ids",
        ],
        "additionalProperties": False,
    }
    analysis_schema = {
        "type": "object",
        "properties": {
            "analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {"type": "integer"},
                        "editorial_context": {"type": "string"},
                        "context_confidence": {"type": "number"},
                        "genericness_check": {"type": "string", "enum": ["pass", "fail"]},
                        "context_evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "statement": {"type": "string"},
                                    "support": {"type": "string"},
                                    "evidence_type": {
                                        "type": "string",
                                        "enum": [
                                            "direct_transcript", "domain_inference",
                                            "editorial_judgment",
                                        ],
                                    },
                                    "segment_ids": {"type": "array", "items": {"type": "integer"}},
                                },
                                "required": ["statement", "support", "evidence_type", "segment_ids"],
                                "additionalProperties": False,
                            },
                        },
                        "theme_match_action": {
                            "type": "string",
                            "enum": ["existing_theme", "propose_new", "abstain"],
                        },
                        "theme_name": {"type": "string"},
                        "theme_summary": {"type": "string"},
                        "question_text": {"type": "string"},
                        "question_summary": {"type": "string"},
                        "connection_context": {"type": "string"},
                        "mapping_confidence": {"type": "number"},
                        "related_people": {"type": "array", "items": {"$ref": "#/$defs/entity_connection"}},
                        "related_companies": {"type": "array", "items": {"$ref": "#/$defs/entity_connection"}},
                        "speaker_title": {"type": "string"},
                        "speaker_company": {"type": "string"},
                        "speaker_metadata_source": {
                            "type": "string",
                            "enum": ["direct_transcript", "episode_metadata", "unknown"],
                        },
                    },
                    "required": [
                        "candidate_index", "editorial_context", "context_confidence",
                        "genericness_check", "context_evidence", "theme_match_action",
                        "theme_name", "theme_summary", "question_text",
                        "question_summary", "connection_context", "mapping_confidence",
                        "related_people", "related_companies", "speaker_title",
                        "speaker_company", "speaker_metadata_source",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["analyses"],
        "$defs": {"entity_connection": entity_connection},
        "additionalProperties": False,
    }
    compact = [
        {
            "candidate_index": index,
            "quote": candidate.get("text"),
            "speaker": candidate.get("speaker"),
            "source_segment_ids": [candidate.get("start_seg"), candidate.get("end_seg")],
            "source_excerpt": candidate.get("source_transcript_excerpt"),
            "ranking_reason": candidate.get("ranking_reason"),
        }
        for index, candidate in enumerate(selected_candidates)
    ]
    data = call_openai_structured(
        client,
        model=model,
        system_prompt="""
You are PodThreads' senior AdTech editor. The quote selection is already final;
never rerank, expand, shorten, or reject it. Draft context that connects the
speaker's idea to the industry's ongoing themes, questions, people, companies,
incentives, and operating history. Write from inside the industry. The posture is
connective, not fact-checking or prosecutorial. Avoid generic AI language,
importance announcements, verdicts, and claim/consequence templates.
""",
        user_prompt=f"""
Podcast: {podcast}
Episode: {episode}

For every selected take:
1. Write 45-90 words of `editorial_context` that adds specific industry context
   rather than paraphrasing the quote. Connect mechanisms, stakeholder tensions,
   related ideas, or a live operator debate. No hype.
2. Separate transcript facts from domain inference in `context_evidence`. Never
   invent statistics, corporate relationships, or speaker intent.
3. Map first to an exact active theme in the controlled registry. Use
   `existing_theme` only with its exact canonical name. Use `propose_new` only
   when no active theme can responsibly contain the idea. Use `abstain` and blank
   mapping fields when evidence is too thin. Never turn a category into a theme.
4. Put the take under one open question within the theme. Reuse an existing
   question verbatim when it is substantively the same.
5. Add people and companies only with labeled evidence. Speaker title and company
   must come from the transcript or episode metadata; otherwise return blank
   strings and `unknown`.

CONTROLLED THEME REGISTRY AND APPROVED GRAPH:
{conversation_taxonomy or "No controlled themes are active. Propose cautiously or abstain."}

SELECTED TAKES:
{json.dumps(compact, ensure_ascii=False)}
""",
        schema_name="podthreads_connective_analysis",
        schema=analysis_schema,
        reasoning_effort=reasoning_effort,
        max_output_tokens=10000,
    )

    analyses_by_index = {
        int(item.get("candidate_index", -1)): item
        for item in data.get("analyses", [])
        if 0 <= int(item.get("candidate_index", -1)) < len(selected_candidates)
    }
    minimum_context_confidence = float(os.environ.get("MIN_CONTEXT_CONFIDENCE", "0.72"))
    enriched = []
    for index, original in enumerate(selected_candidates):
        candidate = dict(original)
        analysis = analyses_by_index.get(index, {})
        start_segment = int(candidate.get("start_seg", -1))
        end_segment = int(candidate.get("end_seg", -1))
        evidence_items = analysis.get("context_evidence", [])
        context_text = str(analysis.get("editorial_context") or "").strip()
        context_confidence = max(
            0.0,
            min(1.0, float(analysis.get("context_confidence", 0) or 0)),
        )
        context_is_reviewable = (
            analysis.get("genericness_check") == "pass"
            and context_confidence >= minimum_context_confidence
            and 30 <= quote_word_count(context_text) <= 120
            and context_evidence_is_source_bounded(
                evidence_items,
                start_segment,
                end_segment,
            )
        )
        action = str(analysis.get("theme_match_action") or "abstain")
        theme_name = str(analysis.get("theme_name") or "").strip()
        controlled_action = theme_match_is_controlled(
            action,
            theme_name,
            conversation_taxonomy,
        )
        mapping_is_reviewable = controlled_action and action != "abstain" and conversation_mapping_is_reviewable(
            analysis,
            start_segment,
            end_segment,
        )
        metadata_source = analysis.get("speaker_metadata_source")
        candidate.update({
            "editorial_context": context_text if context_is_reviewable else None,
            "context_confidence": context_confidence if context_is_reviewable else 0.0,
            "context_evidence": evidence_items if context_is_reviewable else [],
            "context_model": model,
            "context_prompt_version": CONTEXT_PROMPT_VERSION,
            "theme_match_action": action if mapping_is_reviewable else "abstain",
            "theme_name": theme_name if mapping_is_reviewable else None,
            "theme_summary": analysis.get("theme_summary") if mapping_is_reviewable else None,
            "question_text": analysis.get("question_text") if mapping_is_reviewable else None,
            "question_summary": analysis.get("question_summary") if mapping_is_reviewable else None,
            "connection_context": analysis.get("connection_context") if mapping_is_reviewable else None,
            "mapping_confidence": (
                max(0.0, min(1.0, float(analysis.get("mapping_confidence", 0) or 0)))
                if mapping_is_reviewable else 0.0
            ),
            "related_people": analysis.get("related_people", []) if mapping_is_reviewable else [],
            "related_companies": analysis.get("related_companies", []) if mapping_is_reviewable else [],
            "mapping_model": model,
            "mapping_prompt_version": MAPPING_PROMPT_VERSION,
            "speaker_title": analysis.get("speaker_title") if metadata_source != "unknown" else None,
            "speaker_company": analysis.get("speaker_company") if metadata_source != "unknown" else None,
            "analysis_review_flags": {
                "context_reviewable": context_is_reviewable,
                "mapping_reviewable": mapping_is_reviewable,
                "controlled_theme_action": controlled_action,
            },
        })
        enriched.append(candidate)
    return enriched


def rank_and_contextualize_quotes(
    candidates,
    podcast,
    episode,
    client,
    curation_examples="",
    conversation_taxonomy="",
):
    """Compatibility wrapper for older callers; selection remains isolated."""
    ranked = rank_quote_candidates(
        candidates,
        podcast,
        episode,
        client,
        curation_examples=curation_examples,
    )
    return contextualize_and_map_quotes(
        ranked,
        podcast,
        episode,
        client,
        conversation_taxonomy=conversation_taxonomy,
    )


BAKEOFF_STRATEGY_MANIFEST = {
    "legacy_quality_bar": {
        "label": "Restored legacy quality bar",
        "prompt_version": "legacy-gpt4-quality-source-bounded-v1",
        "historical_model": "gpt-4-turbo-preview",
        "purpose": "Preserves the original 20-80 word, self-contained, exceptional-take rubric.",
    },
    "source_grounded_v2": {
        "label": "Pre-hybrid source-grounded v2",
        "prompt_version": "take-candidates-v2-snapshot",
        "purpose": "Captures the longer mechanism-first candidate behavior being replaced.",
    },
    "hybrid_v3": {
        "label": "PodThreads hybrid v3",
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "ranking_prompt_version": RANKING_PROMPT_VERSION,
        "purpose": "Combines the legacy taste bar with complete-transcript source controls.",
    },
}


def _bakeoff_candidate_schema():
    return {
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
                        "quality_score": {"type": "number"},
                        "extraction_reason": {"type": "string"},
                    },
                    "required": [
                        "text", "start_segment_id", "end_segment_id", "speaker",
                        "category", "quality_score", "extraction_reason",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def extract_bakeoff_baseline_candidates(
    text,
    podcast,
    episode,
    client,
    strategy_key,
    chunk_num=0,
):
    """Run frozen prompt baselines through the same strict source contract."""
    if strategy_key not in {"legacy_quality_bar", "source_grounded_v2"}:
        raise ValueError("Unsupported bakeoff baseline strategy")

    if strategy_key == "legacy_quality_bar":
        model = os.environ.get(
            "OPENAI_LEGACY_BASELINE_MODEL",
            os.environ.get("OPENAI_RANKING_MODEL", "gpt-5.6-sol"),
        )
        prompt_version = BAKEOFF_STRATEGY_MANIFEST[strategy_key]["prompt_version"]
        system_prompt = (
            "You are the original PodTakes curator. Extract only exceptional, "
            "thought-provoking quotes that challenge, surprise, or deeply illuminate. "
            "Quality over quantity. Literal source fidelity is mandatory."
        )
        criteria = """
- Quote 20-80 words; 30-50 is ideal.
- Self-contained and understandable without generated context.
- At least one of: hot take that challenges conventional wisdom,
  counterintuitive insight, specific and memorable example or prediction,
  thought-provoking framework, or surprising revelation.
- Reject generic advice, obvious statements, motivational platitudes, interview
  transitions, incomplete thoughts, vague philosophy, biography, and sales pitches.
"""
    else:
        model = os.environ.get("OPENAI_CANDIDATE_MODEL", "gpt-5.6-terra")
        prompt_version = BAKEOFF_STRATEGY_MANIFEST[strategy_key]["prompt_version"]
        system_prompt = (
            "You are the source-grounded v2 candidate retrieval layer. Prefer "
            "specific AdTech claims, mechanisms, incentives, and non-obvious implications."
        )
        criteria = """
- Prefer a specific prediction, causal claim, economic tradeoff, market-structure
  argument, counter-position, or reusable framework.
- The take should change an AdTech operator, publisher, marketer, agency,
  platform, investor, or regulator's decision or assumption.
- Penalize vague futurism, slogans, product pitches, biography, scene-setting,
  summaries, and generic business or AI commentary.
"""

    data = call_openai_structured(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=f"""
Podcast: {podcast}
Episode: {episode}
Transcript section: {chunk_num}

Select zero to five candidates. Zero is correct when no candidate clears the bar.
`text` must be copied from contiguous transcript segments and the segment IDs
must exactly bound its source. Do not rewrite the speaker.

{criteria}

TRANSCRIPT WITH GLOBAL SEGMENT IDS:
{text}
""",
        schema_name=f"podthreads_{strategy_key}_candidates",
        schema=_bakeoff_candidate_schema(),
        reasoning_effort=(
            os.environ.get("OPENAI_LEGACY_BASELINE_REASONING", "high")
            if strategy_key == "legacy_quality_bar"
            else os.environ.get("OPENAI_CANDIDATE_REASONING", "low")
        ),
        max_output_tokens=6000,
    )
    candidates = data.get("candidates", [])[:5]
    for candidate in candidates:
        candidate["quality_score"] = max(
            0.0,
            min(1.0, float(candidate.get("quality_score", 0))),
        )
        # Let the existing deterministic deduplicator preserve the strongest span.
        candidate["domain_specificity"] = candidate["quality_score"]
        candidate["novelty"] = candidate["quality_score"]
        candidate["provocation"] = candidate["quality_score"]
        candidate["evidence_quality"] = candidate["quality_score"]
        candidate["genericness_risk"] = 1 - candidate["quality_score"]
        candidate["extraction_model"] = model
        candidate["extraction_prompt_version"] = prompt_version
    return candidates


def ground_bakeoff_candidate(candidate, corrected_segments, raw_segments):
    """Return a source-bounded bakeoff item or ``None`` for an unsafe span."""
    try:
        start_id = int(candidate.get("start_segment_id"))
        end_id = int(candidate.get("end_segment_id"))
        if start_id < 0 or end_id < start_id or end_id >= len(corrected_segments):
            return None
        source_excerpt = " ".join(
            segment.get("text", "")
            for segment in corrected_segments[start_id:end_id + 1]
        ).strip()
        raw_excerpt = " ".join(
            segment.get("text", "")
            for segment in raw_segments[start_id:end_id + 1]
        ).strip()
        quote_text = str(candidate.get("text") or "").strip()
        quote_normalized = normalize_text(quote_text)
        source_normalized = normalize_text(source_excerpt)
        if not quote_normalized:
            return None
        if quote_normalized not in source_normalized:
            from difflib import SequenceMatcher
            if SequenceMatcher(None, quote_normalized, source_normalized).ratio() < 0.62:
                return None
        return {
            **candidate,
            "start_seg": start_id,
            "end_seg": end_id,
            "source_transcript_excerpt": source_excerpt,
            "raw_source_transcript_excerpt": raw_excerpt,
            "quote_word_count": quote_word_count(quote_text),
        }
    except (TypeError, ValueError):
        return None


@app.function(image=image, secrets=[my_secret], timeout=21600, cpu=2)
def run_extraction_bakeoff(
    episode_limit: int = 5,
    job_id: str = None,
    created_by: str = None,
):
    """Generate a blinded three-strategy bakeoff from saved episode artifacts."""
    from openai import OpenAI
    from supabase import create_client

    bounded_limit = max(1, min(int(episode_limit or 5), 15))
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    run_id = None
    register_pipeline_model_versions(supabase)
    update_processing_job(
        supabase,
        job_id,
        "claimed",
        claimed_at=utcnow_iso(),
        started_at=utcnow_iso(),
        progress={"phase": "loading_held_out_artifacts"},
    )
    try:
        artifacts_result = (
            supabase.table("episode_processing_artifacts")
            .select(
                "episode_guid,podcast_name,episode_name,transcript_segments,"
                "corrected_transcript_segments,transcript_corrections,pipeline_version,updated_at"
            )
            .eq("artifact_status", "complete")
            .order("updated_at", desc=True)
            .limit(bounded_limit)
            .execute()
        )
        artifacts = [
            artifact for artifact in (artifacts_result.data or [])
            if artifact.get("transcript_segments")
        ]
        if not artifacts:
            raise RuntimeError("No complete transcript artifacts are available for a bakeoff")

        dataset_fingerprint = hashlib.sha256(
            "|".join(sorted(str(row.get("episode_guid")) for row in artifacts)).encode("utf-8")
        ).hexdigest()[:16]
        thresholds = {
            "top5_sme_approval": 0.75,
            "source_alignment": 0.98,
            "speaker_accuracy": 0.98,
            "terminology_error_rate": 0.01,
            "maximum_median_words": IDEAL_QUOTE_WORDS_MAX,
        }
        inserted_run = supabase.table("extraction_bakeoff_runs").insert({
            "processing_job_id": job_id,
            "dataset_version": f"heldout-artifacts:{dataset_fingerprint}",
            "pipeline_version": PIPELINE_VERSION,
            "status": "running",
            "strategy_manifest": BAKEOFF_STRATEGY_MANIFEST,
            "episode_count": len(artifacts),
            "thresholds": thresholds,
            "created_by": created_by,
        }).execute()
        run_id = inserted_run.data[0]["id"]
        staged_items = []
        strategy_keys = list(BAKEOFF_STRATEGY_MANIFEST)
        total_steps = len(artifacts) * len(strategy_keys)
        completed_steps = 0

        for artifact in artifacts:
            raw_segments = artifact.get("transcript_segments") or []
            corrected_segments = artifact.get("corrected_transcript_segments") or raw_segments
            chunks = build_extraction_chunks(corrected_segments)
            for strategy_key in strategy_keys:
                all_candidates = []
                for chunk_index, chunk_text in enumerate(chunks, start=1):
                    if strategy_key == "hybrid_v3":
                        candidates = extract_quotes(
                            chunk_text,
                            artifact["podcast_name"],
                            artifact["episode_name"],
                            client,
                            chunk_num=chunk_index,
                            curation_examples="",
                        )
                    else:
                        candidates = extract_bakeoff_baseline_candidates(
                            chunk_text,
                            artifact["podcast_name"],
                            artifact["episode_name"],
                            client,
                            strategy_key,
                            chunk_num=chunk_index,
                        )
                    for candidate in candidates:
                        grounded = ground_bakeoff_candidate(
                            candidate,
                            corrected_segments,
                            raw_segments,
                        )
                        if grounded:
                            all_candidates.append(grounded)
                deduped = deduplicate_candidates(all_candidates)
                if strategy_key == "hybrid_v3":
                    finalists = rank_quote_candidates(
                        deduped[:30],
                        artifact["podcast_name"],
                        artifact["episode_name"],
                        client,
                        curation_examples="",
                    )[:5]
                else:
                    finalists = sorted(
                        deduped,
                        key=lambda item: float(item.get("quality_score", 0)),
                        reverse=True,
                    )[:5]

                for candidate_rank, candidate in enumerate(finalists, start=1):
                    blind_seed = (
                        f"{run_id}|{artifact['episode_guid']}|{strategy_key}|{candidate_rank}"
                    )
                    blind_label = hashlib.sha256(blind_seed.encode("utf-8")).hexdigest()[:8].upper()
                    staged_items.append({
                        "bakeoff_run_id": run_id,
                        "episode_guid": artifact["episode_guid"],
                        "podcast_name": artifact["podcast_name"],
                        "episode_name": artifact["episode_name"],
                        "blind_label": blind_label,
                        "strategy_key": strategy_key,
                        "candidate_rank": candidate_rank,
                        "quote_text": candidate.get("text"),
                        "speaker_name": candidate.get("speaker"),
                        "source_transcript_excerpt": candidate.get("source_transcript_excerpt"),
                        "raw_source_transcript_excerpt": candidate.get("raw_source_transcript_excerpt"),
                        "source_start_segment": candidate.get("start_seg"),
                        "source_end_segment": candidate.get("end_seg"),
                        "quote_word_count": quote_word_count(candidate.get("text", "")),
                        "extraction_model": candidate.get("extraction_model") or "unknown",
                        "extraction_prompt_version": candidate.get("extraction_prompt_version") or BAKEOFF_STRATEGY_MANIFEST[strategy_key]["prompt_version"],
                        "ranking_model": candidate.get("ranking_model"),
                        "ranking_prompt_version": candidate.get("ranking_prompt_version"),
                        "generated_score": candidate.get("quality_score"),
                        "metadata": {
                            "artifact_pipeline_version": artifact.get("pipeline_version"),
                            "transcript_correction_count": len(artifact.get("transcript_corrections") or []),
                        },
                    })
                completed_steps += 1
                update_processing_job(
                    supabase,
                    job_id,
                    "extracting",
                    progress={
                        "bakeoff_run_id": run_id,
                        "completed_strategy_episodes": completed_steps,
                        "total_strategy_episodes": total_steps,
                    },
                )

        if not staged_items:
            raise RuntimeError("Bakeoff strategies produced no source-grounded candidates")
        supabase.table("extraction_bakeoff_items").insert(staged_items).execute()
        supabase.table("extraction_bakeoff_runs").update({
            "status": "reviewing",
            "metrics": {"candidate_count": len(staged_items), "reviewed_count": 0},
        }).eq("id", run_id).execute()
        result = {
            "success": True,
            "bakeoff_run_id": run_id,
            "episode_count": len(artifacts),
            "candidate_count": len(staged_items),
            "status": "reviewing",
        }
        update_processing_job(
            supabase,
            job_id,
            "succeeded",
            result=result,
            completed_at=utcnow_iso(),
        )
        return result
    except Exception as exc:
        if run_id:
            try:
                supabase.table("extraction_bakeoff_runs").update({
                    "status": "failed",
                    "completed_at": utcnow_iso(),
                    "metrics": {"error": str(exc)[:1000]},
                }).eq("id", run_id).execute()
            except Exception as audit_exc:
                print(f"AUDIT_WARNING failed to close bakeoff run {run_id}: {audit_exc}")
        update_processing_job(
            supabase,
            job_id,
            "failed",
            error_code=type(exc).__name__,
            error_message=str(exc)[:4000],
            completed_at=utcnow_iso(),
        )
        raise


@app.function(image=image, secrets=[my_secret], timeout=3600, cpu=2)
def backfill_historical_conversation_mappings(
    limit: int = 12,
    quote_ids: list = None,
    job_id: str = None,
):
    """Stage source-bounded mappings for published takes; never approve or publish."""
    from openai import OpenAI
    from supabase import create_client

    bounded_limit = max(1, min(int(limit or 12), 50))
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    update_processing_job(
        supabase,
        job_id,
        "claimed",
        claimed_at=utcnow_iso(),
        started_at=utcnow_iso(),
        attempt_count=1,
        progress={"phase": "loading_published_takes", "limit": bounded_limit},
    )

    counts = {
        "considered": 0,
        "staged_unreviewed": 0,
        "abstained": 0,
        "source_unavailable": 0,
        "skipped_existing": 0,
        "retried_source_unavailable": 0,
    }
    try:
        existing_result = (
            supabase.table("conversation_mapping_reviews")
            .select("quote_id,workflow_status")
            .limit(5000)
            .execute()
        )
        existing = {
            str(row.get("quote_id")): row
            for row in (existing_result.data or [])
        }

        quote_query = (
            supabase.table("quotes")
            .select(
                "id,text,timestamp_start,timestamp_end,youtube_id,"
                "episode_id,guest_id,created_at"
            )
            .not_.is_("youtube_id", "null")
            .not_.is_("timestamp_start", "null")
            .not_.is_("timestamp_end", "null")
        )
        if quote_ids:
            quote_query = quote_query.in_("id", [str(value) for value in quote_ids])
        rows_result = quote_query.order("created_at", desc=True).limit(500).execute()
        candidates = []
        for row in rows_result.data or []:
            prior = existing.get(str(row.get("id")))
            prior_status = (prior or {}).get("workflow_status")
            if prior_status and prior_status != "source_unavailable":
                counts["skipped_existing"] += 1
                continue
            if prior_status == "source_unavailable":
                counts["retried_source_unavailable"] += 1
            candidates.append(row)
            if len(candidates) >= bounded_limit:
                break

        episode_ids = sorted({row.get("episode_id") for row in candidates if row.get("episode_id")})
        guest_ids = sorted({row.get("guest_id") for row in candidates if row.get("guest_id")})
        episodes = {}
        if episode_ids:
            result = (
                supabase.table("episodes")
                .select("id,title,podcast_id")
                .in_("id", episode_ids)
                .execute()
            )
            episodes = {str(row["id"]): row for row in (result.data or [])}
        podcast_ids = sorted({row.get("podcast_id") for row in episodes.values() if row.get("podcast_id")})
        podcasts = {}
        if podcast_ids:
            result = supabase.table("podcasts").select("id,name").in_("id", podcast_ids).execute()
            podcasts = {str(row["id"]): row for row in (result.data or [])}
        guests = {}
        if guest_ids:
            result = (
                supabase.table("guests")
                .select("id,name,title,company")
                .in_("id", guest_ids)
                .execute()
            )
            guests = {str(row["id"]): row for row in (result.data or [])}

        feed_rows = (
            supabase.table("test_podcast_feeds")
            .select("name,rss_url,active")
            .eq("active", True)
            .execute()
        ).data or []
        taxonomy = fetch_conversation_taxonomy(supabase)
        model = os.environ.get(
            "OPENAI_RANKING_MODEL",
            os.environ.get("OPENAI_EDITORIAL_MODEL", "gpt-5.6-sol"),
        )
        for index, raw_quote in enumerate(candidates):
            counts["considered"] += 1
            quote_id = str(raw_quote["id"])
            episode = episodes.get(str(raw_quote.get("episode_id")), {})
            podcast = podcasts.get(str(episode.get("podcast_id")), {})
            guest = guests.get(str(raw_quote.get("guest_id")), {})
            quote = {
                **raw_quote,
                "speaker_name": guest.get("name"),
                "speaker_title": guest.get("title"),
                "speaker_company": guest.get("company"),
                "episode_name": episode.get("title"),
                "podcast_name": podcast.get("name"),
            }
            update_processing_job(
                supabase,
                job_id,
                "mapping",
                progress={
                    "phase": "mapping_historical_takes",
                    "current": index + 1,
                    "total": len(candidates),
                    "quote_id": quote_id,
                    **counts,
                },
            )

            captions = get_yt_captions(str(quote.get("youtube_id")))
            source_url = (
                f"https://www.youtube.com/watch?v={quote.get('youtube_id')}"
                f"&t={max(0, int(float(quote.get('timestamp_start') or 0)))}s"
            )
            source_kind = "youtube_captions"
            aligned = None
            source_failure = None
            if captions:
                aligned = align_timestamps_to_youtube_captions(
                    str(quote.get("text") or ""),
                    str(quote.get("youtube_id")),
                    int(float(quote.get("timestamp_start") or 0)),
                    int(float(quote.get("timestamp_end") or 0)),
                )
            else:
                try:
                    rss_source = resolve_rss_audio_source(
                        quote.get("podcast_name"),
                        quote.get("episode_name"),
                        feed_rows,
                    )
                    if rss_source:
                        source_kind = "rss_audio_transcript"
                        source_url = rss_source["audio_url"]
                        captions = transcribe_remote_audio_window(
                            source_url,
                            float(quote.get("timestamp_start") or 0),
                            float(quote.get("timestamp_end") or 0),
                            client,
                        )
                        aligned = align_quote_to_segments(
                            str(quote.get("text") or ""),
                            captions,
                            float(quote.get("timestamp_start") or 0),
                            float(quote.get("timestamp_end") or 0),
                        )
                    else:
                        source_failure = "No matching RSS audio enclosure"
                except Exception as exc:
                    source_failure = f"RSS audio fallback failed: {exc}"
                    print(f"  ⚠️  {source_failure}")
            if not captions:
                supabase.table("conversation_mapping_reviews").upsert({
                    "quote_id": quote_id,
                    "processing_job_id": job_id,
                    "source_kind": source_kind,
                    "source_url": source_url,
                    "workflow_status": "source_unavailable",
                    "abstention_reason": source_failure or "No retrievable caption or RSS audio source",
                    "mapping_model": model,
                    "mapping_prompt_version": HISTORICAL_MAPPING_PROMPT_VERSION,
                    "updated_at": utcnow_iso(),
                }, on_conflict="quote_id").execute()
                counts["source_unavailable"] += 1
                continue

            evidence_start = aligned.get("start") if aligned else float(quote.get("timestamp_start") or 0)
            evidence_end = aligned.get("end") if aligned else float(quote.get("timestamp_end") or 0)
            source_evidence = build_caption_evidence(captions, evidence_start, evidence_end)
            if not aligned or not source_evidence:
                fallback = source_evidence or {"segments": [], "excerpt": None}
                supabase.table("conversation_mapping_reviews").upsert({
                    "quote_id": quote_id,
                    "processing_job_id": job_id,
                    "source_kind": source_kind,
                    "source_url": source_url,
                    "source_transcript_excerpt": fallback.get("excerpt"),
                    "source_segments": fallback.get("segments", []),
                    "workflow_status": "abstained",
                    "abstention_reason": "Published quote could not be aligned to captions with sufficient confidence",
                    "mapping_model": model,
                    "mapping_prompt_version": HISTORICAL_MAPPING_PROMPT_VERSION,
                    "updated_at": utcnow_iso(),
                }, on_conflict="quote_id").execute()
                counts["abstained"] += 1
                continue

            mapping = propose_historical_conversation_mapping(
                quote,
                source_evidence,
                client,
                conversation_taxonomy=taxonomy,
            )
            reviewable = historical_mapping_is_reviewable(
                mapping,
                source_evidence["start_segment"],
                source_evidence["end_segment"],
            )
            record = {
                "quote_id": quote_id,
                "processing_job_id": job_id,
                "source_kind": source_kind,
                "source_url": source_url,
                "source_transcript_excerpt": source_evidence["excerpt"],
                "source_start_segment": source_evidence["start_segment"],
                "source_end_segment": source_evidence["end_segment"],
                "source_segments": source_evidence["segments"],
                "source_alignment_confidence": aligned.get("confidence"),
                "proposed_theme_name": mapping.get("theme_name") or None,
                "proposed_theme_summary": mapping.get("theme_summary") or None,
                "proposed_question_text": mapping.get("question_text") or None,
                "proposed_question_summary": mapping.get("question_summary") or None,
                "proposed_people": mapping.get("related_people") or [],
                "proposed_companies": mapping.get("related_companies") or [],
                "proposed_relationship_label": mapping.get("relationship_label") or None,
                "connection_context": mapping.get("connection_context") or None,
                "mapping_confidence": max(0.0, min(1.0, float(mapping.get("mapping_confidence", 0)))),
                "mapping_model": model,
                "mapping_prompt_version": HISTORICAL_MAPPING_PROMPT_VERSION,
                "workflow_status": "unreviewed" if reviewable else "abstained",
                "abstention_reason": None if reviewable else (
                    mapping.get("abstention_reason")
                    or "Proposal did not clear the source-bounded mapping quality gate"
                ),
                "updated_at": utcnow_iso(),
            }
            supabase.table("conversation_mapping_reviews").upsert(
                record,
                on_conflict="quote_id",
            ).execute()
            if reviewable:
                counts["staged_unreviewed"] += 1
            else:
                counts["abstained"] += 1

        result = {"success": True, "limit": bounded_limit, **counts}
        update_processing_job(
            supabase,
            job_id,
            "succeeded",
            progress={"phase": "complete", **counts},
            result=result,
            completed_at=utcnow_iso(),
        )
        return result
    except Exception as exc:
        update_processing_job(
            supabase,
            job_id,
            "failed",
            result={"success": False, **counts},
            error_code="historical_mapping_failed",
            error_message=str(exc),
            completed_at=utcnow_iso(),
        )
        raise


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
            "rubric_version": EDITORIAL_RUBRIC_VERSION,
            "status": "active",
            "configuration": {
                "reasoning_effort": os.environ.get(
                    "OPENAI_RANKING_REASONING",
                    os.environ.get("OPENAI_EDITORIAL_REASONING", "high"),
                ),
                "minimum_quality": float(os.environ.get("MIN_QUOTE_QUALITY", "0.74")),
                "quote_length_words": [MIN_QUOTE_WORDS, MAX_QUOTE_WORDS],
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
            selections = rank_quote_candidates(
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
    automation_enabled = bool(setting.data and setting.data[0].get('value') is True)
    if not automation_enabled:
        print("⏸️ Automated processing is disabled or not explicitly configured")
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
            "ranking", "mapping", "staging", "succeeded"
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


def ensure_draft_gold_set(supabase, created_by=None):
    existing = (
        supabase.table("editorial_gold_sets")
        .select("*")
        .eq("status", "drafting")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if existing.data:
        return existing.data[0]
    version = f"podthreads-gold-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    inserted = supabase.table("editorial_gold_sets").insert({
        "name": "PodThreads AdTech Editorial Gold Set",
        "version": version,
        "rubric_version": EDITORIAL_RUBRIC_VERSION,
        "status": "drafting",
        "target_positive_count": 60,
        "target_negative_count": 40,
        "notes": "Explicit SME choices only; pending legacy candidates are excluded.",
        "created_by": created_by,
    }).execute()
    return inserted.data[0]


def add_gold_set_item(
    supabase,
    *,
    created_by,
    label,
    preferred_quote_text,
    source_transcript_excerpt,
    rationale,
    reviewer_expertise,
    failure_codes=None,
    test_quote_id=None,
    published_quote_id=None,
    bakeoff_item_id=None,
):
    gold_set = ensure_draft_gold_set(supabase, created_by=created_by)
    failure_codes = failure_codes or []
    payload = {
        "gold_set_id": gold_set["id"],
        "test_quote_id": test_quote_id,
        "published_quote_id": published_quote_id,
        "bakeoff_item_id": bakeoff_item_id,
        "label": label,
        "preferred_quote_text": preferred_quote_text,
        "source_transcript_excerpt": source_transcript_excerpt,
        "rationale": rationale,
        "failure_codes": failure_codes,
        "reviewer_expertise": reviewer_expertise or [],
        "source_alignment_verified": "source_alignment" not in failure_codes,
        "terminology_verified": "terminology" not in failure_codes,
        "speaker_verified": "speaker_attribution" not in failure_codes,
        "created_by": created_by,
    }
    return supabase.table("editorial_gold_set_items").insert(payload).execute().data[0]


def latest_bakeoff_review_map(review_rows):
    latest = {}
    for review in sorted(
        review_rows or [],
        key=lambda row: str(row.get("created_at") or ""),
        reverse=True,
    ):
        latest.setdefault(review.get("bakeoff_item_id"), review)
    return latest


def calculate_bakeoff_metrics(items, review_rows):
    from statistics import median

    latest_reviews = latest_bakeoff_review_map(review_rows)
    by_strategy = {}
    for item in items or []:
        strategy = item.get("strategy_key")
        bucket = by_strategy.setdefault(strategy, {
            "candidate_count": 0,
            "reviewed_count": 0,
            "approved_count": 0,
            "ratings": [],
            "word_counts": [],
            "edited_count": 0,
            "preferred_count": 0,
            "failure_counts": {},
        })
        bucket["candidate_count"] += 1
        bucket["word_counts"].append(int(item.get("quote_word_count") or 0))
        review = latest_reviews.get(item.get("id"))
        if not review:
            continue
        bucket["reviewed_count"] += 1
        bucket["approved_count"] += int(review.get("decision") == "approve")
        bucket["ratings"].append(int(review.get("quality_rating") or 0))
        bucket["edited_count"] += int(bool(str(review.get("edited_quote_text") or "").strip()))
        bucket["preferred_count"] += int(bool(review.get("preferred_in_episode")))
        for failure_code in review.get("failure_codes") or []:
            bucket["failure_counts"][failure_code] = bucket["failure_counts"].get(failure_code, 0) + 1

    metrics = {
        "candidate_count": len(items or []),
        "reviewed_count": len(latest_reviews),
        "review_coverage": round(len(latest_reviews) / max(len(items or []), 1), 4),
        "strategies": {},
    }
    for strategy, bucket in by_strategy.items():
        reviewed = bucket["reviewed_count"]
        failures = bucket["failure_counts"]
        metrics["strategies"][strategy] = {
            "candidate_count": bucket["candidate_count"],
            "reviewed_count": reviewed,
            "approval_rate": round(bucket["approved_count"] / max(reviewed, 1), 4),
            "average_rating": round(sum(bucket["ratings"]) / max(len(bucket["ratings"]), 1), 3),
            "edit_rate": round(bucket["edited_count"] / max(reviewed, 1), 4),
            "preferred_count": bucket["preferred_count"],
            "median_words": median(bucket["word_counts"]) if bucket["word_counts"] else 0,
            "source_alignment": round(1 - failures.get("source_alignment", 0) / max(reviewed, 1), 4),
            "speaker_accuracy": round(1 - failures.get("speaker_attribution", 0) / max(reviewed, 1), 4),
            "terminology_error_rate": round(failures.get("terminology", 0) / max(reviewed, 1), 4),
            "generic_rejection_rate": round(failures.get("generic", 0) / max(reviewed, 1), 4),
            "failure_counts": failures,
        }
    return metrics

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

    web_app = FastAPI(title="PodThreads Admin API", docs_url=None, redoc_url=None)
    bearer_scheme = HTTPBearer(auto_error=False)

    allowed_origins = [
        value.strip()
        for value in os.environ.get(
            "ALLOWED_ORIGINS",
            "https://podthreads.com,https://www.podthreads.com,https://getpodtakes.com,https://www.getpodtakes.com,https://podtakes.com,https://www.podtakes.com,http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if value.strip()
    ]
    
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=(
            r"^https://(?:"
            r"(?:[a-z0-9-]+--)?[a-z0-9-]+\.netlify\.app"
            r"|[a-z0-9-]+\.bolt\.host"
            r"|(?:[a-z0-9-]+\.)*webcontainer(?:-api)?\.io"
            r")$"
        ),
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
        proposed_theme_name: str | None = None
        proposed_theme_summary: str | None = None
        proposed_question_text: str | None = None
        proposed_question_summary: str | None = None
        proposed_people: list[dict] | None = None
        proposed_companies: list[dict] | None = None
        connection_context: str | None = None
        theme_match_action: str | None = None
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
        add_to_gold_set: bool = False
        gold_rationale: str | None = None
        gold_failure_codes: list[str] = Field(default_factory=list)

    class AutomationRequest(BaseModel):
        enabled: bool

    class FeedStateRequest(BaseModel):
        active: bool

    class FeedCreateRequest(BaseModel):
        name: str = Field(min_length=2, max_length=200)
        rss_url: str = Field(min_length=8, max_length=2000)

    class EvaluationRequest(BaseModel):
        sample_limit: int = Field(default=40, ge=12, le=100)

    class BakeoffRunRequest(BaseModel):
        episode_limit: int = Field(default=5, ge=1, le=15)

    class BakeoffReviewRequest(BaseModel):
        bakeoff_item_id: str
        decision: str
        quality_rating: int = Field(ge=1, le=5)
        preferred_in_episode: bool = False
        edited_quote_text: str | None = None
        failure_codes: list[str] = Field(default_factory=list)
        notes: str | None = None
        reviewer_expertise: list[str] = Field(default_factory=list)
        add_to_gold_set: bool = False
        supersedes_review_id: str | None = None

    class BakeoffRevealRequest(BaseModel):
        bakeoff_run_id: str

    class ThemeRegistryRequest(BaseModel):
        theme_registry_id: str | None = None
        action: str
        canonical_name: str | None = None
        definition: str | None = None
        aliases: list[str] | None = None
        inclusion_criteria: list[str] | None = None
        exclusion_criteria: list[str] | None = None
        positive_examples: list[str] | None = None
        counter_examples: list[str] | None = None
        reason: str
        reviewer_expertise: list[str] = Field(default_factory=list)

    class GoldSetLockRequest(BaseModel):
        gold_set_id: str

    class HistoricalBackfillRequest(BaseModel):
        limit: int = Field(default=12, ge=1, le=50)
        quote_ids: list[str] | None = None

    class HistoricalMappingReviewRequest(BaseModel):
        mapping_review_id: str
        action: str
        proposed_theme_name: str | None = None
        proposed_theme_summary: str | None = None
        proposed_question_text: str | None = None
        proposed_question_summary: str | None = None
        proposed_people: list[dict] | None = None
        proposed_companies: list[dict] | None = None
        proposed_relationship_label: str | None = None
        connection_context: str | None = None
        reason_code: str | None = None
        reason_detail: str | None = None
        reviewer_expertise: list[str] = Field(default_factory=list)
        target_decision_id: str | None = None

    class HistoricalMappingPublishRequest(BaseModel):
        mapping_review_id: str

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
            "preview_cors": "netlify-bolt-webcontainer",
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

    @web_app.post("/bakeoffs/run")
    async def run_bakeoff_endpoint(req: BakeoffRunRequest, admin=Depends(require_admin)):
        supabase = service_client()
        active = (
            supabase.table("extraction_bakeoff_runs")
            .select("id,status")
            .in_("status", ["running", "reviewing"])
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )
        if active.data:
            raise HTTPException(
                status_code=409,
                detail="Complete or reveal the active bakeoff before starting another.",
            )
        inserted = supabase.table("processing_jobs").insert({
            "idempotency_key": f"extraction-bakeoff:{uuid.uuid4()}",
            "job_type": "extraction_bakeoff",
            "source": "admin",
            "requested_by": admin["id"],
            "parameters": {"episode_limit": req.episode_limit, "blinded": True},
        }).execute()
        job_id = inserted.data[0]["id"]
        function_call = await run_extraction_bakeoff.spawn.aio(
            episode_limit=req.episode_limit,
            job_id=job_id,
            created_by=admin["id"],
        )
        supabase.table("processing_jobs").update({
            "modal_call_id": function_call.object_id,
            "updated_at": utcnow_iso(),
        }).eq("id", job_id).execute()
        return JSONResponse(
            status_code=202,
            content={"success": True, "job_id": job_id, "state": "queued"},
        )

    @web_app.get("/bakeoffs/latest")
    async def latest_bakeoff_endpoint(admin=Depends(require_admin)):
        supabase = service_client()
        run_result = (
            supabase.table("extraction_bakeoff_runs")
            .select("*")
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )
        if not run_result.data:
            return {"success": True, "run": None, "items": [], "reviews": []}
        run = run_result.data[0]
        items_result = (
            supabase.table("extraction_bakeoff_items")
            .select("*")
            .eq("bakeoff_run_id", run["id"])
            .order("episode_name")
            .order("blind_label")
            .execute()
        )
        items = items_result.data or []
        item_ids = [item["id"] for item in items]
        reviews = []
        if item_ids:
            review_result = (
                supabase.table("extraction_bakeoff_reviews")
                .select("*")
                .in_("bakeoff_item_id", item_ids)
                .eq("reviewer_id", admin["id"])
                .order("created_at", desc=True)
                .execute()
            )
            reviews = list(latest_bakeoff_review_map(review_result.data or []).values())

        if run.get("blinded", True):
            run = {
                key: value for key, value in run.items()
                if key not in {"strategy_manifest"}
            }
            metrics = run.get("metrics") or {}
            run["metrics"] = {
                "candidate_count": metrics.get("candidate_count", len(items)),
                "reviewed_count": len(reviews),
                "review_coverage": round(len(reviews) / max(len(items), 1), 4),
            }
            redacted_keys = {
                "strategy_key", "extraction_model", "extraction_prompt_version",
                "ranking_model", "ranking_prompt_version", "generated_score",
            }
            items = [
                {key: value for key, value in item.items() if key not in redacted_keys}
                for item in items
            ]
        return {"success": True, "run": run, "items": items, "reviews": reviews}

    @web_app.post("/bakeoffs/review")
    async def review_bakeoff_endpoint(req: BakeoffReviewRequest, admin=Depends(require_admin)):
        if req.decision not in {"approve", "reject"}:
            raise HTTPException(status_code=422, detail="Decision must be approve or reject")
        if not req.reviewer_expertise:
            raise HTTPException(status_code=422, detail="Reviewer expertise is required")
        allowed_failures = {
            "generic", "too_long", "too_short", "incomplete_thought",
            "source_alignment", "speaker_attribution", "terminology",
            "not_adtech_specific", "low_signal", "other",
        }
        unknown_failures = set(req.failure_codes) - allowed_failures
        if unknown_failures:
            raise HTTPException(status_code=422, detail=f"Unknown failure codes: {sorted(unknown_failures)}")
        if req.decision == "reject" and not req.failure_codes and not (req.notes or "").strip():
            raise HTTPException(status_code=422, detail="A rejected take needs a failure code or note")
        if req.add_to_gold_set and not (req.notes or "").strip():
            raise HTTPException(status_code=422, detail="Gold-set examples require an editorial rationale")

        supabase = service_client()
        item_result = (
            supabase.table("extraction_bakeoff_items")
            .select("*,extraction_bakeoff_runs(status,blinded)")
            .eq("id", req.bakeoff_item_id)
            .single()
            .execute()
        )
        if not item_result.data:
            raise HTTPException(status_code=404, detail="Bakeoff item not found")
        item = item_result.data
        run_state = item.get("extraction_bakeoff_runs") or {}
        if run_state.get("status") != "reviewing" or not run_state.get("blinded", True):
            raise HTTPException(status_code=409, detail="This bakeoff is not open for blinded review")

        if req.supersedes_review_id:
            prior = (
                supabase.table("extraction_bakeoff_reviews")
                .select("id")
                .eq("id", req.supersedes_review_id)
                .eq("bakeoff_item_id", req.bakeoff_item_id)
                .eq("reviewer_id", admin["id"])
                .limit(1)
                .execute()
            )
            if not prior.data:
                raise HTTPException(status_code=422, detail="Superseded review does not match this item and reviewer")

        inserted = supabase.table("extraction_bakeoff_reviews").insert({
            "bakeoff_item_id": req.bakeoff_item_id,
            "reviewer_id": admin["id"],
            "decision": req.decision,
            "quality_rating": req.quality_rating,
            "preferred_in_episode": req.preferred_in_episode,
            "edited_quote_text": (req.edited_quote_text or "").strip() or None,
            "failure_codes": req.failure_codes,
            "notes": (req.notes or "").strip() or None,
            "reviewer_expertise": req.reviewer_expertise,
            "supersedes_review_id": req.supersedes_review_id,
        }).execute()
        review = inserted.data[0]
        gold_item = None
        gold_warning = None
        if req.add_to_gold_set:
            rationale = (req.notes or "").strip()
            try:
                gold_item = add_gold_set_item(
                    supabase,
                    created_by=admin["id"],
                    label="positive" if req.decision == "approve" else "negative",
                    preferred_quote_text=(req.edited_quote_text or item["quote_text"]).strip(),
                    source_transcript_excerpt=item.get("source_transcript_excerpt"),
                    rationale=rationale,
                    reviewer_expertise=req.reviewer_expertise,
                    failure_codes=req.failure_codes,
                    bakeoff_item_id=req.bakeoff_item_id,
                )
            except Exception as exc:
                if "duplicate key" not in str(exc).lower() and "23505" not in str(exc):
                    gold_warning = str(exc)

        return {
            "success": True,
            "review": review,
            "gold_item": gold_item,
            "gold_warning": gold_warning,
        }

    @web_app.post("/bakeoffs/reveal")
    async def reveal_bakeoff_endpoint(req: BakeoffRevealRequest, admin=Depends(require_admin)):
        supabase = service_client()
        run_result = (
            supabase.table("extraction_bakeoff_runs")
            .select("*")
            .eq("id", req.bakeoff_run_id)
            .single()
            .execute()
        )
        if not run_result.data:
            raise HTTPException(status_code=404, detail="Bakeoff run not found")
        run = run_result.data
        items = (
            supabase.table("extraction_bakeoff_items")
            .select("*")
            .eq("bakeoff_run_id", req.bakeoff_run_id)
            .execute()
        ).data or []
        item_ids = [item["id"] for item in items]
        reviews = (
            supabase.table("extraction_bakeoff_reviews")
            .select("*")
            .in_("bakeoff_item_id", item_ids)
            .eq("reviewer_id", admin["id"])
            .order("created_at", desc=True)
            .execute()
        ).data or [] if item_ids else []
        metrics = calculate_bakeoff_metrics(items, reviews)
        if metrics["review_coverage"] < 1:
            raise HTTPException(
                status_code=409,
                detail="Review every blinded candidate before revealing strategy identity.",
            )
        hybrid = metrics.get("strategies", {}).get("hybrid_v3", {})
        thresholds = run.get("thresholds") or {}
        release_gate_passed = bool(hybrid) and all([
            hybrid.get("approval_rate", 0) >= thresholds.get("top5_sme_approval", 0.75),
            hybrid.get("source_alignment", 0) >= thresholds.get("source_alignment", 0.98),
            hybrid.get("speaker_accuracy", 0) >= thresholds.get("speaker_accuracy", 0.98),
            hybrid.get("terminology_error_rate", 1) <= thresholds.get("terminology_error_rate", 0.01),
            hybrid.get("median_words", 999) <= thresholds.get("maximum_median_words", 50),
        ])
        metrics["release_gate_passed"] = release_gate_passed
        updated = (
            supabase.table("extraction_bakeoff_runs")
            .update({
                "status": "completed",
                "blinded": False,
                "revealed_at": utcnow_iso(),
                "completed_at": utcnow_iso(),
                "metrics": metrics,
            })
            .eq("id", req.bakeoff_run_id)
            .execute()
        )
        return {
            "success": True,
            "run": updated.data[0] if updated.data else {**run, "metrics": metrics, "blinded": False},
            "activation_changed": False,
        }

    @web_app.get("/theme-registry")
    async def get_theme_registry_endpoint(admin=Depends(require_admin)):
        supabase = service_client()
        themes = (
            supabase.table("adtech_theme_registry")
            .select("*")
            .order("status")
            .order("canonical_name")
            .execute()
        ).data or []
        return {"success": True, "themes": themes}

    @web_app.get("/editorial-taxonomy")
    async def get_editorial_taxonomy_endpoint(admin=Depends(require_admin)):
        """Return private, review-safe suggestions for the take curation workspace."""
        supabase = service_client()
        registry = (
            supabase.table("adtech_theme_registry")
            .select(
                "id,canonical_name,definition,aliases,inclusion_criteria,"
                "exclusion_criteria,status,registry_version"
            )
            .eq("status", "active")
            .order("canonical_name")
            .execute()
        ).data or []
        graph_themes = (
            supabase.table("conversation_themes")
            .select("id,name,summary,publication_status")
            .order("name")
            .execute()
        ).data or []
        graph_questions = (
            supabase.table("conversation_questions")
            .select("id,theme_id,question_text,summary,publication_status")
            .order("question_text")
            .execute()
        ).data or []
        staged_rows = (
            supabase.table("test_quotes")
            .select(
                "proposed_theme_name,proposed_theme_summary,"
                "proposed_question_text,proposed_question_summary,"
                "mapping_review_status"
            )
            .in_("approval_status", ["approved", "promoted"])
            .limit(5000)
            .execute()
        ).data or []
        categories = (
            supabase.table("categories")
            .select("id,name,description")
            .order("name")
            .execute()
        ).data or []
        people = (
            supabase.table("guests")
            .select("id,name,title,company,linkedin_url")
            .order("name")
            .limit(2000)
            .execute()
        ).data or []
        graph_entities = (
            supabase.table("conversation_entities")
            .select("entity_type,name,description,publication_status")
            .order("name")
            .limit(2000)
            .execute()
        ).data or []

        theme_name_by_id = {
            str(theme.get("id")): str(theme.get("name") or "").strip()
            for theme in graph_themes
        }
        question_keys = set()
        questions = []
        for question in graph_questions:
            if question.get("publication_status") not in {"approved", "published"}:
                continue
            theme_name = theme_name_by_id.get(str(question.get("theme_id")), "")
            question_text = str(question.get("question_text") or "").strip()
            if not theme_name or not question_text:
                continue
            key = (theme_name.casefold(), question_text.casefold())
            if key in question_keys:
                continue
            question_keys.add(key)
            questions.append({
                "id": question.get("id"),
                "theme_name": theme_name,
                "question_text": question_text,
                "summary": question.get("summary"),
                "source": "published_graph",
                "publication_status": question.get("publication_status"),
            })
        for row in staged_rows:
            if row.get("mapping_review_status") != "approved":
                continue
            theme_name = str(row.get("proposed_theme_name") or "").strip()
            question_text = str(row.get("proposed_question_text") or "").strip()
            if not theme_name or not question_text:
                continue
            key = (theme_name.casefold(), question_text.casefold())
            if key in question_keys:
                continue
            question_keys.add(key)
            questions.append({
                "id": f"staged:{hashlib.sha256('|'.join(key).encode('utf-8')).hexdigest()[:16]}",
                "theme_name": theme_name,
                "question_text": question_text,
                "summary": row.get("proposed_question_summary"),
                "source": "staged_review",
                "publication_status": "staged",
            })

        company_by_name = {}
        for person in people:
            company = str(person.get("company") or "").strip()
            if company:
                company_by_name.setdefault(company.casefold(), {
                    "name": company,
                    "description": "Appears in verified speaker metadata.",
                    "source": "speaker_metadata",
                })
        for entity in graph_entities:
            if entity.get("entity_type") != "company":
                continue
            name = str(entity.get("name") or "").strip()
            if name:
                company_by_name[name.casefold()] = {
                    "name": name,
                    "description": entity.get("description"),
                    "source": "conversation_graph",
                }

        return {
            "success": True,
            "themes": registry,
            "questions": sorted(
                questions,
                key=lambda item: (item["theme_name"].casefold(), item["question_text"].casefold()),
            ),
            "categories": categories,
            "people": people,
            "companies": sorted(company_by_name.values(), key=lambda item: item["name"].casefold()),
        }

    @web_app.post("/theme-registry")
    async def update_theme_registry_endpoint(req: ThemeRegistryRequest, admin=Depends(require_admin)):
        allowed_actions = {"create", "edit", "activate", "retire", "restore"}
        if req.action not in allowed_actions:
            raise HTTPException(status_code=422, detail="Unsupported theme registry action")
        if not req.reviewer_expertise:
            raise HTTPException(status_code=422, detail="Reviewer expertise is required")
        if not req.reason.strip():
            raise HTTPException(status_code=422, detail="An audit reason is required")
        supabase = service_client()
        before = {}
        if req.action != "create":
            if not req.theme_registry_id:
                raise HTTPException(status_code=422, detail="theme_registry_id is required")
            current = (
                supabase.table("adtech_theme_registry")
                .select("*")
                .eq("id", req.theme_registry_id)
                .single()
                .execute()
            )
            if not current.data:
                raise HTTPException(status_code=404, detail="Theme registry entry not found")
            before = current.data

        editable = {
            "canonical_name": req.canonical_name,
            "definition": req.definition,
            "aliases": req.aliases,
            "inclusion_criteria": req.inclusion_criteria,
            "exclusion_criteria": req.exclusion_criteria,
            "positive_examples": req.positive_examples,
            "counter_examples": req.counter_examples,
        }
        updates = {key: value for key, value in editable.items() if value is not None}
        if req.action == "create":
            if not (req.canonical_name or "").strip() or not (req.definition or "").strip():
                raise HTTPException(status_code=422, detail="Canonical name and definition are required")
            updates.update({
                "canonical_name": req.canonical_name.strip(),
                "definition": req.definition.strip(),
                "status": "proposed",
                "metadata": {"created_by_admin_api": True},
            })
            inserted = supabase.table("adtech_theme_registry").insert(updates).execute()
            after = inserted.data[0]
        else:
            if req.action in {"activate", "restore"}:
                candidate = {**before, **updates}
                if not candidate.get("inclusion_criteria") or not candidate.get("exclusion_criteria"):
                    raise HTTPException(
                        status_code=422,
                        detail="Active themes require inclusion and exclusion criteria.",
                    )
                updates.update({"status": "active", "reviewed_by": admin["id"], "reviewed_at": utcnow_iso()})
            elif req.action == "retire":
                updates.update({"status": "retired", "reviewed_by": admin["id"], "reviewed_at": utcnow_iso()})
            updates["updated_at"] = utcnow_iso()
            changed = (
                supabase.table("adtech_theme_registry")
                .update(updates)
                .eq("id", req.theme_registry_id)
                .execute()
            )
            after = changed.data[0] if changed.data else {**before, **updates}

        decision = supabase.table("theme_registry_decisions").insert({
            "theme_registry_id": after["id"],
            "reviewer_id": admin["id"],
            "decision": req.action,
            "before_state": before,
            "after_state": after,
            "reason": req.reason.strip(),
        }).execute()
        return {"success": True, "theme": after, "decision_id": decision.data[0]["id"]}

    @web_app.post("/gold-sets/lock")
    async def lock_gold_set_endpoint(req: GoldSetLockRequest, admin=Depends(require_admin)):
        supabase = service_client()
        gold_set_result = (
            supabase.table("editorial_gold_sets")
            .select("*")
            .eq("id", req.gold_set_id)
            .single()
            .execute()
        )
        if not gold_set_result.data:
            raise HTTPException(status_code=404, detail="Gold set not found")
        gold_set = gold_set_result.data
        items = (
            supabase.table("editorial_gold_set_items")
            .select("label,source_alignment_verified,terminology_verified,speaker_verified")
            .eq("gold_set_id", req.gold_set_id)
            .execute()
        ).data or []
        positives = sum(item.get("label") == "positive" for item in items)
        negatives = sum(item.get("label") == "negative" for item in items)
        if positives < gold_set["target_positive_count"] or negatives < gold_set["target_negative_count"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Gold set needs {gold_set['target_positive_count']} positive and "
                    f"{gold_set['target_negative_count']} negative examples; it has "
                    f"{positives} positive and {negatives} negative."
                ),
            )
        if any(not all([
            item.get("source_alignment_verified"),
            item.get("terminology_verified"),
            item.get("speaker_verified"),
        ]) for item in items):
            raise HTTPException(status_code=409, detail="Every gold example needs source, terminology, and speaker verification")
        updated = (
            supabase.table("editorial_gold_sets")
            .update({
                "status": "locked",
                "locked_by": admin["id"],
                "locked_at": utcnow_iso(),
                "updated_at": utcnow_iso(),
            })
            .eq("id", req.gold_set_id)
            .execute()
        )
        return {"success": True, "gold_set": updated.data[0]}

    @web_app.post("/historical-mappings/backfill")
    async def historical_mapping_backfill_endpoint(
        req: HistoricalBackfillRequest,
        admin=Depends(require_admin),
    ):
        """Queue a bounded private backfill; the worker cannot approve or publish."""
        supabase = service_client()
        inserted = supabase.table("processing_jobs").insert({
            "idempotency_key": f"historical-mapping:{uuid.uuid4()}",
            "job_type": "historical_mapping",
            "source": "admin",
            "requested_by": admin["id"],
            "parameters": {"limit": req.limit, "quote_ids": req.quote_ids},
        }).execute()
        job_id = inserted.data[0]["id"]
        function_call = await backfill_historical_conversation_mappings.spawn.aio(
            limit=req.limit,
            quote_ids=req.quote_ids,
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

    @web_app.post("/historical-mappings/review")
    async def historical_mapping_review_endpoint(
        req: HistoricalMappingReviewRequest,
        admin=Depends(require_admin),
    ):
        allowed_actions = {"edit", "approve", "reject", "needs_revision", "undo"}
        if req.action not in allowed_actions:
            raise HTTPException(status_code=422, detail="Unsupported historical mapping action")
        if req.action in {"approve", "reject", "needs_revision"} and not req.reviewer_expertise:
            raise HTTPException(status_code=422, detail="Reviewer expertise is required")
        if req.action in {"reject", "needs_revision"} and not (
            req.reason_code or (req.reason_detail or "").strip()
        ):
            raise HTTPException(status_code=422, detail="A reason is required")

        supabase = service_client()
        current = (
            supabase.table("conversation_mapping_reviews")
            .select("*")
            .eq("id", req.mapping_review_id)
            .single()
            .execute()
        )
        if not current.data:
            raise HTTPException(status_code=404, detail="Historical mapping review not found")
        before = current.data
        if before.get("workflow_status") == "published":
            raise HTTPException(status_code=409, detail="Published mappings are immutable")

        if req.action == "undo":
            if not req.target_decision_id:
                raise HTTPException(status_code=422, detail="target_decision_id is required")
            decision = (
                supabase.table("conversation_mapping_review_decisions")
                .select("before_state")
                .eq("id", req.target_decision_id)
                .eq("mapping_review_id", req.mapping_review_id)
                .single()
                .execute()
            )
            allowed_restore = {
                "proposed_theme_name", "proposed_theme_summary",
                "proposed_question_text", "proposed_question_summary",
                "proposed_people", "proposed_companies",
                "proposed_relationship_label", "connection_context",
                "workflow_status", "abstention_reason", "reviewed_by", "reviewed_at",
            }
            updates = {
                key: value
                for key, value in ((decision.data or {}).get("before_state") or {}).items()
                if key in allowed_restore
            }
            if not updates:
                raise HTTPException(status_code=422, detail="Decision has no restorable state")
        else:
            editable = {
                "proposed_theme_name": req.proposed_theme_name,
                "proposed_theme_summary": req.proposed_theme_summary,
                "proposed_question_text": req.proposed_question_text,
                "proposed_question_summary": req.proposed_question_summary,
                "proposed_people": req.proposed_people,
                "proposed_companies": req.proposed_companies,
                "proposed_relationship_label": req.proposed_relationship_label,
                "connection_context": req.connection_context,
            }
            updates = {
                key: value.strip() if isinstance(value, str) else value
                for key, value in editable.items()
                if value is not None
            }
            if req.action == "edit" and before.get("workflow_status") == "approved":
                updates.update({
                    "workflow_status": "needs_revision",
                    "reviewed_by": None,
                    "reviewed_at": None,
                })
            elif req.action == "approve":
                proposed = {
                    "theme_name": updates.get("proposed_theme_name", before.get("proposed_theme_name")),
                    "theme_summary": updates.get("proposed_theme_summary", before.get("proposed_theme_summary")),
                    "question_text": updates.get("proposed_question_text", before.get("proposed_question_text")),
                    "question_summary": updates.get("proposed_question_summary", before.get("proposed_question_summary")),
                    "connection_context": updates.get("connection_context", before.get("connection_context")),
                    "related_people": updates.get("proposed_people", before.get("proposed_people")) or [],
                    "related_companies": updates.get("proposed_companies", before.get("proposed_companies")) or [],
                    "mapping_confidence": before.get("mapping_confidence"),
                }
                if not historical_mapping_is_reviewable(
                    proposed,
                    before.get("source_start_segment") if before.get("source_start_segment") is not None else -1,
                    before.get("source_end_segment") if before.get("source_end_segment") is not None else -1,
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="Mapping does not clear the source, completeness, or confidence gate",
                    )
                updates.update({
                    "workflow_status": "approved",
                    "abstention_reason": None,
                    "reviewed_by": admin["id"],
                    "reviewed_at": utcnow_iso(),
                })
            elif req.action == "reject":
                updates.update({
                    "workflow_status": "rejected",
                    "reviewed_by": admin["id"],
                    "reviewed_at": utcnow_iso(),
                })
            elif req.action == "needs_revision":
                updates.update({
                    "workflow_status": "needs_revision",
                    "reviewed_by": admin["id"],
                    "reviewed_at": utcnow_iso(),
                })

        updates["updated_at"] = utcnow_iso()
        updated = (
            supabase.table("conversation_mapping_reviews")
            .update(updates)
            .eq("id", req.mapping_review_id)
            .execute()
        )
        after = updated.data[0] if updated.data else {**before, **updates}
        decision = supabase.table("conversation_mapping_review_decisions").insert({
            "mapping_review_id": req.mapping_review_id,
            "reviewer_id": admin["id"],
            "decision": req.action,
            "reason_code": req.reason_code,
            "reason_detail": req.reason_detail,
            "reviewer_expertise": req.reviewer_expertise,
            "before_state": before,
            "after_state": after,
            "metadata": {"quote_id": before.get("quote_id")},
        }).execute()
        return {
            "success": True,
            "mapping": after,
            "decision_id": decision.data[0]["id"] if decision.data else None,
        }

    @web_app.post("/historical-mappings/publish")
    async def historical_mapping_publish_endpoint(
        req: HistoricalMappingPublishRequest,
        admin=Depends(require_admin),
    ):
        try:
            result = service_client().rpc(
                "publish_historical_conversation_mapping",
                {
                    "p_mapping_review_id": req.mapping_review_id,
                    "p_publisher_id": admin["id"],
                },
            ).execute()
            return {"success": True, "quote_id": result.data}
        except Exception as exc:
            return JSONResponse(status_code=422, content={"success": False, "error": str(exc)})

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
            "reject_context", "approve_mapping", "reject_mapping", "undo",
        }
        if req.action not in allowed_actions:
            raise HTTPException(status_code=422, detail="Unsupported review action")
        if req.action in {
            "approve", "reject", "approve_context", "reject_context",
            "approve_mapping", "reject_mapping",
        } and not req.reviewer_expertise:
            raise HTTPException(status_code=422, detail="Reviewer expertise is required for an editorial decision")

        supabase = service_client()
        staged_result = (
            supabase.table("test_quotes").select("*").eq("id", req.quote_id).single().execute()
        )
        if not staged_result.data:
            raise HTTPException(status_code=404, detail="Quote not found")
        before = staged_result.data
        if req.add_to_gold_set:
            if req.action not in {"approve", "reject"}:
                raise HTTPException(
                    status_code=422,
                    detail="Only an explicit take approval or rejection can become gold evidence",
                )
            if not (req.gold_rationale or "").strip():
                raise HTTPException(status_code=422, detail="Gold-set examples require an editorial rationale")
            if not str(before.get("source_transcript_excerpt") or "").strip():
                raise HTTPException(status_code=422, detail="Gold-set examples require source transcript evidence")
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
                "rejection_reason", "editorial_notes", "proposed_theme_name",
                "proposed_theme_summary", "proposed_question_text",
                "proposed_question_summary", "proposed_people",
                "proposed_companies", "connection_context",
                "theme_match_action",
                "mapping_review_status", "mapping_reviewed_by",
                "mapping_reviewed_at",
            }
            updates = {key: value for key, value in restore.items() if key in allowed_restore}
            if not updates:
                raise HTTPException(status_code=422, detail="Decision has no restorable state")
        else:
            editable_values = {
                "quote_text": req.quote_text,
                "editorial_context": req.editorial_context,
                "proposed_theme_name": req.proposed_theme_name,
                "proposed_theme_summary": req.proposed_theme_summary,
                "proposed_question_text": req.proposed_question_text,
                "proposed_question_summary": req.proposed_question_summary,
                "proposed_people": req.proposed_people,
                "proposed_companies": req.proposed_companies,
                "connection_context": req.connection_context,
                "theme_match_action": req.theme_match_action,
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
            if req.theme_match_action and req.theme_match_action not in {
                "existing_theme", "propose_new", "abstain",
            }:
                raise HTTPException(status_code=422, detail="Unsupported theme match action")
            if req.action == "edit":
                updates.update(editorial_gate_invalidations(before, updates))
            if req.action == "approve":
                take_record = {**before, **updates}
                missing = missing_take_verification_fields(take_record)
                if missing:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Verify {', '.join(missing)} before approving this take",
                    )
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
            elif req.action == "approve_mapping":
                mapping_values = {
                    "theme": updates.get("proposed_theme_name", before.get("proposed_theme_name")),
                    "theme_summary": updates.get("proposed_theme_summary", before.get("proposed_theme_summary")),
                    "question": updates.get("proposed_question_text", before.get("proposed_question_text")),
                    "question_summary": updates.get("proposed_question_summary", before.get("proposed_question_summary")),
                    "connection_context": updates.get("connection_context", before.get("connection_context")),
                }
                active_registry = (
                    supabase.table("adtech_theme_registry")
                    .select("canonical_name,definition")
                    .eq("status", "active")
                    .execute()
                ).data or []
                selected_theme = next((
                    theme for theme in active_registry
                    if str(theme.get("canonical_name") or "").strip().casefold()
                    == str(mapping_values["theme"] or "").strip().casefold()
                ), None)
                if not selected_theme:
                    raise HTTPException(
                        status_code=422,
                        detail="Select an active controlled theme before approving the mapping",
                    )
                canonical_theme = str(selected_theme.get("canonical_name") or "").strip()
                mapping_values["theme"] = canonical_theme
                mapping_values["theme_summary"] = (
                    str(mapping_values["theme_summary"] or "").strip()
                    or str(selected_theme.get("definition") or "").strip()
                )
                updates.update({
                    "proposed_theme_name": mapping_values["theme"],
                    "proposed_theme_summary": mapping_values["theme_summary"],
                    "theme_match_action": "existing_theme",
                })
                if any(not str(value or "").strip() for value in mapping_values.values()):
                    raise HTTPException(
                        status_code=422,
                        detail="Theme, question, question summary, and connection context are required",
                    )
                if len(str(mapping_values["connection_context"]).split()) < 20:
                    raise HTTPException(status_code=422, detail="Connection context is too thin for SME approval")
                reviewable_mapping = {
                    "theme_name": mapping_values["theme"],
                    "theme_summary": mapping_values["theme_summary"],
                    "question_text": mapping_values["question"],
                    "question_summary": mapping_values["question_summary"],
                    "connection_context": mapping_values["connection_context"],
                    "related_people": updates.get("proposed_people", before.get("proposed_people")) or [],
                    "related_companies": updates.get("proposed_companies", before.get("proposed_companies")) or [],
                }
                if not conversation_mapping_is_reviewable(
                    reviewable_mapping,
                    before.get("source_start_segment") if before.get("source_start_segment") is not None else -1,
                    before.get("source_end_segment") if before.get("source_end_segment") is not None else -1,
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="Connections require complete labels and reviewable evidence inside the recorded source boundary",
                    )
                updates.update({
                    "mapping_review_status": "approved",
                    "mapping_reviewed_by": admin["id"],
                    "mapping_reviewed_at": utcnow_iso(),
                })
            elif req.action == "reject_mapping":
                if not req.reason_code and not req.reason_detail:
                    raise HTTPException(status_code=422, detail="A mapping rejection reason is required")
                updates["mapping_review_status"] = "rejected"

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
                    "rejection_reason", "editorial_notes", "proposed_theme_name",
                    "proposed_theme_summary", "proposed_question_text",
                    "proposed_question_summary", "proposed_people",
                    "proposed_companies", "connection_context",
                    "theme_match_action",
                    "mapping_review_status", "mapping_reviewed_by",
                    "mapping_reviewed_at",
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
            "prompt_version": (
                before.get("mapping_prompt_version")
                if "mapping" in req.action
                else before.get("context_prompt_version")
            ),
            "metadata": audit_metadata,
        }).execute()
        gold_item = None
        gold_warning = None
        if req.add_to_gold_set:
            if req.action not in {"approve", "reject"}:
                raise HTTPException(
                    status_code=422,
                    detail="Only an explicit take approval or rejection can become gold evidence",
                )
            if not (req.gold_rationale or "").strip():
                raise HTTPException(status_code=422, detail="Gold-set examples require an editorial rationale")
            if not str(after.get("source_transcript_excerpt") or "").strip():
                raise HTTPException(status_code=422, detail="Gold-set examples require source transcript evidence")
            try:
                gold_item = add_gold_set_item(
                    supabase,
                    created_by=admin["id"],
                    label="positive" if req.action == "approve" else "negative",
                    preferred_quote_text=after.get("quote_text"),
                    source_transcript_excerpt=after.get("source_transcript_excerpt"),
                    rationale=req.gold_rationale.strip(),
                    reviewer_expertise=req.reviewer_expertise,
                    failure_codes=req.gold_failure_codes,
                    test_quote_id=req.quote_id,
                )
            except Exception as exc:
                if "duplicate key" not in str(exc).lower() and "23505" not in str(exc):
                    gold_warning = str(exc)
        return {
            "success": True,
            "quote": after,
            "decision_id": decision.data[0]["id"] if decision.data else None,
            "gold_item": gold_item,
            "gold_warning": gold_warning,
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


@app.function(image=image, secrets=[my_secret], timeout=3700)
def trigger_historical_backfill(backfill_limit: int = 12):
    """Create an auditable operator job for a bounded historical mapping run."""
    from supabase import create_client

    bounded_limit = max(1, min(backfill_limit, 50))
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    job = supabase.table("processing_jobs").insert({
        "idempotency_key": f"operator-historical-mapping:{uuid.uuid4()}",
        "job_type": "historical_mapping",
        "source": "admin",
        "parameters": {
            "limit": bounded_limit,
            "operator_surface": "modal_cli",
        },
    }).execute()
    job_id = job.data[0]["id"]
    result = backfill_historical_conversation_mappings.remote(
        limit=bounded_limit,
        job_id=job_id,
    )
    return {"job_id": job_id, **result}


@app.local_entrypoint()
def main(
    action: str = "health",
    max_episodes: int = 1,
    days_back: int = 7,
    backfill_limit: int = 12,
    youtube_id: str = "V1M1mDyuJKM",
):
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
    elif action == "historical-backfill":
        result = trigger_historical_backfill.remote(backfill_limit=backfill_limit)
    elif action == "caption-check":
        result = caption_source_check.remote(youtube_id=youtube_id)
    else:
        raise ValueError(
            "action must be health, process, scheduled-check, historical-backfill, or caption-check"
        )

    print(json.dumps(result, indent=2, default=str))
