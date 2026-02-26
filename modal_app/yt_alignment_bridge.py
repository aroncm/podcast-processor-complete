import os
import json
import re
import time
import requests
from pathlib import Path
from difflib import SequenceMatcher
from supabase import create_client
import yt_dlp

# ── CONFIGURATION ────────────────────────────────────────────────────────────

# Load env vars from .env in parent directory
env_path = Path(__file__).parent.parent / ".env"
env_vars = {}
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                env_vars[key] = value

SUPABASE_URL = env_vars.get('SUPABASE_URL')
SUPABASE_KEY = env_vars.get('SUPABASE_SERVICE_ROLE_KEY') or env_vars.get('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials in .env")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Local cache for captions to minimize YT requests
CACHE_DIR = Path(__file__).parent / "yt_captions_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── LOGIC ───────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Rigorous text normalization for caption matching."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_yt_captions_local(youtube_id: str) -> list | None:
    """Fetch captions using local browser cookies to bypass bot gates."""
    cache_path = CACHE_DIR / f"{youtube_id}.json"
    
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
            
    print(f"  🎬 Fetching YouTube captions for {youtube_id} (using Chrome cookies)...")
    ydl_opts = {
        'skip_download': True,
        'writeautosubs': True,
        'subtitleslangs': ['en.*'],
        'quiet': True,
        'no_warnings': True,
        'cookiesfrombrowser': ('chrome',), # Use authenticated Chrome session
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_id, download=False)
            sub_url = None
            
            # Manual subs
            if 'subtitles' in info and 'en' in info['subtitles']:
                for fmt in info['subtitles']['en']:
                    if fmt.get('ext') == 'json3':
                        sub_url = fmt['url']
                        break
            
            # Auto subs
            if not sub_url and 'automatic_captions' in info:
                en_keys = [k for k in info['automatic_captions'].keys() if k.startswith('en')]
                if en_keys:
                    for fmt in info['automatic_captions'][en_keys[0]]:
                        if fmt.get('ext') == 'json3':
                            sub_url = fmt['url']
                            break
            
            if not sub_url:
                print(f"  ⚠️ No English json3 captions found for {youtube_id}")
                return None
            
            resp = requests.get(sub_url)
            if resp.status_code != 200:
                print(f"  ⚠️ Failed to download captions from {sub_url[:50]}...")
                return None
                
            data = resp.json()
            processed = []
            for event in data.get('events', []):
                if 'segs' not in event: continue
                start_ms = event.get('tStartMs', 0)
                duration_ms = event.get('dDurationMs', 1)
                text = "".join([s.get('utf8', '') for s in event['segs']]).strip()
                if not text: continue
                
                processed.append({
                    'start': start_ms / 1000.0,
                    'end': (start_ms + duration_ms) / 1000.0,
                    'raw_text': text,
                    'norm_text': normalize_text(text),
                    'word_count': len(text.split())
                })
            
            # Save to cache
            with open(cache_path, 'w') as f:
                json.dump(processed, f)
                
            return processed
            
    except Exception as e:
        print(f"  ❌ Error fetching captions for {youtube_id}: {e}")
        return None

def align_quote(quote_text, youtube_id, whisper_start, whisper_end):
    """Port of the deterministic alignment engine."""
    captions = get_yt_captions_local(youtube_id)
    if not captions:
        return None

    norm_quote = normalize_text(quote_text)
    quote_words = norm_quote.split()
    quote_word_count = len(quote_words)
    if quote_word_count < 5:
        return None

    best_candidate = None
    best_composite_score = 0
    second_best_composite = 0
    
    min_win = max(1, int(quote_word_count * 0.55))
    max_win = int(quote_word_count * 2.2)
    
    # Search around whisper timestamp
    SEARCH_RADIUS = 300 
    start_time_limit = max(0, whisper_start - SEARCH_RADIUS)
    end_time_limit = whisper_end + SEARCH_RADIUS
    
    start_idx = 0
    end_idx = len(captions)
    for i, c in enumerate(captions):
        if c['start'] < start_time_limit: start_idx = i
        if c['start'] > end_time_limit: 
            end_idx = i
            break
            
    for i in range(start_idx, end_idx):
        for win_size in range(1, 15):
            if i + win_size > len(captions): break
            window_events = captions[i : i+win_size]
            window_text = " ".join([e['norm_text'] for e in window_events])
            window_words = window_text.split()
            
            if len(window_words) < min_win: continue
            if len(window_words) > max_win: break
            
            sm_ratio = SequenceMatcher(None, norm_quote, window_text).ratio()
            
            needle_set = set(quote_words)
            candidate_set = set(window_words)
            common = needle_set & candidate_set
            
            f1, recall = 0, 0
            if common:
                prec = len(common) / len(candidate_set)
                rec = len(common) / len(needle_set)
                f1 = 2 * (prec * rec) / (prec + rec)
                recall = rec
            
            composite = (0.6 * sm_ratio) + (0.25 * f1) + (0.15 * recall)
            if composite > best_composite_score:
                if best_candidate and not (i + win_size <= best_candidate['start_idx'] or i >= best_candidate['end_idx']):
                    pass # overlapping
                else:
                    second_best_composite = best_composite_score
                
                best_composite_score = composite
                best_candidate = {
                    'start_idx': i, 'end_idx': i + win_size,
                    'start_time': window_events[0]['start'],
                    'end_time': window_events[-1]['end']
                }
            elif composite > second_best_composite:
                if best_candidate and not (i + win_size <= best_candidate['start_idx'] or i >= best_candidate['end_idx']):
                    pass
                else:
                    second_best_composite = composite

    if not best_candidate: return None
        
    MIN_COMPOSITE = 0.75 if quote_word_count < 30 else 0.70
    MIN_MARGIN = 0.04 if quote_word_count < 30 else 0.03
    
    if best_composite_score < MIN_COMPOSITE or (best_composite_score - second_best_composite) < MIN_MARGIN:
        return None

    # Sentence logic + padding
    exact_start, exact_end = best_candidate['start_time'], best_candidate['end_time']
    f_start, f_end = max(0, int(exact_start - 30)), int(exact_end + 30)
    
    # Sentence start
    for i in range(len(captions)-1, -1, -1):
        c = captions[i]
        if c['start'] <= f_start:
            is_boundary = False
            if i > 0:
                prev = captions[i-1]
                if any(p in prev['raw_text'] for p in ['.', '!', '?']): is_boundary = True
                elif c['start'] - prev['end'] > 1.5: is_boundary = True
            if is_boundary:
                f_start = int(c['start'])
                break
        if c['start'] < f_start - 15: break

    # Sentence end
    for i in range(len(captions)):
        c = captions[i]
        if c['end'] >= f_end:
            if any(p in c['raw_text'] for p in ['.', '!', '?']):
                f_end = int(c['end'])
                break
            if i + 1 < len(captions):
                if captions[i+1]['start'] - c['end'] > 1.5:
                    f_end = int(c['end'])
                    break
        if c['end'] > f_end + 15: break
            
    return {'start': f_start, 'end': f_end, 'confidence': round(best_composite_score, 3)}

# ── MAIN LOOP ───────────────────────────────────────────────────────────────

def run_bridge():
    print("🌉 Starting YouTube Alignment Bridge...")
    
    # 1. Fetch quotes needing alignment
    res = supabase.table('test_quotes') \
        .select('id, quote_text, youtube_id, timestamp_start, timestamp_end') \
        .is_('yt_timestamp_confidence', 'null') \
        .not_.is_('youtube_id', 'null') \
        .limit(20) \
        .execute()
        
    quotes = res.data or []
    if not quotes:
        print("✅ No quotes pending alignment.")
        return

    print(f"🔍 Found {len(quotes)} quotes pending alignment.")
    
    for q in quotes:
        print(f"\nProcessing ID {q['id']}: '{q['quote_text'][:40]}...'")
        result = align_quote(q['quote_text'], q['youtube_id'], q['timestamp_start'], q['timestamp_end'])
        
        if result:
            print(f"  🎯 Match Found: {result['start']}s - {result['end']}s (conf={result['confidence']})")
            # Update Supabase
            supabase.table('test_quotes').update({
                'timestamp_start': result['start'],
                'timestamp_end': result['end'],
                'yt_timestamp_confidence': result['confidence']
            }).eq('id', q['id']).execute()
            print("  ✅ Updated Supabase.")
        else:
            print("  ⚠️ No confident match found. Setting confidence to 0.0 to mark as attempted.")
            supabase.table('test_quotes').update({
                'yt_timestamp_confidence': 0.0
            }).eq('id', q['id']).execute()

if __name__ == "__main__":
    while True:
        try:
            run_bridge()
            print("\nSleeping for 60s...")
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nBridge stopped.")
            break
        except Exception as e:
            print(f"❌ Unhandled error in bridge loop: {e}")
            time.sleep(10)
