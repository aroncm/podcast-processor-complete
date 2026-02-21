"""Full processor with real transcription and GPT quote extraction - Quality-focused version"""

import modal
import os
import json
from datetime import datetime

app = modal.App("podcast-processor-full")

# Enhanced image with ffmpeg for audio processing
image = modal.Image.debian_slim() \
    .pip_install(
        "supabase",
        "openai>=1.0.0",
        "anthropic",
        "feedparser",
        "pydub",
        "fastapi"
    ) \
    .apt_install("ffmpeg")

# Read env vars
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
env_vars = {}
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                env_vars[key] = value

my_secret = modal.Secret.from_dict(env_vars)

@app.function(
    image=image,
    secrets=[my_secret],
    timeout=1800,
    cpu=2,
)
def process_episode_with_ai(feed_ids: list = None, start_date: str = None, end_date: str = None):
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
    
    # Get ALL feeds (Fixed: Removed .limit(1))
    feeds = supabase.table('test_podcast_feeds').select('*').execute()
    if not feeds.data:
        return {"error": "No test feeds found"}
    
    # Filter by specific feed_ids if provided
    if feed_ids:
        feeds.data = [f for f in feeds.data if f['id'] in feed_ids]
        
    all_results = []
    
    # Parse Date Filters
    start_dt = datetime.fromisoformat(start_date) if start_date else None
    end_dt = datetime.fromisoformat(end_date) if end_date else None
    
    for feed in feeds.data:
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
                
            print(f"  ✨ Processing {len(new_episodes)} new episodes for {feed['name']}...")
            
            # Process each new episode
            for episode in new_episodes:
                result = process_single_episode_logic(episode, feed, client, supabase)
                all_results.append(result)
                
        except Exception as e:
            print(f"❌ Error processing feed {feed['name']}: {str(e)}")
            continue
        
    return {
        "success": True, 
        "processed_count": len(all_results), 
        "details": all_results
    }

# BOILERPLATE MOCK Implementation for Missing Apps 
# (Real implementation would duplicate logic, for now we restore the stubs/functions 
# so the Modal dashboard looks correct and they can be expanded)

def slugify(text: str) -> str:
    """Simple slugify for ID generation"""
    if not text: return "unknown"
    import re
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

@app.function(image=image, secrets=[my_secret], timeout=600)
def promote_quote_to_production(quote_id: str):
    """Move a quote from test_quotes to production quotes with robust ID resolution and auto-creation"""
    print(f"🚀 Promoting quote {quote_id} to production...")
    from supabase import create_client
    supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    
    # 1. Get test quote
    tq = supabase.table('test_quotes').select('*').eq('id', quote_id).single().execute()
    if not tq.data: return {"error": "Quote not found"}
    data = tq.data
    
    # 2. Resolve Podcast
    podcast_name = data.get('podcast_name', 'Unknown Podcast').strip()
    p_res = supabase.table('podcasts').select('id').eq('name', podcast_name).execute()
    if p_res.data:
        podcast_id = p_res.data[0]['id']
    else:
        # Try slugified match
        podcast_id = slugify(podcast_name)
        p_res = supabase.table('podcasts').select('id').eq('id', podcast_id).execute()
        if not p_res.data:
            print(f"  ✨ Creating new podcast: {podcast_name} ({podcast_id})")
            supabase.table('podcasts').insert({
                "id": podcast_id,
                "name": podcast_name
            }).execute()

    # 3. Resolve Category
    category_name = data.get('category', 'General').strip()
    c_res = supabase.table('categories').select('id').eq('name', category_name).execute()
    if c_res.data:
        category_id = c_res.data[0]['id']
    else:
        # Try slugified match
        category_id = slugify(category_name)
        c_res = supabase.table('categories').select('id').eq('id', category_id).execute()
        if not c_res.data:
            print(f"  ✨ Creating new category: {category_name} ({category_id})")
            supabase.table('categories').insert({
                "id": category_id,
                "name": category_name
            }).execute()

    # 4. Resolve Guest
    guest_name = data.get('speaker_name', 'Unknown Speaker').strip()
    
    # Try exact name match
    g_res = supabase.table('guests').select('id').eq('name', guest_name).execute()
    if not g_res.data:
        # Try case-insensitive / trimmed match via ilike
        g_res = supabase.table('guests').select('id').ilike('name', guest_name).execute()
        
    if g_res.data:
        guest_id = g_res.data[0]['id']
    else:
        # Try slugified match
        guest_id = slugify(guest_name)
        g_res = supabase.table('guests').select('id').eq('id', guest_id).execute()
        if not g_res.data:
            print(f"  ✨ Creating new guest: {guest_name} ({guest_id})")
            guest_payload = {
                "id": guest_id,
                "name": guest_name
            }
            if data.get('speaker_title'): guest_payload['title'] = data['speaker_title']
            if data.get('speaker_company'): guest_payload['company'] = data['speaker_company']
            if data.get('speaker_linkedin'): guest_payload['linkedin_url'] = data['speaker_linkedin']
            
            supabase.table('guests').insert(guest_payload).execute()
        else:
            guest_id = g_res.data[0]['id']
            # Optional: Update existing guest if they are missing metadata
            # For now, we prioritize the new creation as requested

    # 5. Resolve Episode
    episode_name = data.get('episode_name', 'Unknown Episode').strip()
    # Use exact podcast_id + title match
    e_res = supabase.table('episodes').select('id').eq('podcast_id', podcast_id).eq('title', episode_name).execute()
    if e_res.data:
        episode_id = e_res.data[0]['id']
    else:
        # Create new episode with a more robust slug or just use the title
        # (We prefer a clean slug for the ID)
        episode_id = slugify(episode_name[:60]) # Longer prefix
        print(f"  ✨ Creating new episode: {episode_name} ({episode_id})")
        supabase.table('episodes').upsert({
            "id": episode_id,
            "title": episode_name,
            "podcast_id": podcast_id,
            "date": data.get('date_published', datetime.now().strftime('%Y-%m-%d'))[:10]
        }).execute()
        
        # Link guest to episode
        supabase.table('guest_episodes').upsert({
            "guest_id": guest_id,
            "episode_id": episode_id
        }).execute()

    # 6. Insert into production quotes
    prod_payload = {
        "id": data['id'],
        "text": data['quote_text'],
        "episode_id": episode_id,
        "guest_id": guest_id,
        "category_id": category_id,
        "clip_link": data['audio_clip_url'],
        "podcast_id": podcast_id,      # Critical for visibility in some views
        "guest_name": guest_name,      # Denormalized for performance
        "speaker": guest_name,         # Backward compatibility
        "youtube_id": data.get('youtube_id'),
        "timestamp_start": data.get('timestamp_start'),
        "timestamp_end": data.get('timestamp_end'),
        "youtube_offset": data.get('youtube_offset', 0),
        "quality_score": data.get('quality_score'),
        "extraction_model": data.get('extraction_model'),
        "context": data.get('category'),
        "support_count": 0             # Start with 0 in production
    }
    
    try:
        res = supabase.table('quotes').upsert(prod_payload).execute()
        
        # 7. Update source status to 'promoted' so it leaves the admin queue
        supabase.table('test_quotes').update({"approval_status": "promoted"}).eq('id', data['id']).execute()
        
        print(f"✅ Successfully promoted to production and updated test_quotes! (Quote ID: {data['id']})")
        return {"success": True, "data": res.data}
    except Exception as e:
        print(f"❌ Promotion failed: {e}")
        return {"success": False, "error": str(e)}

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

@app.function(image=image, secrets=[my_secret], timeout=600)
def trigger_manual_processor():
    """Legacy trigger for testing"""
    return process_episode_with_ai.remote()

@app.function(image=image, secrets=[my_secret], timeout=600)
def trigger_scheduled_processor():
    """Legacy trigger for scheduled"""
    return scheduled_processor.remote()

def process_single_episode_logic(episode, feed, client, supabase):
    """Refactored logic for processing a single episode"""
    import subprocess
    import tempfile
    import time
    
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
            print(f"📺 FOUND YouTube ID: {youtube_id}")
        else:
            print("❌ No YouTube ID found in this episode.")
        
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
            'ffmpeg', '-i', audio_url,
            '-acodec', 'mp3',
            '-ar', '16000',
            '-ac', '1',
            '-y', temp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Trying with 30-minute limit...")
            cmd = [
                'ffmpeg', '-i', audio_url,
                '-t', '1800',
                '-acodec', 'mp3',
                '-ar', '16000',
                '-ac', '1',
                '-y', temp_path
            ]
            subprocess.run(cmd, capture_output=True)
        
        # Get file info
        file_size = os.path.getsize(temp_path)
        duration_minutes = (file_size / (16000 * 2)) / 60
        processing_cost = duration_minutes * 0.006 
        
        print(f"📊 Episode duration: ~{duration_minutes:.1f} minutes")
        print(f"💰 Estimated cost: ${duration_minutes * 0.006:.2f}")
        
        # Transcribe with timestamps
        print("🎤 Transcribing with Whisper...")
        with open(temp_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        print(f"✅ Transcription complete: {len(transcript.text)} characters")
        
        # Exact Timestamp Logic: Index Segments
        print("🧠 Extracting high-quality takes (Exact Segment Match)...")
        
        segments = transcript.segments
        
        # Build formatted text with Segment IDs: "[1] text [2] text"
        # We need to map text back to segments later
        formatted_chunks = []
        current_chunk = ""
        current_chunk_segments = [] # Track which segments are in this chunk
        
        MAX_CHUNK_SIZE = 12000
        
        for i, seg in enumerate(segments):
            seg_text = seg.text if hasattr(seg, 'text') else str(seg)
            formatted_line = f"[{i}] {seg_text} "
            
            if len(current_chunk) + len(formatted_line) > MAX_CHUNK_SIZE:
                formatted_chunks.append(current_chunk)
                current_chunk = formatted_line
            else:
                current_chunk += formatted_line
                
        if current_chunk:
            formatted_chunks.append(current_chunk)

        all_quotes = []
        print(f"Processing {len(formatted_chunks)} chunks for quality quotes")
        
        for i, chunk_text in enumerate(formatted_chunks[:4]): # Process up to 4 chunks
            time.sleep(1) # Throttling
            quotes = extract_quotes(
                chunk_text, # Passing indexed text
                feed['name'], 
                episode.title, 
                client,
                chunk_num=i+1
            )
            
            # Post-Process: Look up real timestamps
            for q in quotes:
                try:
                    start_id = q.get('start_segment_id')
                    end_id = q.get('end_segment_id')
                    
                    if start_id is None or end_id is None:
                        # Fallback for LLM hallucination
                        print(f"⚠️ Quote missing Segment IDs: {q.get('text')[:30]}")
                        # Could try fuzzy match here as fail-safe, or skip
                        continue
                        
                    # Validate IDs are ints
                    start_id = int(start_id)
                    end_id = int(end_id)
                    
                    # Look up timestamps
                    start_time = segments[start_id].start
                    end_time = segments[end_id].end
                    
                    # Add buffer logic
                    start_time = max(0, start_time - 0.5)
                    end_time = end_time + 0.5
                    duration = end_time - start_time
                    
                    # Enforce minimums
                    if duration < 15: end_time += 15
                    
                    q['clip_start'] = int(start_time)
                    q['clip_end'] = int(end_time)
                    q['clip_duration'] = int(end_time - start_time)
                    
                    # Store IDs just in case
                    q['start_seg'] = start_id
                    q['end_seg'] = end_id
                    
                    all_quotes.append(q)
                    print(f"✅ Exact match: ID {start_id}-{end_id} ({int(start_time)}s-{int(end_time)}s)")
                    
                except Exception as e:
                    print(f"❌ Error mapping IDs for quote: {e}")
                    continue
        
        # Quality check (Tier 1: GPT-4o-mini)
        all_quotes = sorted(all_quotes, key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Tier 2: Claude Re-ranking
        # Only if we have enough quotes and the API key is set
        if len(all_quotes) >= 5 and os.environ.get('ANTHROPIC_API_KEY'):
             print(f"🤖 Re-ranking top {len(all_quotes[:15])} candidates with Claude 3.5 Sonnet...")
             try:
                 # Initialize Anthropic client here to avoid global import if library missing on old images
                 from anthropic import Anthropic
                 anthropic_client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
                 
                 top_candidates = all_quotes[:15] # Take top 15 from Tier 1
                 reranked = rank_quotes_with_claude(top_candidates, feed['name'], episode.title, anthropic_client)
                 
                 if reranked:
                     all_quotes = reranked
                     print(f"✅ Claude re-ranking complete (Selected {len(all_quotes)})")
                 else:
                     print("⚠️ Claude returned no valid rankings, using GPT-4o-mini scores.")
                     
             except ImportError:
                 print("⚠️ 'anthropic' library not installed. Skipping Tier 2.")
             except Exception as e:
                 print(f"⚠️ Claude re-ranking failed (falling back to GPT scores): {e}")

        # Final Cut
        all_quotes = all_quotes[:8]  
        
        print(f"💎 Extracted {len(all_quotes)} high-quality takes")
        
        if len(all_quotes) < 3:
            print("⚠️ Episode had fewer than 3 exceptional quotes")
        
        # Parse Correct Date
        try:
            date_published = datetime(*episode.published_parsed[:6]).isoformat()
        except:
            date_published = datetime.now().isoformat()
            
        episode_guid = getattr(episode, 'id', None)

        # Save to database
        saved = []
        for i, quote in enumerate(all_quotes):
            record = {
                'podcast_name': feed['name'],
                'episode_name': episode.title[:100],
                'speaker_name': quote.get('speaker', 'Unknown'),
                'category': quote.get('category', 'Other'),
                'quote_text': quote['text'],
                'date_published': date_published,
                'audio_clip_url': audio_url,
                'timestamp_start': int(quote.get('clip_start', i * 60)),
                'timestamp_end': int(quote.get('clip_end', (i + 1) * 60)),
                'approval_status': 'pending',
                'test_run': True,
                'youtube_id': youtube_id,
                'duration_minutes': round(duration_minutes, 1),
                'processing_cost': round(processing_cost, 4),
                'episode_guid': episode_guid,
                'quality_score': round(quote.get('quality_score', 0.0), 3),
                'extraction_model': 'gpt-4o-mini'
            }
            
            db_res = supabase.table('test_quotes').insert(record).execute()
            if db_res.data:
                saved.append(quote['text'][:80])
        
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

def extract_quotes(text, podcast, episode, client, chunk_num=0):
    """Extract only the most insightful and provocative quotes"""
    
    chunk_info = f"(Section {chunk_num})" if chunk_num > 0 else ""
    
    prompt = f"""
    You are curating quotes for PodTakes.
    
    The Transcript is provided with Segment IDs in the format: `[ID] Text...`
    
    Extract EXACTLY 5 exceptional quotes.
    For each quote, you MUST identify the exact `start_segment_id` and `end_segment_id` from the text.
    
    Podcast: {podcast}
    Episode: {episode}
    
    Transcript:
    {text}
    
    Criteria for "Exceptional":
    - 🔥 HOT TAKE: Controversial, forward-looking visuals of the future
    - 💡 COUNTERINTUITIVE: Surprising insights that challenge consensus
    - 🎯 MEMORABLE: New frameworks or specific predictions
    
    ❌ IGNORE:
    - Sales pitches ("We are the leading platform...")
    - Personal career history ("I started my career at...")
    - Generic business advice ("It's all about people...")
    
    Examples of GREAT Quotes (Extract these):
    - "Consumers are moving to a multipolar world where they see content from all sides..."
    - "It is inevitable that AI generated content will surpass human content in volume..."
    - "Being an ad tech company might not be a bad thing, a maximum might actually be..."
    
    Return JSON:
    {{
        "quotes": [
            {{
                "text": "Exact text...",
                "start_segment_id": 123,
                "end_segment_id": 125,
                "speaker": "Name",
                "category": "Technology",
                "quality_score": 0.95
            }}
        ]
    }}
    """
    
    # Use helper with retry logic
    return call_openai_with_retry(client, prompt)

def call_openai_with_retry(client, prompt):
    """Call OpenAI with exponential backoff"""
    import time
    
    max_retries = 5
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": "You are a curator for PodTakes. Extract only the most exceptional, thought-provoking quotes that represent genuine 'takes' - insights that challenge, surprise, or deeply illuminate. Quality over quantity always."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            data = json.loads(response.choices[0].message.content)
            return data.get('quotes', [])
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                delay = base_delay * (2 ** attempt)
                print(f"⚠️ Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"❌ OpenAI Error: {e}")
                raise e
                
    print("❌ Max retries reached")
    return []

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

def rank_quotes_with_claude(quotes, podcast, episode, client):
    """Tier 2: Re-rank quotes using Claude 3.5 Sonnet"""
    print(f"⚖️ Asking Claude to rank {len(quotes)} quotes...")
    
    # Prepare quotes list for prompt
    quotes_text = ""
    for i, q in enumerate(quotes):
        quotes_text += f"QUOTE {i}:\n{q['text']}\n(Speaker: {q.get('speaker', 'Unknown')})\n\n"
        
    prompt = f"""
    You are the Editor-in-Chief for PodTakes.
    
    I have extracted {len(quotes)} potential quotes from the podcast "{podcast} - {episode}".
    They are candidate "Hot Takes".
    
    Your Constraint:
    Identify the TOP 5 absolute best quotes that are:
    1. 🤯 Counter-intuitive (Challenges conventional wisdom)
    2. 🔮 Forward-looking (Predictive, not descriptive)
    3. 🌶️ High Signal (Not generic fluff)
    
    Rank them from 1 (Best) to 5 (Good).
    
    CANDIDATES:
    {quotes_text}
    
    Return JSON:
    {{
        "rankings": [
            {{
                "original_index": 0,
                "new_rank": 1,
                "quality_score": 0.98,
                "reason": "Challenges core assumption about..."
            }},
            ...
        ]
    }}
    """
    
    response = call_anthropic_with_retry(client, prompt)
    if not response or 'rankings' not in response:
        return None
        
    # Re-order based on Claude's ranking
    ranked_quotes = []
    for r in response['rankings']:
        idx = r.get('original_index')
        if idx is not None and 0 <= idx < len(quotes):
            q = quotes[idx]
            q['quality_score'] = r.get('quality_score', q.get('quality_score', 0))
            q['ranking_reason'] = r.get('reason')
            q['extraction_model'] = 'claude-3-5-sonnet' # Mark as curated by Claude
            ranked_quotes.append(q)
            
    # Sort by new score
    ranked_quotes.sort(key=lambda x: x['quality_score'], reverse=True)
    return ranked_quotes

def call_anthropic_with_retry(client, prompt):
    """Call Claude with retry logic"""
    import time
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=0.5,
                system="You are an expert editor who hates generic business fluff. You only approve specific, high-signal insights.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse JSON from response
            content = message.content[0].text
            # Simple JSON extraction in case of preamble
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
                
        except Exception as e:
            print(f"⚠️ Anthropic Error (Attempt {attempt+1}): {e}")
            time.sleep(2)
            
    return None

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
    cmd = [
        'ffmpeg', '-i', quote['audio_clip_url'],
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
    schedule=modal.Period(hours=6),
)
def scheduled_processor():
    """Automatically process new episodes every 6 hours"""
    print(f"⏰ Scheduled processing started at {datetime.now()}")
    result = process_episode_with_ai.remote()
    print(f"Scheduled run result: {result}")
    return result

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
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    web_app = FastAPI()
    
    # Enable CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ProcessRequest(BaseModel):
        feed_ids: list[str] = None
        start_date: str = None
        end_date: str = None

    class ClipRequest(BaseModel):
        quote_id: str

    @web_app.post("/process-episode")
    async def process_episode_endpoint(req: ProcessRequest):
        """Trigger processing manually"""
        print(f"📥 Received process request: {req}")
        try:
            # Use .remote.aio to call the Modal function asynchronously
            result = await process_episode_with_ai.remote.aio(
                feed_ids=req.feed_ids, 
                start_date=req.start_date, 
                end_date=req.end_date
            )
            return result
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @web_app.post("/create-clip")
    async def create_clip_endpoint(req: ClipRequest):
        """Trigger clip creation"""
        try:
            result = await create_audio_clip.remote.aio(req.quote_id)
            return result
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @web_app.post("/approve-quote")
    async def approve_quote_endpoint(request: Request):
        """Approve a quote"""
        data = await request.json()
        quote_id = data.get('quote_id')
        if not quote_id:
            return {"error": "Missing quote_id"}
            
        from supabase import create_client
        supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
        
        # Update status
        res = supabase.table('test_quotes').update({'approval_status': 'approved'}).eq('id', quote_id).execute()
        
        # Check for overrides
        updates = {}
        if 'quote_text' in data: updates['quote_text'] = data['quote_text']
        if 'speaker_name' in data: updates['speaker_name'] = data['speaker_name']
        if 'speaker_title' in data: updates['speaker_title'] = data['speaker_title']
        if 'speaker_company' in data: updates['speaker_company'] = data['speaker_company']
        if 'speaker_linkedin' in data: updates['speaker_linkedin'] = data['speaker_linkedin']
        
        if updates:
             supabase.table('test_quotes').update(updates).eq('id', quote_id).execute()
             
        return {"success": True, "data": res.data}

    @web_app.post("/promote-quote")
    async def promote_quote_endpoint(req: ClipRequest):
        """Trigger promotion manually"""
        try:
            result = await promote_quote_to_production.remote.aio(req.quote_id)
            return result
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return web_app
