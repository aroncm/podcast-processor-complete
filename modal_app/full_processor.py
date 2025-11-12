"""Full processor with real transcription and GPT quote extraction"""

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
        "feedparser",
        "pydub"
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
def process_episode_with_ai():
    """Process full episode with intelligent extraction"""
    
    import feedparser
    import subprocess
    import tempfile
    from supabase import create_client
    from openai import OpenAI
    
    print("🚀 Starting AI-powered processing...")
    
    # Initialize clients
    supabase = create_client(
        os.environ['SUPABASE_URL'],
        os.environ['SUPABASE_KEY']
    )
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    # Get feed
    feeds = supabase.table('test_podcast_feeds').select('*').limit(1).execute()
    if not feeds.data:
        return {"error": "No test feeds found"}
    
    feed = feeds.data[0]
    parsed = feedparser.parse(feed['rss_url'])
    
    # Find unprocessed episode
    episode = None
    for entry in parsed.entries[:10]:
        existing = supabase.table('test_quotes') \
            .select('id') \
            .like('episode_name', f"{entry.title[:50]}%") \
            .execute()
        
        if not existing.data:
            episode = entry
            print(f"📎 Found new episode: {entry.title}")
            break
        else:
            print(f"⏭️  Skipping: {entry.title[:50]}")
    
    if not episode:
        return {"message": "All recent episodes processed"}
    
    # Get audio URL
    audio_url = episode.enclosures[0].get('href') if episode.enclosures else None
    if not audio_url:
        return {"error": "No audio URL found"}
    
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
    
    # Extract smart quotes
    print("🧠 Extracting intelligent quotes...")
    
    text = transcript.text
    max_chunk = 12000
    all_quotes = []
    
    if len(text) > max_chunk:
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        print(f"Processing {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:3]):
            quotes = extract_quotes(
                chunk, 
                feed['name'], 
                episode.title, 
                client,
                chunk_num=i+1
            )
            all_quotes.extend(quotes)
    else:
        all_quotes = extract_quotes(
            text, 
            feed['name'], 
            episode.title, 
            client
        )
    
    print(f"💎 Extracted {len(all_quotes)} quotes")
    
    # Find natural boundaries for clips
    if hasattr(transcript, 'segments'):
        for quote in all_quotes:
            boundaries = find_clip_boundaries_fixed(quote, transcript.segments)
            quote.update(boundaries)
    
    # Save to database
    saved = []
    for i, quote in enumerate(all_quotes[:15]):
        record = {
            'podcast_name': feed['name'],
            'episode_name': episode.title[:100],
            'speaker_name': quote.get('speaker', 'Unknown'),
            'category': quote.get('category', 'Other'),
            'quote_text': quote['text'],
            'date_published': datetime.now().isoformat(),
            'audio_clip_url': audio_url,
            'timestamp_start': int(quote.get('clip_start', i * 60)),
            'timestamp_end': int(quote.get('clip_end', (i + 1) * 60)),
            'approval_status': 'pending',
            'test_run': True
        }
        
        result = supabase.table('test_quotes').insert(record).execute()
        if result.data:
            saved.append(quote['text'][:80])
    
    os.remove(temp_path)
    
    return {
        "success": True,
        "episode": episode.title,
        "duration_minutes": round(duration_minutes, 1),
        "quotes_extracted": len(saved),
        "sample_quotes": saved[:3],
        "processing_cost": round(duration_minutes * 0.006, 2)
    }

def extract_quotes(text, podcast, episode, client, chunk_num=0):
    """Extract high-quality quotes"""
    
    chunk_info = f"(Section {chunk_num})" if chunk_num > 0 else ""
    
    prompt = f"""
    Extract 5-8 exceptional podcast quotes {chunk_info}.
    
    Podcast: {podcast}
    Episode: {episode}
    
    Transcript:
    {text}
    
    Find quotes that are:
    - Self-contained insights (20-60 words ideal)
    - Genuinely interesting or surprising  
    - Would make someone want to listen to the full episode
    
    Return JSON:
    {{"quotes": [
        {{
            "text": "The exact quote",
            "speaker": "Name or Host/Guest",
            "category": "Business|Technology|Life|Science|Culture|Politics|Other"
        }}
    ]}}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "Extract only exceptional quotes."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    
    data = json.loads(response.choices[0].message.content)
    return data.get('quotes', [])

def find_clip_boundaries_fixed(quote, segments):
    """Find natural sentence boundaries for clips"""
    
    quote_lower = quote['text'][:30].lower()
    
    match_idx = None
    for i, seg in enumerate(segments):
        seg_text = seg.text if hasattr(seg, 'text') else str(seg)
        if quote_lower in seg_text.lower():
            match_idx = i
            break
    
    if match_idx is None:
        return {}
    
    start_idx = max(0, match_idx - 1)
    end_idx = min(len(segments) - 1, match_idx + 1)
    
    start_time = segments[start_idx].start if hasattr(segments[start_idx], 'start') else 0
    end_time = segments[end_idx].end if hasattr(segments[end_idx], 'end') else 60
    
    start_time = max(0, start_time - 0.5)
    end_time = end_time + 0.5
    
    duration = end_time - start_time
    if duration < 20:
        end_time = start_time + 30
    elif duration > 90:
        end_time = start_time + 90
    
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
    result = process_episode_with_ai()
    print(f"Scheduled run result: {result}")
    return result
