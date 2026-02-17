import modal
import os

# Reuse the app definition from your main file or create a simple one
app = modal.App("yt-backfill")
image = modal.Image.debian_slim().pip_install("supabase", "google-api-python-client")

@app.local_entrypoint()
def main():
    from supabase import create_client
    from googleapiclient.discovery import build

    # Initialize clients using Modal secrets
    supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    youtube = build("youtube", "v3", developerKey=os.environ['YOUTUBE_API_KEY'])

    # Get all pending quotes missing a YouTube ID
    res = supabase.table('test_quotes').select('id, podcast_name, episode_name')\
        .is_('youtube_id', 'null').eq('approval_status', 'pending').execute()
    
    quotes = res.data or []
    print(f"🔄 Found {len(quotes)} quotes to backfill...")

    processed_episodes = {} 

    for q in quotes:
        key = f"{q['podcast_name']} {q['episode_name']}"
        
        if key not in processed_episodes:
            print(f"🔍 Searching YouTube: {key}")
            try:
                request = youtube.search().list(q=key, part="id", maxResults=1, type="video")
                response = request.execute()
                yt_id = response['items'][0]['id']['videoId'] if response.get('items') else "NOT_FOUND"
                processed_episodes[key] = yt_id
            except Exception as e:
                print(f"❌ API Error: {e}")
                processed_episodes[key] = "NOT_FOUND"

        if processed_episodes[key] != "NOT_FOUND":
            supabase.table('test_quotes').update({'youtube_id': processed_episodes[key]}).eq('id', q['id']).execute()
            print(f" ✅ Updated {q['id']} with {processed_episodes[key]}")
