import modal
import os
import requests

secrets_list = [
    modal.Secret.from_name("my-processor-secret"),
    modal.Secret.from_name("custom-secret"),
    modal.Secret.from_name("custom-secret-2")
]

image = modal.Image.debian_slim().pip_install("supabase", "requests")
app = modal.App("prod-backfill-join")

@app.function(image=image, secrets=secrets_list, timeout=3600)
def execute_cleanup():
    from supabase import create_client
    
    yt_key = os.environ.get('YOUTUBE_API_KEY')
    supabase = create_client(os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_KEY'))
    
    # We join the podcasts table to get the name via the podcast_id
    res = supabase.table('quotes').select(
        'id, episode_name, podcasts(name)'
    ).is_('youtube_id', 'null').eq('approval_status', 'pending').execute()
    
    quotes = res.data or []
    print(f"🚀 Found {len(quotes)} production quotes needing IDs.")

    processed_queries = {} 
    for q in quotes:
        # Extract the joined podcast name
        podcast_name = q.get('podcasts', {}).get('name', 'Unknown Podcast')
        episode_name = q.get('episode_name', 'Unknown Episode')
        
        # Build the tiered search query
        raw_name = f"{podcast_name} {episode_name}"
        search_query = raw_name.replace("Episode ", "").split(" with ")[0].split(" ft. ")[0].split(":")[0].strip()
        
        if search_query not in processed_queries:
            print(f"🔍 Searching YouTube: '{search_query}'")
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {'part': 'id', 'q': search_query, 'key': yt_key, 'maxResults': 1, 'type': 'video'}
            
            try:
                r = requests.get(url, params=params)
                items = r.json().get('items', [])
                if items:
                    yt_id = items[0]['id']['videoId']
                    processed_queries[search_query] = yt_id
                    print(f"   ✨ Found ID: {yt_id}")
                else:
                    processed_queries[search_query] = None
            except:
                processed_queries[search_query] = None

        if processed_queries.get(search_query):
            supabase.table('quotes').update({'youtube_id': processed_queries[search_query]}).eq('id', q['id']).execute()
            print(f"   ✅ Saved {q['id']}")

@app.local_entrypoint()
def main():
    execute_cleanup.remote()
