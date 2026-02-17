import modal
import os
import requests

secrets_list = [
    modal.Secret.from_name("my-processor-secret"),
    modal.Secret.from_name("custom-secret"),
    modal.Secret.from_name("custom-secret-2")
]

image = modal.Image.debian_slim().pip_install("supabase", "requests")
app = modal.App("prod-youtube-backfill")

@app.function(image=image, secrets=secrets_list, timeout=3600)
def run_production_cleanup():
    from supabase import create_client
    
    yt_key = os.environ.get('YOUTUBE_API_KEY')
    # Ensure custom-secret-2 is your SERVICE_ROLE_KEY
    supabase = create_client(os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_KEY'))
    
    print("📡 Querying production quotes...")
    res = supabase.table('quotes').select(
        'id, episodes(title), podcasts(name)'
    ).is_('youtube_id', 'null').execute()
    
    quotes = res.data or []
    print(f"🚀 Found {len(quotes)} quotes needing YouTube IDs.")

    processed_episodes = {} 
    
    for q in quotes:
        podcast_name = q.get('podcasts', {}).get('name', 'Unknown')
        ep_data = q.get('episodes', {})
        episode_title = ep_data.get('title') or ep_data.get('episode_title') or "Unknown"
        
        clean_query = f"{podcast_name} {episode_title}".replace("Episode ", "").split(" with ")[0].split(" ft. ")[0].split(":")[0].strip()
        
        if clean_query not in processed_episodes:
            print(f"🔍 Searching YouTube: '{clean_query}'")
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {'part': 'id', 'q': clean_query, 'key': yt_key, 'maxResults': 1, 'type': 'video'}
            
            try:
                r = requests.get(url, params=params)
                data = r.json()
                if "error" in data:
                    if "quota" in data['error'].get('message', '').lower():
                        print("🛑 QUOTA EXCEEDED. Stopping.")
                        return
                    processed_episodes[clean_query] = None
                else:
                    items = data.get('items', [])
                    processed_episodes[clean_query] = items[0]['id']['videoId'] if items else None
            except:
                processed_episodes[clean_query] = None

        # --- REFINED UPDATE LOGIC ---
        target_yt_id = processed_episodes.get(clean_query)
        if target_yt_id:
            # We use the raw 'id' from the record to ensure the match is exact
            update_res = supabase.table('quotes').update({'youtube_id': target_yt_id}).eq('id', q['id']).execute()
            
            if update_res.data:
                print(f"   ✅ DB UPDATED: Quote {q['id']}")
            else:
                print(f"   ❌ DB FAILED: Quote {q['id']} - Check RLS settings or Keys.")

@app.local_entrypoint()
def main():
    run_production_cleanup.remote()
