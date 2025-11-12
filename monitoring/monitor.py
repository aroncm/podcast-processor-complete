#!/usr/bin/env python3
'''Monitor podcast processing system'''

import os
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def monitor_system():
    supabase = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_KEY')
    )
    
    # Get stats from last 24 hours
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    
    # Test system stats
    test_quotes = supabase.table('test_quotes') \
        .select('*') \
        .gte('created_at', cutoff) \
        .execute()
    
    print("\n" + "=" * 50)
    print("📊 SYSTEM STATUS - Last 24 Hours")
    print("=" * 50)
    print(f"Test Quotes: {len(test_quotes.data if test_quotes.data else [])}")
    
    # Episode processing
    if test_quotes.data:
        episodes = set(q['episode_name'] for q in test_quotes.data)
        print(f"Episodes Processed: {len(episodes)}")
        
        for ep in episodes:
            quotes_count = len([q for q in test_quotes.data if q['episode_name'] == ep])
            print(f"  • {ep[:50]}... ({quotes_count} quotes)")
    
    # Cost estimate
    modal_cost = len(episodes if test_quotes.data else []) * 0.02
    print(f"\n💰 Estimated Costs:")
    print(f"  Modal: ~${modal_cost:.2f}")
    print(f"  Whisper API: ~${modal_cost:.2f}")

if __name__ == "__main__":
    monitor_system()
