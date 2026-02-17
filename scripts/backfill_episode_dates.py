#!/usr/bin/env python3
"""
Backfill script to fix incorrect date_published values in test_quotes table.
Finds episodes where date_published matches created_at and updates them with the correct RSS feed date.
"""

import os
import sys
from datetime import datetime
from supabase import create_client
import feedparser

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_env():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    with open(env_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

def get_episode_published_date(episode):
    """
    Extract published date from RSS episode entry.
    Tries multiple fields to find the most accurate date.
    """
    from dateutil import parser as date_parser

    # Try published_parsed first (most reliable)
    if hasattr(episode, 'published_parsed') and episode.published_parsed:
        try:
            return datetime(*episode.published_parsed[:6])
        except (TypeError, ValueError):
            pass

    # Try updated_parsed as fallback
    if hasattr(episode, 'updated_parsed') and episode.updated_parsed:
        try:
            return datetime(*episode.updated_parsed[:6])
        except (TypeError, ValueError):
            pass

    # Try parsing string fields as last resort
    for field in ['published', 'pubDate', 'updated']:
        if hasattr(episode, field):
            try:
                return date_parser.parse(getattr(episode, field))
            except (ValueError, TypeError, AttributeError):
                continue

    return None

def find_episode_in_feed(feed_url, episode_title):
    """
    Find an episode in an RSS feed by title.
    Returns the episode entry if found, None otherwise.
    """
    print(f"  🔍 Searching RSS feed for: {episode_title[:50]}...")
    parsed = feedparser.parse(feed_url)

    # Try exact match first
    for entry in parsed.entries:
        if entry.title == episode_title:
            return entry

    # Try partial match (episode titles might be truncated in DB)
    for entry in parsed.entries:
        if entry.title.startswith(episode_title[:50]) or episode_title.startswith(entry.title[:50]):
            return entry

    return None

def backfill_dates():
    """Main backfill function"""
    load_env()

    supabase = create_client(
        os.environ['SUPABASE_URL'],
        os.environ['SUPABASE_KEY']
    )

    print("\n" + "="*70)
    print("📅 BACKFILL EPISODE PUBLISHED DATES")
    print("="*70)

    # Get all quotes
    print("\n📊 Fetching all quotes from database...")
    result = supabase.table('test_quotes').select('id, episode_name, podcast_name, date_published, created_at').execute()

    # Group by episode
    episodes = {}
    for quote in result.data:
        ep_key = (quote['episode_name'], quote['podcast_name'])
        if ep_key not in episodes:
            episodes[ep_key] = {
                'quote_ids': [],
                'created_at': quote['created_at'][:10],
                'date_published': quote['date_published'][:10] if quote['date_published'] else None
            }
        episodes[ep_key]['quote_ids'].append(quote['id'])

    # Find suspicious episodes (created = published)
    suspicious = []
    for (ep_name, podcast_name), data in episodes.items():
        if data['date_published'] and data['created_at'] == data['date_published']:
            suspicious.append({
                'episode_name': ep_name,
                'podcast_name': podcast_name,
                'quote_ids': data['quote_ids'],
                'current_date': data['date_published']
            })

    print(f"\n✅ Total episodes: {len(episodes)}")
    print(f"❌ Episodes with suspicious dates: {len(suspicious)}")

    if not suspicious:
        print("\n🎉 No episodes need backfilling!")
        return

    # Get all feeds
    feeds_result = supabase.table('test_podcast_feeds').select('*').execute()
    feeds_by_name = {feed['name']: feed for feed in feeds_result.data}

    print(f"\n📡 Found {len(feeds_by_name)} podcast feeds\n")
    print("="*70)

    # Process each suspicious episode
    fixed_count = 0
    failed_count = 0

    for i, episode_data in enumerate(suspicious, 1):
        episode_name = episode_data['episode_name']
        podcast_name = episode_data['podcast_name']
        quote_ids = episode_data['quote_ids']
        current_date = episode_data['current_date']

        print(f"\n[{i}/{len(suspicious)}] Processing: {episode_name[:50]}")
        print(f"  📻 Podcast: {podcast_name}")
        print(f"  📝 Quotes: {len(quote_ids)}")
        print(f"  📅 Current date: {current_date}")

        # Find the feed for this podcast
        if podcast_name not in feeds_by_name:
            print(f"  ❌ Feed not found for podcast: {podcast_name}")
            failed_count += 1
            continue

        feed = feeds_by_name[podcast_name]

        # Find episode in RSS feed
        episode_entry = find_episode_in_feed(feed['rss_url'], episode_name)

        if not episode_entry:
            print(f"  ❌ Episode not found in RSS feed")
            failed_count += 1
            continue

        # Extract correct published date
        correct_date = get_episode_published_date(episode_entry)

        if not correct_date:
            print(f"  ⚠️  No published date found in RSS feed")
            failed_count += 1
            continue

        correct_date_str = correct_date.isoformat()
        correct_date_display = correct_date.strftime('%Y-%m-%d')

        # Check if date actually needs updating
        if correct_date_display == current_date:
            print(f"  ✅ Date is already correct: {correct_date_display}")
            continue

        print(f"  ✨ Correct date found: {correct_date_display}")
        print(f"  📝 Updating {len(quote_ids)} quotes...")

        # Update all quotes for this episode
        try:
            supabase.table('test_quotes').update({
                'date_published': correct_date_str
            }).in_('id', quote_ids).execute()

            print(f"  ✅ Successfully updated!")
            fixed_count += 1

        except Exception as e:
            print(f"  ❌ Error updating: {str(e)}")
            failed_count += 1

    # Summary
    print("\n" + "="*70)
    print("📊 BACKFILL SUMMARY")
    print("="*70)
    print(f"✅ Episodes fixed: {fixed_count}")
    print(f"❌ Episodes failed: {failed_count}")
    print(f"⏭️  Episodes skipped (already correct): {len(suspicious) - fixed_count - failed_count}")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        backfill_dates()
    except KeyboardInterrupt:
        print("\n\n⚠️  Backfill interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
