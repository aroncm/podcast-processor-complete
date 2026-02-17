#!/usr/bin/env python3
"""
Backfill processing_cost for existing test_quotes rows using estimated GPT and clip costs.

Uses the following cost model:
  - Whisper transcription: $0.006 per audio minute
  - GPT-4o quote extraction: $0.0686 per quote (derived from 457,665 tokens costing $14.40 across 210 quotes)
  - Clip generation: $0.002 per quote (Modal compute + Supabase storage estimate)

Adjust the ESTIMATED_* constants below if your real-world rates differ.
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from supabase import create_client
from dateutil import parser as date_parser

WHISPER_COST_PER_MIN = 0.006
ESTIMATED_GPT_COST_PER_QUOTE = 0.0686
ESTIMATED_CLIP_COST_PER_QUOTE = 0.002
BATCH_SIZE = 200


def format_episode_guid(title, date_value=None, fallback=None):
    date_part = None
    if date_value:
        try:
            parsed = date_parser.parse(str(date_value))
            date_part = parsed.date().isoformat()
        except (ValueError, TypeError):
            date_part = None

    unique = date_part or fallback or "unknown"
    return f"{(title or 'unknown').strip()}|{unique}"


def load_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file at {env_path}")
    load_dotenv(env_path)


def fetch_all_quotes(client) -> List[Dict]:
    quotes: List[Dict] = []
    page = 0
    page_size = 1000

    while True:
        start = page * page_size
        end = start + page_size - 1
        resp = client.table("test_quotes").select("*").range(start, end).execute()
        data = resp.data or []
        quotes.extend(data)

        if len(data) < page_size:
            break
        page += 1

    return quotes


def chunked(items: List[Dict], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main():
    load_env()

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    quotes = fetch_all_quotes(client)

    episodes: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for quote in quotes:
        key = (quote.get("podcast_name") or "unknown", quote.get("episode_name") or "unknown")
        episodes[key].append(quote)

    updates: List[Dict] = []

    for (podcast_name, episode_name), episode_quotes in episodes.items():
        duration = episode_quotes[0].get("duration_minutes") or 0
        quote_count = len(episode_quotes)

        whisper_cost = (duration or 0) * WHISPER_COST_PER_MIN
        gpt_cost = quote_count * ESTIMATED_GPT_COST_PER_QUOTE
        clip_cost = quote_count * ESTIMATED_CLIP_COST_PER_QUOTE
        total_cost = whisper_cost + gpt_cost + clip_cost

        per_quote_cost = round(total_cost / max(quote_count, 1), 4)

        for quote in episode_quotes:
            episode_guid = format_episode_guid(
                quote.get("episode_name"),
                quote.get("date_published"),
                quote.get("episode_audio_url") or quote["id"]
            )
            updates.append({
                "id": quote["id"],
                "processing_cost": per_quote_cost,
                "episode_guid": episode_guid
            })

    for batch in chunked(updates, BATCH_SIZE):
        client.table("test_quotes").upsert(batch).execute()

    print(f"Updated {len(updates)} quotes across {len(episodes)} episodes.")


if __name__ == "__main__":
    main()
