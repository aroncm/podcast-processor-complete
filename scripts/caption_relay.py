"""Nightly caption relay for PodThreads (Oct 28, 2026). Runs on Craig's Mac from launchd (com.podthreads.caption-relay).

YouTube blocks caption requests from cloud servers, so the backend cannot verify a quote's timing on its own. This
script fetches the captions from this Mac, slowly, and hands them to the backend's own audited steps:
  1. apply_relayed_youtube_alignments: the strict timing check (unchanged standard); a match is recorded as verified.
  2. propose_spoken_words: for quotes that still fail (old-pipeline paraphrases), the speaker's own words as a proposal
     the editor applies on the episode screen.
Captions are cached on disk, so each video is fetched once. When YouTube starts refusing requests the script stops
fetching and processes what it has; the next run carries on. Safe to run any number of times (a lock prevents overlap).
"""
import fcntl
import gzip
import hashlib
import html
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("MODAL_PROFILE", "aron-personal")
import modal  # noqa: E402

HOME = Path.home() / "Library" / "Application Support" / "PodThreads"
CACHE = HOME / "captions"
APP = "podcast-processor-full"
PAUSE_SECONDS = 30          # between YouTube requests; faster runs got this Mac blocked
NO_CAPTIONS_RETRY_DAYS = 7  # a video without captions is asked again after a week
BATCH = 100


def log(*parts):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *parts, flush=True)


def cached(youtube_id):
    path = CACHE / f"{youtube_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if isinstance(data, dict) and data.get("none"):   # no captions last time
        if datetime.now(timezone.utc) - datetime.fromisoformat(data["at"]) > timedelta(days=NO_CAPTIONS_RETRY_DAYS):
            path.unlink()
            return None
        return []
    return data


def fetch(api, youtube_id):
    """Captions for one video, [] when it has none, or None when YouTube is refusing requests (stop for tonight)."""
    from youtube_transcript_api import _errors as errors
    try:
        events = [{"text": html.unescape(s.text), "start": s.start, "end": s.start + max(s.duration, 0.01), "source": "youtube_transcript_api"}
                  for s in api.fetch(youtube_id, languages=["en"])]
        (CACHE / f"{youtube_id}.json").write_text(json.dumps(events))
        return events
    except (errors.IpBlocked, errors.RequestBlocked) as exc:
        log("YouTube is refusing requests; stopping fetches for this run:", type(exc).__name__)
        return None
    except (errors.TranscriptsDisabled, errors.NoTranscriptFound, errors.VideoUnavailable, errors.VideoUnplayable) as exc:
        (CACHE / f"{youtube_id}.json").write_text(json.dumps({"none": True, "at": datetime.now(timezone.utc).isoformat(), "why": type(exc).__name__}))
        return []
    except Exception as exc:
        if "429" in str(exc) or "Too Many Requests" in str(exc):
            log("YouTube rate limit; stopping fetches for this run")
            return None
        log("caption fetch failed", youtube_id, type(exc).__name__, str(exc)[:120])
        return []


def bundle(tracks):
    blob = gzip.compress(json.dumps(tracks).encode())
    return blob, hashlib.sha256(blob).hexdigest()


def main():
    from youtube_transcript_api import YouTubeTranscriptApi
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = open(HOME / "relay.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another run is in progress"); return
    work = modal.Function.from_name(APP, "caption_relay_worklist").remote()
    log(f"{len(work)} pending quotes need timing, in {len({w['youtube_id'] for w in work})} videos")
    api, tracks, blocked = YouTubeTranscriptApi(), {}, False
    for youtube_id in sorted({w["youtube_id"] for w in work}):
        events = cached(youtube_id)
        if events is None and not blocked:
            events = fetch(api, youtube_id)
            blocked = events is None
            time.sleep(PAUSE_SECONDS)
        if events:
            tracks[youtube_id] = events
    log(f"captions on hand for {len(tracks)} videos")

    # 1. the strict timing check, for quotes that have not been through it with these captions
    align = modal.Function.from_name(APP, "apply_relayed_youtube_alignments")
    targets = [w for w in work if w["youtube_id"] in tracks and not w["spoken_words"]]
    verified = 0
    for i in range(0, len(targets), BATCH):
        batch = targets[i:i + BATCH]
        blob, digest = bundle({v: tracks[v] for v in {w["youtube_id"] for w in batch}})
        try:
            out = align.remote("all_test", [w["id"] for w in batch], blob, digest, dry_run=False)
            verified += int(out.get("verified") or 0)
        except Exception as exc:
            log("timing check failed for a batch:", str(exc)[:200])
    log(f"timing verified for {verified} quotes")

    # 2. the speaker's own words, for quotes that still fail and have not been looked at
    remaining = {w["id"] for w in modal.Function.from_name(APP, "caption_relay_worklist").remote() if not w["spoken_words"]}
    targets = [w for w in targets if w["id"] in remaining]
    propose = modal.Function.from_name(APP, "propose_spoken_words")
    for i in range(0, len(targets), BATCH):
        batch = targets[i:i + BATCH]
        blob, digest = bundle({v: tracks[v] for v in {w["youtube_id"] for w in batch}})
        try:
            log("speaker's own words:", propose.remote([w["id"] for w in batch], blob, digest))
        except Exception as exc:
            log("spoken words failed for a batch:", str(exc)[:200])
    log("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("relay failed:", type(exc).__name__, str(exc)[:300])
        sys.exit(1)
