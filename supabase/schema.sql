-- Test quotes table
CREATE TABLE IF NOT EXISTS test_quotes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  podcast_name TEXT,
  episode_name TEXT,
  speaker_name TEXT,
  category TEXT,
  quote_text TEXT NOT NULL,
  date_published TIMESTAMP WITH TIME ZONE,
  audio_clip_url TEXT,
  timestamp_start INTEGER,
  timestamp_end INTEGER,
  approval_status TEXT DEFAULT 'pending',
  test_run BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Test podcast feeds table
CREATE TABLE IF NOT EXISTS test_podcast_feeds (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  rss_url TEXT NOT NULL,
  active BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_test_quotes_approval ON test_quotes(approval_status);
CREATE INDEX IF NOT EXISTS idx_test_quotes_episode ON test_quotes(episode_name);
CREATE INDEX IF NOT EXISTS idx_test_quotes_created ON test_quotes(created_at DESC);

-- Insert default feed
INSERT INTO test_podcast_feeds (name, rss_url)
VALUES ('The Daily', 'https://feeds.simplecast.com/54nAGcIl')
ON CONFLICT DO NOTHING;
