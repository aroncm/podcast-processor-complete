-- Migration: Add cost tracking and duration fields
-- Run this in your Supabase SQL editor

-- Add cost tracking fields to test_quotes table
ALTER TABLE test_quotes
  ADD COLUMN IF NOT EXISTS episode_audio_url TEXT,
  ADD COLUMN IF NOT EXISTS processing_cost DECIMAL(10,2),
  ADD COLUMN IF NOT EXISTS duration_minutes DECIMAL(10,2);

-- Add index for faster episode queries (used in processed episodes list)
CREATE INDEX IF NOT EXISTS idx_test_quotes_episode_podcast
  ON test_quotes(episode_name, podcast_name, created_at DESC);

-- Add index for cost aggregation queries
CREATE INDEX IF NOT EXISTS idx_test_quotes_processing_cost
  ON test_quotes(processing_cost)
  WHERE processing_cost IS NOT NULL;

-- Add helpful comment
COMMENT ON COLUMN test_quotes.processing_cost IS 'API cost in USD for processing this quote (Whisper API: $0.006 per minute)';
COMMENT ON COLUMN test_quotes.duration_minutes IS 'Episode audio duration in minutes';
COMMENT ON COLUMN test_quotes.episode_audio_url IS 'URL to the full episode audio file';
