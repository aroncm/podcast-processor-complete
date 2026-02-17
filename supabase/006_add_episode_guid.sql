-- Migration: add episode_guid for duplicate detection
-- Run in Supabase SQL editor

ALTER TABLE test_quotes
  ADD COLUMN IF NOT EXISTS episode_guid TEXT;

CREATE INDEX IF NOT EXISTS idx_test_quotes_episode_guid
  ON test_quotes(episode_guid);
