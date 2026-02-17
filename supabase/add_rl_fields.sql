-- Run this in your Supabase SQL Editor

-- 1. Add quality_score to store the LLM's confidence/rating (0.0 to 1.0)
ALTER TABLE test_quotes 
ADD COLUMN IF NOT EXISTS quality_score DECIMAL(4,3);

-- 2. Add extraction_model to track which model generated the quote
ALTER TABLE test_quotes 
ADD COLUMN IF NOT EXISTS extraction_model TEXT DEFAULT 'gpt-4o-mini';

-- 3. Add column to track if a quote was used for training (future proofing)
ALTER TABLE test_quotes 
ADD COLUMN IF NOT EXISTS used_for_training BOOLEAN DEFAULT FALSE;

-- 4. Add index for faster sorting/filtering by quality
CREATE INDEX IF NOT EXISTS idx_test_quotes_quality ON test_quotes(quality_score DESC);

COMMENT ON COLUMN test_quotes.quality_score IS 'Quality score (0-1) assigned by the extraction model';
COMMENT ON COLUMN test_quotes.extraction_model IS 'Model identifier used for extraction (e.g., gpt-4o-mini, claude-3-5-sonnet)';
