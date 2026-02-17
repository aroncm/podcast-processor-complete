-- Migration: Add automation settings and logs
-- Run this in your Supabase SQL editor

-- Create automation settings table
CREATE TABLE IF NOT EXISTS automation_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key TEXT UNIQUE NOT NULL,
    value BOOLEAN NOT NULL DEFAULT true,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Insert default setting for automated processing (enabled by default)
INSERT INTO automation_settings (key, value)
VALUES ('automated_processing_enabled', true)
ON CONFLICT (key) DO NOTHING;

-- Create automation logs table to track scheduled runs
CREATE TABLE IF NOT EXISTS automation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type TEXT NOT NULL, -- 'scheduled', 'manual', 'batch'
    status TEXT NOT NULL, -- 'success', 'error', 'skipped', 'running'
    result JSONB,
    error_message TEXT,
    episodes_processed INTEGER DEFAULT 0,
    quotes_extracted INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_automation_logs_created_at ON automation_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_automation_logs_run_type ON automation_logs(run_type);
CREATE INDEX IF NOT EXISTS idx_automation_logs_status ON automation_logs(status);

-- Enable RLS
ALTER TABLE automation_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE automation_logs ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read settings
CREATE POLICY "Allow authenticated users to read settings"
ON automation_settings FOR SELECT
TO authenticated
USING (true);

-- Allow authenticated users to update settings
CREATE POLICY "Allow authenticated users to update settings"
ON automation_settings FOR UPDATE
TO authenticated
USING (true);

-- Allow authenticated users to read logs
CREATE POLICY "Allow authenticated users to read logs"
ON automation_logs FOR SELECT
TO authenticated
USING (true);

-- Allow service role to insert logs
CREATE POLICY "Allow service role to insert logs"
ON automation_logs FOR INSERT
TO service_role
USING (true);

-- Allow anon users to read settings (for public access)
CREATE POLICY "Allow anon to read settings"
ON automation_settings FOR SELECT
TO anon
USING (true);

COMMENT ON TABLE automation_settings IS 'Stores automation configuration like enable/disable flags';
COMMENT ON TABLE automation_logs IS 'Tracks all automated processing runs with results';
