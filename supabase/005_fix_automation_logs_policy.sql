-- Fix: Add UPDATE policy for automation_logs table
-- This allows the Modal backend to update log entries after processing completes

-- Allow service role to update logs
CREATE POLICY "Allow service role to update logs"
ON automation_logs FOR UPDATE
TO service_role
USING (true)
WITH CHECK (true);

-- Optional: Also allow anon to update (only if you need this for manual updates from frontend)
-- CREATE POLICY "Allow anon to update logs"
-- ON automation_logs FOR UPDATE
-- TO anon
-- USING (true)
-- WITH CHECK (true);
