# Automation Controls Setup Guide

## Summary
Added full automation controls to toggle the scheduled processor on/off and view processing history.

## What Changed

### 1. Backend Changes (Modal)
**File:** `modal_app/full_processor.py`

- Changed schedule from **every 6 hours** to **daily at midnight UTC**
- Added logic to check `automation_settings` table before running
- Added automatic logging to `automation_logs` table for every run
- Schedule will skip processing if disabled, logging the skip

### 2. Database Changes (Supabase)
**File:** `supabase/automation_settings_migration.sql`

Created two new tables:
- `automation_settings` - stores enable/disable toggle
- `automation_logs` - tracks all automated runs with results

### 3. Frontend Changes (React)
**File:** `src/components/EpisodeQueue.tsx`

Added:
- Toggle switch to enable/disable automation
- Real-time status display (enabled/paused)
- Recent activity log showing last 5 automated runs
- Color-coded status (success=green, skipped=yellow, error=red)

## Deployment Steps

### Step 1: Run Database Migration
1. Go to your Supabase project dashboard
2. Click **SQL Editor**
3. Copy and paste the contents of `supabase/automation_settings_migration.sql`
4. Click **Run**

This creates the tables and sets automation to **enabled** by default.

### Step 2: Deploy Backend Changes
```bash
cd modal_app
modal deploy full_processor.py
```

This updates the scheduled function to:
- Run daily at midnight UTC (instead of every 6 hours)
- Check the enable/disable setting before processing
- Log all runs to the database

### Step 3: Deploy Frontend Changes
Copy the updated `EpisodeQueue.tsx` to your Bolt.new project.

## How It Works

### Automation Toggle
- **ON (Green)**: Scheduled job runs daily at midnight UTC
- **OFF (Gray)**: Scheduled job skips processing and logs the skip

### Activity Log
Shows the last 5 automated runs with:
- **Success**: Green - shows quotes extracted and episodes processed
- **Skipped**: Yellow - automation was disabled
- **Error**: Red - shows error message

### Schedule
- **Frequency**: Daily at 00:00 UTC
- **Cron**: `0 0 * * *`
- **Can't be changed**: Schedule is baked into Modal deployment
- **Can be disabled**: Toggle prevents processing without redeploying

## Testing

### Test the Toggle
1. Go to Admin → Episode Processing
2. Click the toggle switch (should turn gray)
3. Check Supabase `automation_settings` table - `value` should be `false`
4. Toggle back on - `value` should be `true`

### Test Scheduled Processing
Since it runs at midnight UTC, you can either:
1. **Wait for next run** - Check logs the next day
2. **Manually trigger** - Run one episode manually and check it logs correctly
3. **View logs** - Query `automation_logs` table to see past runs

## Monitoring

### Check if automation is working:
```sql
-- View recent automated runs
SELECT * FROM automation_logs
WHERE run_type = 'scheduled'
ORDER BY created_at DESC
LIMIT 10;
```

### Check current setting:
```sql
-- Check if automation is enabled
SELECT * FROM automation_settings
WHERE key = 'automated_processing_enabled';
```

## Troubleshooting

### Toggle not working
- Check browser console for errors
- Verify Supabase RLS policies allow updates
- Ensure `automation_settings` table exists

### No logs appearing
- Scheduled function may not have run yet (runs at midnight UTC)
- Check Modal logs to see if function is executing
- Verify `automation_logs` table exists

### Processing still running when disabled
- Check `automation_settings.value` is actually `false`
- Modal deployment may not have updated - redeploy
- Check Modal logs to see if function is checking the setting

## Future Enhancements

Possible additions:
- Email notifications for successful/failed runs
- Adjustable schedule (hourly, weekly, etc.)
- Run history with detailed results
- Manual trigger for scheduled logic
