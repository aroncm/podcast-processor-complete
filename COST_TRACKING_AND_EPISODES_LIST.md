# Cost Tracking & Episodes List Implementation

## Summary
Added comprehensive cost tracking and a processed episodes list to the Episode Processing admin tab.

## What Was Added

### 1. Database Changes (Supabase)
**New columns in `test_quotes` table:**
- `processing_cost` (DECIMAL) - API cost per quote (distributed from episode total)
- `duration_minutes` (DECIMAL) - Episode audio duration
- `episode_audio_url` (TEXT) - Full episode audio URL (was used but not in schema)

**New indexes:**
- `idx_test_quotes_episode_podcast` - For faster episode queries
- `idx_test_quotes_processing_cost` - For cost aggregation

### 2. Backend Changes (Modal)
**File:** `modal_app/full_processor.py`

**Changes:**
- Cost is now calculated and divided among all quotes from the same episode
- Each quote record saves: `processing_cost` and `duration_minutes`
- Cost formula: `(duration_minutes * $0.006) / number_of_quotes`

**Lines modified:** 340-367

### 3. Frontend Changes (Bolt.new)
**File:** `src/components/EpisodeQueue.tsx`

**Added:**
1. **4th Stats Card** - "Total API Cost" showing sum of all processing costs
2. **Processed Episodes List** - Collapsible section showing:
   - Episode name and podcast name
   - Date processed
   - Quote count per episode
   - Duration and cost per episode
   - Latest 20 episodes (sorted by date)

**New interfaces:**
```typescript
interface ProcessingStats {
  totalEpisodes: number;
  totalQuotes: number;
  avgQuotesPerEpisode: number;
  totalCost: number;  // ← NEW
}

interface ProcessedEpisode {
  episode_name: string;
  podcast_name: string;
  created_at: string;
  quote_count: number;
  duration_minutes: number;
  processing_cost: number;
}
```

**New functions:**
- `loadProcessedEpisodes()` - Loads and aggregates episode data
- `toggleEpisodesList()` - Shows/hides the collapsible episodes list

## UI Layout

```
Episode Processing Tab
┌─────────────────────────────────────────────────────────────┐
│ ┌─────────┬─────────┬──────────┬─────────────┐             │
│ │Episodes │ Quotes  │Avg/Ep    │Total Cost ← NEW           │
│ │ 15      │ 82      │ 5.5      │ $12.45                    │
│ └─────────┴─────────┴──────────┴─────────────┘             │
│                                                              │
│ [Podcast Feeds section...]                                  │
│                                                              │
│ [Episode Processing section...]                             │
│                                                              │
│ [Automated Processing section...]                           │
│                                                              │
│ ▼ Processed Episodes (15 total) ← NEW COLLAPSIBLE          │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ "The Daily" - Trump's Latest Move                    │   │
│ │ Open Market • Nov 17                                 │   │
│ │ 🗨 5 quotes • ⏱ 45.2 min • 💲 $0.27                 │   │
│ │─────────────────────────────────────────────────────│   │
│ │ "My First Million" - AI Winter                       │   │
│ │ MFM • Nov 16                                         │   │
│ │ 🗨 7 quotes • ⏱ 62.1 min • 💲 $0.37                 │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Steps

### Step 1: Run Database Migration
1. Go to your Supabase project dashboard
2. Click **SQL Editor**
3. Copy and paste the contents of: `supabase/004_cost_tracking_migration.sql`
4. Click **Run**

This creates the new columns and indexes.

### Step 2: Deploy Backend
```bash
cd /Users/craigaron/podcast-processor-new/podcast-processor-complete/modal_app
modal deploy full_processor.py
```

This updates the backend to save cost data for new episodes.

### Step 3: Update Frontend
Copy the updated `EpisodeQueue.tsx` to your Bolt.new project:
- Source: `/Users/craigaron/Desktop/Podtakes Dev/podtakes_updated_restore/src/components/EpisodeQueue.tsx`
- Destination: Your Bolt.new project at `src/components/EpisodeQueue.tsx`

## How It Works

### Cost Calculation
1. **During processing:** Backend calculates total episode cost: `duration * $0.006`
2. **Per quote:** Cost is divided equally among all quotes: `total_cost / quote_count`
3. **In database:** Each quote stores its portion of the cost
4. **In UI:** Frontend sums all quote costs to show total

### Episodes List
1. **Click to expand:** Click "Processed Episodes" header to load and display
2. **Lazy loading:** Data only loads when you expand the section
3. **Aggregation:** Groups quotes by episode name + podcast name
4. **Sorting:** Shows latest 20 episodes by processing date
5. **Refresh:** Reopening the section reloads fresh data

## Testing

### Test Cost Tracking
1. Process a new episode after deploying
2. Check Supabase `test_quotes` table - `processing_cost` should be populated
3. Check Episode Processing stats - Total Cost should increase
4. Verify: `(episode_duration * 0.006) / quote_count` = cost per quote

### Test Episodes List
1. Go to Admin → Episode Processing
2. Click "Processed Episodes (X total)" to expand
3. Should see list of processed episodes with:
   - Episode and podcast names
   - Processing date
   - Quote count, duration, and cost
4. Click again to collapse

## Data Migration Note

**Important:** Episodes processed BEFORE this update will have:
- ✅ `episode_audio_url` populated (already being saved)
- ❌ `processing_cost` = NULL
- ❌ `duration_minutes` = NULL

**Impact:**
- Old episodes will show $0.00 cost
- Old episodes will not show duration
- Total Cost stat will only reflect NEW episodes

**Optional backfill:** If you want to populate costs for old episodes, you would need to:
1. Query automation_logs to find historical processing costs
2. Update test_quotes records based on episode_name matching

## Files Modified

| File | Purpose | Status |
|------|---------|--------|
| `supabase/004_cost_tracking_migration.sql` | Database migration | ✅ Created |
| `modal_app/full_processor.py` | Save cost data | ✅ Modified |
| `src/components/EpisodeQueue.tsx` | Display cost & episodes | ✅ Modified |

## API Cost Breakdown

**Whisper API Pricing:** $0.006 per minute of audio

**Example episode (60 minutes, 8 quotes):**
- Total cost: 60 × $0.006 = **$0.36**
- Cost per quote: $0.36 / 8 = **$0.045**
- All 8 quotes store `processing_cost: 0.045`
- UI total cost: 8 × $0.045 = **$0.36** ✓

## Future Enhancements

Possible additions:
- Monthly/weekly cost reports
- Cost breakdown by podcast
- Episode search/filter in processed list
- Export episodes list to CSV
- Cost alerts/budgets
