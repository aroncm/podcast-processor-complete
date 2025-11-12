# Deployment Guide

## Modal Deployment
```bash
# Deploy functions
modal deploy modal_app/full_processor.py

# Get webhook URLs
modal app list
```

## Supabase Setup
1. Create project at supabase.com
2. Run schema.sql in SQL editor
3. Create 'audio-clips' storage bucket (public)
4. Copy service role key to .env

## Bolt Deployment
1. Copy bolt_admin files to your Bolt app
2. Push to GitHub
3. Connect to Bolt Cloud
4. Add environment variables
5. Deploy
