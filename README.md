# Podcast Processor System

A complete podcast processing system with AI-powered quote extraction and audio clip generation.

## Features
- 🎙️ Automatic podcast episode processing
- 🤖 AI quote extraction using GPT
- 🎬 Intelligent audio clip creation
- 📊 Admin dashboard for quote approval
- ⚡ 10x faster than traditional processing

## Tech Stack
- **Processing**: Modal (serverless compute)
- **Database**: Supabase
- **AI**: OpenAI Whisper & GPT-3.5/4
- **Frontend**: Next.js (Bolt)
- **Storage**: Supabase Storage

## Setup
1. Copy `.env.example` to `.env` and fill in your credentials
2. Deploy Modal functions: `modal deploy modal_app/full_processor.py`
3. Run database migrations in Supabase
4. Deploy Bolt admin UI

## Usage
- Process episodes: `modal run modal_app/full_processor.py::process_episode_with_ai`
- Create clips: `modal run modal_app/full_processor.py::create_audio_clip --quote-id ID`
- Monitor: `python monitoring/monitor.py`
