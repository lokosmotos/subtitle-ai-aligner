# AI Subtitle Aligner 🎬

An AI-powered tool for automatically aligning English and Chinese subtitles.

## Features

- 🤖 AI-powered subtitle alignment using sentence transformers
- ⏱️ Timing-based matching with semantic understanding
- 📊 Real-time alignment results with confidence scores
- 💾 Download aligned SRT files (Chinese on top, English on bottom)
- 🌐 Web-based interface - no installation required

## How to Use

1. **Paste SRT Content**: Copy and paste your English and Chinese SRT files
2. **Align**: Click "Align Subtitles" to let AI find the best matches
3. **Review**: Check the alignment results and confidence scores
4. **Download**: Get your perfectly aligned SRT file for burning

## Deployment

This project is configured for deployment on Render.com:

### Backend (Python/Flask)
- **Service**: `subtitle-ai-backend`
- **Runtime**: Python 3.9
- **Start Command**: `cd backend && gunicorn app:app`

### Frontend (Static)
- **Service**: `subtitle-ai-frontend` 
- **Type**: Static Site
- **Publish Path**: `./frontend`

## Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
