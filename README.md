# YouTube Video Transcription & Subtitles Pipeline

A modular pipeline to download YouTube videos, transcribe audio, enhance transcripts with LLM, and add subtitles.

## Features

- **Flexible ASR**: Groq API (cloud) or local Whisper models
- **LLM Enhancement**: Improve grammar, punctuation, and capitalization
- **Translation**: Translate to any target language
- **Subtitle Options**: Soft-embedded or burned-in subtitles
- **YAML Configuration**: Easy parameter management

## System Requirements

- **Tested on**: Ubuntu 24.04, Python 3.12
- **FFmpeg**: Required for audio/video processing

### Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install ffmpeg
```

## Installation

### 1. Clone and Setup Virtual Environment

```bash
git clone <repository-url>
cd Subtitle-Generation
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

**For Cloud Processing (Groq API):**
```bash
pip install -r groq_requirements.txt
```

**For Local Processing (Whisper models):**
```bash
pip install -r requirements.txt
```

### 3. Configuration

Copy and edit the configuration file:
```bash
cp config.sample.yaml config.yaml
nano config.yaml  # Edit with your settings
```

**Required settings:**
- Set `video.url` to your YouTube URL
- Set `api.groq_api_key` for Groq API usage
- Configure `llm.enhancer.enabled` and `llm.translator.enabled` as needed

## Quick Start

### Basic Usage
```bash
python main.py
```

### With Command Line Overrides
```bash
python main.py --url "https://youtube.com/watch?v=VIDEO_ID" --target-lang "Spanish"
```

## Configuration Examples

### Cloud Processing (Groq API)
```yaml
video:
  url: "https://www.youtube.com/watch?v=s01QuLpjISc"

asr:
  backend: "groq"

llm:
  backend: "groq"
  model: "llama3-8b-8192"
  enhancer:
    enabled: true
  translator:
    enabled: true
    target_language: "Spanish"

api:
  groq_api_key: "gsk_your_api_key_here"
```

### Local Processing (No API Required)
```yaml
video:
  url: "https://www.youtube.com/watch?v=s01QuLpjISc"

asr:
  backend: "local"
  whisper_model: "medium"

llm:
  enhancer:
    enabled: false
  translator:
    enabled: false

subtitles:
  mode: "soft"
```

### Enhancement Only (No Translation)
```yaml
llm:
  enhancer:
    enabled: true
  translator:
    enabled: false
```

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --config, -c        Configuration file path (default: config.yaml)
  --url               YouTube URL (overrides config)
  --asr-backend       ASR backend: local, groq
  --whisper-model     Whisper model for local ASR
  --llm-backend       LLM backend: groq
  --subtitle-mode     Subtitle mode: soft, burn
  --target-lang       Target language for translation
  --box-opacity       Opacity for burned subtitles (0.0-1.0)
```

## Available Models

### Groq ASR Models
- `distil-whisper-large-v3-en` (recommended)
- `whisper-large-v3`

### Groq LLM Models
- `llama3-8b-8192` (fast, good quality)
- `llama3-70b-8192` (slower, higher quality)
- `openai/gpt-oss-120b` (alternative)
- `mixtral-8x7b-32768` (large context window)

### Local Whisper Models
- `base` (fastest, lowest quality)
- `small`, `medium`, `large`, `large-v2`, `large-v3` (increasing quality/size)

## Project Structure

```
Subtitle-Generation/
├── config.yaml              # Main configuration
├── config.sample.yaml       # Configuration template
├── main.py                  # Main entry point
├── requirements.txt         # Dependencies for local processing
├── groq_requirements.txt    # Dependencies for cloud processing
├── downloader/              # YouTube video download
├── transcriber/             # Audio transcription (local/Groq)
├── enhancer/               # LLM transcript enhancement
├── translator/             # LLM translation
├── subtitles/              # Subtitle generation
├── pipeline/               # End-to-end orchestration
└── utils/                  # Shared utilities
```

## Output Structure

Files are organized under `outputs/<video_id>/`:
```
outputs/VIDEO_ID/
├── video/          # Downloaded video
├── audio/          # Extracted audio
├── transcripts/    # Raw ASR output (JSON/SRT)
├── enhanced/       # Enhanced transcripts
├── translated/     # Translated transcripts (if enabled)
└── subtitled/      # Final video with subtitles
```

## Environment Variables

Alternative to config file settings:
```bash
export GROQ_API_KEY="gsk_your_api_key_here"
export LLM_MODEL_NAME="llama3-8b-8192"
export ASR_MODEL_NAME="distil-whisper-large-v3-en"
```

## API Keys

Get your Groq API key from: https://console.groq.com/keys

## Notes

- **Groq API**: Fast and cost-effective for both ASR and LLM tasks
- **Local processing**: No API costs but requires more computational resources
- **Burned subtitles**: Re-encode video (slower) vs soft subtitles (faster)
- **Large models**: Higher quality but slower processing and API costs