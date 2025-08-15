# YouTube Video Transcription & Subtitles Pipeline

This project provides a modular pipeline to:

1. Download a YouTube video
2. Transcribe the audio with precise timestamps (choose Groq, OpenAI Whisper API, or local open-source Whisper)
3. Improve the transcript using an LLM
4. Add the transcript as subtitles (soft-embedded or burned-in)

## Structure

- `downloader/`: Download YouTube videos using `yt-dlp`
- `transcriber/`: Transcribe audio using Groq, OpenAI Whisper, or local Whisper
- `enhancer/`: Improve transcript with an LLM
- `subtitles/`: Generate and attach subtitles to video
- `pipeline/`: Orchestrated end-to-end runner
- `utils/`: Shared helpers for logging, paths, ffmpeg, I/O, timecodes, config


### FFmpeg Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install ffmpeg
```

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html) or use Chocolatey:
```bash
choco install ffmpeg
```

## Setup

Activate your existing virtual environment and install dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Copy the example env and set required keys:

```bash
cp env.sample .env
# edit .env to add your keys
```

### Environment Configuration

The project supports multiple backends:

- **Groq** (recommended): Fast and cost-effective for both ASR and LLM tasks
- **Local**: Open-source models running locally

Configure your `.env` file with the appropriate keys:

```bash
# For Groq (recommended)
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL_NAME=openai/gpt-oss-120b
ASR_MODEL_NAME=distil-whisper-large-v3-en

# For OpenAI (alternative)
OPENAI_API_KEY=your_openai_api_key_here
```

## Quickstart (End-to-End)

### Using Groq (Recommended)

```bash
# Example: run full pipeline with Groq for both ASR and LLM, with burned-in subtitles
python pipeline/run.py \
  --url "https://www.youtube.com/watch?v=s01QuLpjISc" \
  --asr_backend groq \
  --llm_backend groq \
  --subtitle_mode burn
```

### Using OpenAI (to be tested)

```bash
# Example: run full pipeline with OpenAI Whisper + OpenAI LLM, and soft-embed subtitles
python pipeline/run.py \
  --url "https://www.youtube.com/watch?v=s01QuLpjISc" \
  --asr_backend openai \
  --llm_backend openai \
  --subtitle_mode burn
```

### Using Local Whisper  (to be tested)

```bash
python pipeline/run.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --asr_backend local \
  --whisper_model base \
  --subtitle_mode soft
```

## Outputs

Artifacts are organized under `outputs/<video_id>/`:

- `video/`: downloaded video files
- `audio/`: extracted audio
- `transcripts/`: raw ASR JSON and SRT
- `enhanced/`: improved transcripts (JSON and SRT)
- `subtitled/`: final video with subtitles (soft or burned-in)

## Per-Component Usage

Each component has its own README:

- `downloader/README.md`
- `transcriber/README.md`
- `enhancer/README.md`
- `subtitles/README.md`
- `pipeline/README.md`

## Testing

This project has been tested on:
- **Ubuntu 24.04** 
- **python 3.12** 

## Notes

- Large models and API usage incur compute/API costs. Use chunking options for long videos.
- Burned-in subtitles re-encode the video, which is slower than soft-embedding.
- Local Whisper speed depends heavily on PyTorch/CUDA setup.
- Groq provides fast inference with competitive pricing for both ASR and LLM tasks.
