# Transcriber

Provides two ASR backends producing precise timestamps:

- `local`: open-source Whisper via `openai-whisper` or `faster-whisper`
- `openai`: Whisper API via `openai` client

Outputs JSON segments and SRT files under `outputs/<video_id>/transcripts/`.

## Usage (local)

```bash
python transcriber/transcribe_local.py \
  --audio /path/to/audio.wav \
  --video-id ABC123 \
  --model base
```

## Usage (OpenAI)

```bash
python transcriber/transcribe_openai.py \
  --audio /path/to/audio.wav \
  --video-id ABC123
```
