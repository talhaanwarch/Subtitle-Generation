# Pipeline

Runs the complete workflow:

1. Download YouTube video
2. Extract audio
3. Transcribe (local Whisper or OpenAI Whisper)
4. Enhance transcript via LLM
5. Add subtitles (soft or burned-in)

## Usage

```bash
python pipeline/run.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --asr_backend local|openai \
  --whisper_model base \
  --llm_backend openai \
  --subtitle_mode soft|burn
```
