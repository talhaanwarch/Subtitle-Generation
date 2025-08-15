# Enhancer

Improves raw ASR transcripts using an LLM (OpenAI by default). It fixes punctuation, casing, formatting, and merges/splits segments while preserving timestamps as much as possible.

## Usage

```bash
python enhancer/enhance_transcript.py \
  --video-id ABC123 \
  --backend openai \
  --input-json outputs/ABC123/transcripts/asr_local.json
```
