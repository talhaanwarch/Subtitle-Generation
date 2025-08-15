# Translator Module

This module provides functionality to translate subtitle segments from one language to another using LLM services.

## Features

- Translates subtitle segments while preserving timing information
- Supports any target language via natural language specification
- Maintains subtitle formatting and structure
- Uses OpenAI-compatible API endpoints

## Usage

### Standalone Usage

```bash
python translator/translate_transcript.py \
    --video-id "VIDEO_ID" \
    --input-json "path/to/enhanced.json" \
    --target-lang "Spanish" \
    --backend openai
```

### Pipeline Integration

Translation is automatically integrated into the main pipeline when the `--target-lang` parameter is provided:

```bash
python main.py \
    --url "https://youtube.com/watch?v=VIDEO_ID" \
    --target-lang "French"
```

## Parameters

- `--target-lang`: Target language for translation (e.g., "Spanish", "French", "German", "Japanese")
- `--backend`: Translation backend (currently supports "openai")
- `--video-id`: Video ID for organizing output files
- `--input-json`: Path to JSON file containing segments to translate

## Output

The translator creates:
- `translated_{lang_code}.json`: JSON file with translated segments and metadata
- `translated_{lang_code}.srt`: SRT subtitle file with translations

## Environment Variables

Requires the same environment variables as the enhancer:
- `GROQ_API_KEY`: API key for Groq/OpenAI-compatible service
- `LLM_MODEL_NAME`: Model name to use for translation

## Language Support

The translator accepts natural language descriptions for target languages:
- "Spanish" or "Español"
- "French" or "Français" 
- "German" or "Deutsch"
- "Japanese" or "日本語"
- "Chinese (Simplified)" or "中文"
- Any other language name

The system will automatically generate appropriate file names using language codes.
