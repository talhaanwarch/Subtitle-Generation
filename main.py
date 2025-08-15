#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import traceback
# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.run import run_pipeline_with_config  # type: ignore
from utils.config import load_config, Config  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube → Transcript → Subtitles (root entrypoint)")
    parser.add_argument("--config", "-c", help="Path to configuration file (default: config.yaml)")
    parser.add_argument("--url", help="YouTube URL (overrides config)")
    parser.add_argument("--asr-backend", choices=["local", "groq"], help="ASR backend (overrides config)")
    parser.add_argument("--whisper-model", help="Whisper model for local ASR (overrides config)")
    parser.add_argument("--llm-backend", choices=["groq"], help="LLM backend (overrides config)")
    parser.add_argument("--subtitle-mode", choices=["soft", "burn"], help="Subtitle mode (overrides config)")
    parser.add_argument("--target-lang", help="Target language for translation (overrides config)")
    parser.add_argument("--box-opacity", type=float, help="Opacity of subtitle background box (overrides config)")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config or 'config.yaml'}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.url:
        config.video.url = args.url
    if args.asr_backend:
        config.asr.backend = args.asr_backend
    if args.whisper_model:
        config.asr.whisper_model = args.whisper_model
    if args.llm_backend:
        config.llm.backend = args.llm_backend
    if args.subtitle_mode:
        config.subtitles.mode = args.subtitle_mode
    if args.target_lang:
        config.llm.translator.target_language = args.target_lang
        config.llm.translator.enabled = True
    if args.box_opacity is not None:
        config.subtitles.box_opacity = args.box_opacity

    # Validate required parameters
    if not config.video.url:
        print("Error: YouTube URL is required. Provide it via --url or in config file.")
        sys.exit(1)

    # Run pipeline
    try:
        result = run_pipeline_with_config(config)
        print(result)
    except Exception as e:
        print(f"Error running pipeline: {e},traceback {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main() 