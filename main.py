#!/usr/bin/env python3
import os
import sys
import argparse

# Prefer absolute path imports for the local project
PROJECT_ROOT = "/media/talha/work/prokects"
VIDEO_DIR = os.path.join(PROJECT_ROOT, "Video")
if VIDEO_DIR not in sys.path:
	sys.path.insert(0, VIDEO_DIR)

from pipeline.run import run_pipeline  # type: ignore


def main() -> None:
	parser = argparse.ArgumentParser(description="YouTube → Transcript → Subtitles (root entrypoint)")
	parser.add_argument("--url", required=True)
	parser.add_argument("--asr_backend", choices=["local", "openai"], default="local")
	parser.add_argument("--whisper_model", default=os.environ.get("DEFAULT_WHISPER_MODEL", "base"))
	parser.add_argument("--llm_backend", choices=["openai"], default="openai")
	parser.add_argument("--subtitle_mode", choices=["soft", "burn"], default="soft")
	args = parser.parse_args()

	result = run_pipeline(
		url=args.url,
		asr_backend=args.asr_backend,
		whisper_model=args.whisper_model,
		llm_backend=args.llm_backend,
		subtitle_mode=args.subtitle_mode,
	)
	print(result)


if __name__ == "__main__":
	main() 