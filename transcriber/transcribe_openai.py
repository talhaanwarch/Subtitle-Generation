import os
import sys
import argparse
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# ensure package path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.paths import ensure_workdirs
from utils.logging_utils import get_logger
from utils.io_utils import write_json, write_srt

logger = get_logger(__name__)


def transcribe_with_openai(audio_path: str) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))
    with open(audio_path, "rb") as f:
        # Attempt to use verbose_json to get segments when supported
        transcript = client.audio.transcriptions.create(
            model=os.environ.get("ASR_MODEL_NAME"),
            file=f,
            response_format="verbose_json",
            temperature=0.0,
        )
    segments: List[Dict[str, Any]] = []
    # Map to our schema if segments are present
    if hasattr(transcript, "segments") and transcript.segments:
        for seg in transcript.segments:
            start = float(getattr(seg, "start", 0.0) or 0.0)
            end = float(getattr(seg, "end", start) or start)
            text = str(getattr(seg, "text", "")).strip()
            segments.append({"start": start, "end": end, "text": text})
    else:
        # Fallback to single full-text segment
        text = transcript.text.strip() if hasattr(transcript, "text") else ""
        segments.append({"start": 0.0, "end": 0.0, "text": text})
    language = getattr(transcript, "language", None)
    return {"language": language, "segments": segments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()

    work = ensure_workdirs(args.video_id)
    logger.info(f"Transcribing (OpenAI) {args.audio} → {work.transcripts_dir}")

    data = transcribe_with_openai(args.audio)
    json_path = os.path.join(work.transcripts_dir, "asr_openai.json")
    srt_path = os.path.join(work.transcripts_dir, "asr_openai.srt")
    write_json(json_path, data)
    write_srt(srt_path, data["segments"]) 
    logger.info(f"Saved: {json_path}\nSaved: {srt_path}")


if __name__ == "__main__":
    main() 