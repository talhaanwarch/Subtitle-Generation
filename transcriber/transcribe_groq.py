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
from utils.config import Config

logger = get_logger(__name__)


def transcribe_with_groq(audio_path: str, config: Config) -> Dict[str, Any]:
    from openai import OpenAI

    # Use the config for API key and model
    api_key = config.api.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key is required. Set it in config or GROQ_API_KEY environment variable.")
    
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    logger.info("language groq whisper",config.video.input_language)
    with open(audio_path, "rb") as f:
        # Attempt to use verbose_json to get segments when supported
        transcript = client.audio.transcriptions.create(
            model=config.asr.groq_model,
            file=f,
            response_format="verbose_json",
            temperature=0.1,
            language=config.video.input_language
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
    logger.info(f"Transcribing (Groq) {args.audio} → {work.transcripts_dir}")

    # Load the actual configuration from config.yaml
    from utils.config import load_config
    config = load_config()
    
  
    if os.environ.get("GROQ_API_KEY"):
        config.api.groq_api_key = os.environ.get("GROQ_API_KEY")

    data = transcribe_with_groq(args.audio, config)
    json_path = os.path.join(work.transcripts_dir, "asr_groq.json")
    srt_path = os.path.join(work.transcripts_dir, "asr_groq.srt")
    write_json(json_path, data)
    write_srt(srt_path, data["segments"]) 
    logger.info(f"Saved: {json_path}\nSaved: {srt_path}")


if __name__ == "__main__":
    main() 

#python transcriber/transcribe_groq.py --audio outputs/3PI8ugmLxcc/audio/audio.wav --video-id 3PI8ugmLxcc