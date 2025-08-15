import os
import sys
import json
import argparse
from typing import Dict, Any, List

# ensure package path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.paths import ensure_workdirs
from utils.logging_utils import get_logger
from utils.io_utils import write_json, write_srt
from utils.config import Config

logger = get_logger(__name__)


def transcribe_with_whisper(audio_path: str, config:Config) -> Dict[str, Any]:
    # Use faster-whisper on CPU for local transcription
    from faster_whisper import WhisperModel
    model_name=config.asr.whisper_model
    language=config.video.input_language
    model = WhisperModel(model_name, device="auto", compute_type="float16")

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=False,
    )

    segments: List[Dict[str, Any]] = []
    for seg in segments_iter:
        segments.append(
            {
                "start": float(seg.start or 0.0),
                "end": float(seg.end or (seg.start or 0.0)),
                "text": (seg.text or "").strip(),
            }
        )

    language = getattr(info, "language", None)
    return {"language": language, "segments": segments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--video-id", required=True)
   
    args = parser.parse_args()

    work = ensure_workdirs(args.video_id)
    logger.info(f"Transcribing (local faster-whisper CPU) {args.audio} → {work.transcripts_dir}")

    from utils.config import load_config
    config = load_config()

    data = transcribe_with_whisper(args.audio, config)

    json_path = os.path.join(work.transcripts_dir, "asr_local.json")
    srt_path = os.path.join(work.transcripts_dir, "asr_local.srt")
    write_json(json_path, data)
    write_srt(srt_path, data["segments"]) 
    logger.info(f"Saved: {json_path}\nSaved: {srt_path}")


if __name__ == "__main__":
    main() 

##python transcriber/transcribe_local.py --audio outputs/3PI8ugmLxcc/audio/audio.wav --video-id 3PI8ugmLxcc