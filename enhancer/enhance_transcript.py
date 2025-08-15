import os
import sys
import json
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
from utils.io_utils import read_json, write_json, write_srt

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You will receive a JSON array of segments with fields: start, end, text. "
    "Improve spelling, punctuation, casing, and fix obvious ASR mistakes without changing meaning. "
    "Preserve timing boundaries and keep the same number of segments where possible. "
    "Return a JSON object with a single key 'segments' mapping to the improved array. "
    "Do not include any prose, explanation, or markdown—return JSON only.\n\n"
    "Example input:\n"
    "[\n"
    "  {\"start\": 0.0, \"end\": 1.2, \"text\": \"hello everbody welcome 2 the show\"},\n"
    "  {\"start\": 1.2, \"end\": 2.6, \"text\": \"im your host\"}\n"
    "]\n\n"
    "Example output:\n"
    "{\n"
    "  \"segments\": [\n"
    "    {\"start\": 0.0, \"end\": 1.2, \"text\": \"Hello everybody, welcome to the show.\"},\n"
    "    {\"start\": 1.2, \"end\": 2.6, \"text\": \"I'm your host.\"}\n"
    "  ]\n"
    "}"
)


def enhance_with_openai(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(segments, ensure_ascii=False)},
    ]


    request_kwargs = {
        "model": os.environ.get("LLM_MODEL_NAME"),
        "messages": messages,
        "temperature": 0.0,
    }

    try:
        completion = client.chat.completions.create(
            **request_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning(f"JSON mode failed, retrying without response_format: {e}")
        completion = client.chat.completions.create(
            **request_kwargs,
        )

    if not getattr(completion, "choices", None):
        raise RuntimeError("LLM completion returned no choices")

    content = completion.choices[0].message.content
   
    try:
        data = json.loads(content)
    except Exception:
        # fallback: wrap if assistant returned array directly
        if content.strip().startswith("["):
            data = json.loads(content)
        else:
            # try to coerce into the expected object if it's a bare array
            raise
    # allow either direct list or object with segments
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected LLM response format")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--input-json", required=True, help="path to ASR json with segments")
    parser.add_argument("--backend", default="openai", choices=["openai"])
    args = parser.parse_args()

    work = ensure_workdirs(args.video_id)
    raw = read_json(args.input_json)
    segments = raw.get("segments", [])

    if args.backend == "openai":
        enhanced_segments = enhance_with_openai(segments)
    else:
        enhanced_segments = segments

    enhanced_json = {"segments": enhanced_segments}
    json_path = os.path.join(work.enhanced_dir, "enhanced.json")
    srt_path = os.path.join(work.enhanced_dir, "enhanced.srt")
    write_json(json_path, enhanced_json)
    write_srt(srt_path, enhanced_segments)
    logger.info(f"Saved: {json_path}\nSaved: {srt_path}")


if __name__ == "__main__":
    main() 