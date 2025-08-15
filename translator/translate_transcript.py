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

def get_translation_system_prompt(target_lang: str) -> str:
    """Generate system prompt for translation"""
    return (
        f"You will receive a JSON array of subtitle segments with fields: start, end, text. "
        f"Translate the text content to {target_lang} while preserving the exact timing boundaries. "
        f"Keep the same number of segments and maintain natural flow for subtitles. "
        f"Return a JSON object with a single key 'segments' mapping to the translated array. "
        f"Do not include any prose, explanation, or markdown—return JSON only.\n\n"
        f"Example input:\n"
        f"[\n"
        f"  {{\"start\": 0.0, \"end\": 1.2, \"text\": \"Hello everybody, welcome to the show.\"}},\n"
        f"  {{\"start\": 1.2, \"end\": 2.6, \"text\": \"I'm your host.\"}}\n"
        f"]\n\n"
        f"Example output (for Spanish):\n"
        f"{{\n"
        f"  \"segments\": [\n"
        f"    {{\"start\": 0.0, \"end\": 1.2, \"text\": \"Hola a todos, bienvenidos al programa.\"}},\n"
        f"    {{\"start\": 1.2, \"end\": 2.6, \"text\": \"Soy su anfitrión.\"}}\n"
        f"  ]\n"
        f"}}"
    )


def translate_with_openai(segments: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
    """Translate segments using OpenAI-compatible API"""
    from openai import OpenAI

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))
    system_prompt = get_translation_system_prompt(target_lang)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(segments, ensure_ascii=False)},
    ]

    request_kwargs = {
        "model": os.environ.get("LLM_MODEL_NAME"),
        "messages": messages,
        "temperature": 0.1,  # Slightly higher for more natural translations
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
    parser = argparse.ArgumentParser(description="Translate subtitle segments to target language")
    parser.add_argument("--video-id", required=True, help="Video ID for output directory")
    parser.add_argument("--input-json", required=True, help="Path to input JSON with segments")
    parser.add_argument("--target-lang", required=True, help="Target language for translation (e.g., 'Spanish', 'French', 'German')")
    parser.add_argument("--backend", default="openai", choices=["openai"], help="Translation backend")
    args = parser.parse_args()

    work = ensure_workdirs(args.video_id)
    raw = read_json(args.input_json)
    segments = raw.get("segments", [])

    if not segments:
        logger.warning("No segments found in input JSON")
        return

    logger.info(f"Translating {len(segments)} segments to {args.target_lang}")

    if args.backend == "openai":
        translated_segments = translate_with_openai(segments, args.target_lang)
    else:
        translated_segments = segments

    # Create output directory for translations
    translated_dir = os.path.join(work.root, "translated")
    os.makedirs(translated_dir, exist_ok=True)
    
    # Use language code for file naming
    lang_code = args.target_lang.lower().replace(" ", "_")
    translated_json = {"segments": translated_segments, "target_language": args.target_lang}
    json_path = os.path.join(translated_dir, f"translated_{lang_code}.json")
    srt_path = os.path.join(translated_dir, f"translated_{lang_code}.srt")
    
    write_json(json_path, translated_json)
    write_srt(srt_path, translated_segments)
    logger.info(f"Saved: {json_path}\nSaved: {srt_path}")


if __name__ == "__main__":
    main()
