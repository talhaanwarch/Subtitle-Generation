import os
import sys
import json
import argparse
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.config import Config
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
    "Your task is to improve the transcript by considering the full context across all segments.\n\n"
    "ENHANCEMENT GUIDELINES:\n"
    "1. CONTEXT AWARENESS: Read through all segments first to understand the overall topic, speaker style, and content flow before making changes\n"
    "2. CONSISTENCY: Maintain consistent terminology, names, and technical terms throughout the entire transcript\n"
    "3. COHERENCE: Ensure sentences flow naturally across segment boundaries, fixing awkward breaks while preserving timing\n"
    "4. ACCURACY: Fix spelling, grammar, punctuation, and obvious ASR errors (like 'hear' vs 'here', number recognition)\n"
    "5. FORMATTING: Apply proper capitalization, punctuation, and sentence structure\n"
    "6. PRESERVATION: Keep the same number of segments with identical timing boundaries (start/end values)\n"
    "7. MEANING: Never change the speaker's intended meaning or add information not present in the audio\n"
    "8. CONFIDENCE: Only fix wrong word transcriptions when you're highly confident based on context - when in doubt, preserve the original\n\n"
    "SPECIFIC IMPROVEMENTS:\n"
    "- Fix common ASR mistakes: 'their/there/they're', 'to/too/two', 'your/you're', etc.\n"
    "- Correct homophones based on context (e.g., 'right' vs 'write', 'no' vs 'know')\n"
    "- Fix obviously wrong word transcriptions when you're fully confident based on context\n"
    "- Replace nonsensical words or phrases with contextually appropriate alternatives\n"
    "- Correct misheard technical terms, proper nouns, and domain-specific vocabulary\n"
    "- Fix grammatically impossible constructions (e.g., 'I are going' → 'I am going')\n"
    "- Resolve ambiguous transcriptions using surrounding context for clarity\n"
    "- Fix run-on sentences or fragment sentences appropriately\n"
    "- Ensure proper names and technical terms are spelled consistently\n"
    "- Add appropriate punctuation for clarity\n"
    "- Maintain speaker's natural speech patterns and style\n\n"
    "URDU SCRIPT CORRECTIONS:\n"
    "- Fix obvious Urdu script errors and character mistakes\n"
    "- Correct common Urdu phrases that were mistranscribed:\n"
    "  • 'فگر مک کرنا' → 'فکر مت کرنا' (don't worry)\n"
    "  • 'ليکن' → 'لیکن' (but - correct ya character)\n"
    "  • 'هے' → 'ہے' (is - correct heh character)\n"
    "- Fix incorrect Arabic ya (ي) with correct Urdu ya (ی)\n"
    "- Fix incorrect Arabic heh (ه) with correct Urdu heh (ہ)\n"
    "- Ensure proper Urdu diacritics and character forms\n"
    "- Maintain consistency in Urdu script throughout\n"
    "- Fix grammatically incorrect Urdu constructions\n\n"
    "Return a JSON object with a single key 'segments' mapping to the improved array. "
    "Do not include any prose, explanation, or markdown—return JSON only.\n\n"
    "Example input:\n"
    "[\n"
    '  {"start": 0.0, "end": 1.2, "text": "hello everbody welcome 2 the show"},\n'
    '  {"start": 1.2, "end": 2.6, "text": "im your host john and today"},\n'
    '  {"start": 2.6, "end": 4.0, "text": "where talking about machine learning"},\n'
    '  {"start": 4.0, "end": 5.5, "text": "its a facinating feel that has many applications"}\n'
    "]\n\n"
    "Example output:\n"
    "{\n"
    '  "segments": [\n'
    '    {"start": 0.0, "end": 1.2, "text": "Hello everybody, welcome to the show."},\n'
    '    {"start": 1.2, "end": 2.6, "text": "I\'m your host John, and today"},\n'
    '    {"start": 2.6, "end": 4.0, "text": "we\'re talking about machine learning."},\n'
    '    {"start": 4.0, "end": 5.5, "text": "It\'s a fascinating field that has many applications."}\n'
    "  ]\n"
    "}"
)


def enhance_with_groq(
    segments: List[Dict[str, Any]], config: "Config"
) -> List[Dict[str, Any]]:
    """Enhanced transcript segments using Groq API."""
    from openai import OpenAI

    # Use the config for API key and model
    api_key = config.api.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Groq API key is required. Set it in config or GROQ_API_KEY environment variable."
        )

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(segments, ensure_ascii=False)},
    ]

    request_kwargs = {
        "model": config.llm.model,
        "messages": messages,
        "temperature": config.llm.enhancer.temperature,
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


# Backward compatibility alias
def enhance_with_openai(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Backward compatibility wrapper for enhance_with_groq."""
    from utils.config import Config

    config = Config()
    config.llm.model = os.environ.get("LLM_MODEL_NAME", "llama3-8b-8192")
    config.api.groq_api_key = os.environ.get("GROQ_API_KEY", "")
    return enhance_with_groq(segments, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--input-json", required=True, help="path to ASR json with segments"
    )
    parser.add_argument("--backend", default="groq", choices=["openai", "groq"])
    args = parser.parse_args()

    work = ensure_workdirs(args.video_id)
    raw = read_json(args.input_json)
    segments = raw.get("segments", [])

    if args.backend in ["openai", "groq"]:
        enhanced_segments = enhance_with_openai(
            segments
        )  # Uses backward compatibility wrapper
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
