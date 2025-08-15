import os
import sys
import json
import argparse
from typing import Dict, Any
from yt_dlp import YoutubeDL

# Ensure imports work when run as a script
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.paths import ensure_workdirs, safe_filename
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def download_youtube(url: str, outputs_root: str) -> Dict[str, Any]:
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(outputs_root, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id")
        title = info.get("title")
        filename = ydl.prepare_filename(info)
        # ensure mp4 extension when merged
        base, _ = os.path.splitext(filename)
        video_path = base + ".mp4"
    return {"video_id": video_id, "title": title, "video_path": video_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    # Temporary root; pipeline will relocate into per-video folders
    out_root = os.path.abspath(os.path.join(BASE_DIR, "tmp_downloads"))
    os.makedirs(out_root, exist_ok=True)

    data = download_youtube(args.url, out_root)
    logger.info(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main() 