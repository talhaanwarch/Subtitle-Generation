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

from utils.paths import ensure_workdirs, safe_filename, DEFAULT_OUTPUTS_ROOT
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def extract_video_id_from_url(url: str) -> str:
    """Extract video ID from YouTube URL without downloading."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("id")


def check_video_exists(video_id: str) -> bool:
    """Check if video already exists in outputs folder."""
    video_dir = os.path.join(DEFAULT_OUTPUTS_ROOT, video_id)
    if not os.path.exists(video_dir):
        return False
    
    # Check if the video file exists in the video subdirectory
    video_subdir = os.path.join(video_dir, "video")
    if not os.path.exists(video_subdir):
        return False
    
    # Look for the video file (should be named with video_id.mp4)
    video_file = os.path.join(video_subdir, f"{video_id}.mp4")
    return os.path.exists(video_file)


def get_existing_video_info(video_id: str, url: str) -> Dict[str, Any]:
    """Get video info for already downloaded video."""
    video_dir = os.path.join(DEFAULT_OUTPUTS_ROOT, video_id)
    video_subdir = os.path.join(video_dir, "video")
    video_file = os.path.join(video_subdir, f"{video_id}.mp4")
    
    # Try to read metadata if it exists
    metadata_file = os.path.join(video_dir, "metadata.json")
    title = None
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                title = metadata.get("title")
        except:
            pass
    
    # If no title in metadata, extract it from URL
    if not title:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title")
    
    return {"video_id": video_id, "title": title, "video_path": video_file}


def download_youtube(url: str, outputs_root: str) -> Dict[str, Any]:
    # First, extract video ID to check if it already exists
    video_id = extract_video_id_from_url(url)
    
    # Check if video already exists in outputs folder
    if check_video_exists(video_id):
        logger.info(f"Video {video_id} already downloaded, skipping download")
        return get_existing_video_info(video_id, url)
    
    # If video doesn't exist, proceed with download
    logger.info(f"Downloading video {video_id}")
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