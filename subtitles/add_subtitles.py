import os
import sys
import argparse

# ensure package path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.paths import ensure_workdirs
from utils.logging_utils import get_logger
from utils.ffmpeg_utils import add_subtitles_soft, burn_subtitles

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--srt", required=True)
    parser.add_argument("--mode", choices=["soft", "burn"], default="soft")
    args = parser.parse_args()

    work = ensure_workdirs(args.video_id)

    if args.mode == "soft":
        out_path = os.path.join(work.subtitled_dir, "with_subtitles_soft.mp4")
        add_subtitles_soft(args.input_video, args.srt, out_path)
    else:
        out_path = os.path.join(work.subtitled_dir, "with_subtitles_burned.mp4")
        burn_subtitles(args.input_video, args.srt, out_path)

    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main() 