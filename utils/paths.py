import os
from dataclasses import dataclass
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()


DEFAULT_OUTPUTS_ROOT = os.environ.get("OUTPUTS_ROOT") or os.path.join(
    os.path.dirname(__file__), "..", "outputs"
)
DEFAULT_OUTPUTS_ROOT = os.path.abspath(DEFAULT_OUTPUTS_ROOT)


@dataclass
class WorkDirs:
    root: str
    video_dir: str
    audio_dir: str
    transcripts_dir: str
    enhanced_dir: str
    subtitled_dir: str


def ensure_workdirs(video_id: str) -> WorkDirs:
    root = os.path.join(DEFAULT_OUTPUTS_ROOT, video_id)
    video_dir = os.path.join(root, "video")
    audio_dir = os.path.join(root, "audio")
    transcripts_dir = os.path.join(root, "transcripts")
    enhanced_dir = os.path.join(root, "enhanced")
    subtitled_dir = os.path.join(root, "subtitled")

    for path in [root, video_dir, audio_dir, transcripts_dir, enhanced_dir, subtitled_dir]:
        os.makedirs(path, exist_ok=True)

    return WorkDirs(
        root=root,
        video_dir=video_dir,
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        enhanced_dir=enhanced_dir,
        subtitled_dir=subtitled_dir,
    )


def safe_filename(name: str) -> str:
    keepchars = (" ", ".", "_", "-")
    return "".join(c for c in name if c.isalnum() or c in keepchars).rstrip()


def split_ext(path: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(path)
    return base, ext 