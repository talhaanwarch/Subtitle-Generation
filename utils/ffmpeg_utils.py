import os
import shlex
import subprocess
from typing import Optional
from .logging_utils import get_logger

logger = get_logger(__name__)


def run_cmd(command: str) -> None:
    logger.debug(f"Running: {command}")
    process = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        output = process.stdout.decode("utf-8", errors="ignore")
        logger.error(output)
        raise RuntimeError(f"Command failed: {command}")


def extract_audio(
    input_video_path: str,
    output_audio_path: str,
    sample_rate_hz: int = 16000,
    mono: bool = True,
) -> str:
    channels_arg = "-ac 1" if mono else ""
    cmd = (
        f"ffmpeg -y -i {shlex.quote(input_video_path)} -vn -acodec pcm_s16le "
        f"-ar {sample_rate_hz} {channels_arg} {shlex.quote(output_audio_path)}"
    )
    run_cmd(cmd)
    return output_audio_path


def add_subtitles_soft(input_video: str, srt_file: str, output_path: str, language: str = "eng") -> str:
    # mp4 soft subs require mov_text codec
    cmd = (
        f"ffmpeg -y -i {shlex.quote(input_video)} -i {shlex.quote(srt_file)} "
        f"-c copy -c:s mov_text -metadata:s:s:0 language={shlex.quote(language)} "
        f"{shlex.quote(output_path)}"
    )
    run_cmd(cmd)
    return output_path


def burn_subtitles(input_video: str, srt_file: str, output_path: str) -> str:
    vf = f"subtitles={shlex.quote(srt_file)}"
    cmd = (
        f"ffmpeg -y -i {shlex.quote(input_video)} -vf {vf} -c:a copy {shlex.quote(output_path)}"
    )
    run_cmd(cmd)
    return output_path 