import os
import sys
import json
import argparse
import shutil

# ensure package path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
	sys.path.insert(0, BASE_DIR)

from utils.paths import ensure_workdirs
from utils.logging_utils import get_logger
from utils.ffmpeg_utils import extract_audio
from utils.io_utils import write_json, write_srt
from downloader.download_youtube import download_youtube

logger = get_logger(__name__)


def run_pipeline(url: str, asr_backend: str = "local", whisper_model: str = "base", llm_backend: str = "openai", subtitle_mode: str = "soft") -> dict:
	# 1) Download video
	tmp_root = os.path.join(BASE_DIR, "tmp_downloads")
	os.makedirs(tmp_root, exist_ok=True)
	logger.info(f"Downloading video: {url}")
	dl_info = download_youtube(url, tmp_root)
	video_id = dl_info["video_id"]
	title = dl_info.get("title")
	video_path = dl_info["video_path"]
	work = ensure_workdirs(video_id)

	# relocate video into per-video directory
	dst_video_path = os.path.join(work.video_dir, os.path.basename(video_path))
	if os.path.abspath(video_path) != os.path.abspath(dst_video_path):
		shutil.move(video_path, dst_video_path)
		video_path = dst_video_path

	# save metadata
	write_json(os.path.join(work.root, "metadata.json"), {"video_id": video_id, "title": title})

	# 2) Extract audio
	audio_path = os.path.join(work.audio_dir, "audio.wav")
	logger.info("Extracting audio")
	extract_audio(video_path, audio_path, sample_rate_hz=16000, mono=True)

	# 3) Transcribe
	if asr_backend == "local":
		from transcriber.transcribe_local import transcribe_with_whisper
		asr_json = transcribe_with_whisper(audio_path, whisper_model)
		asr_json_path = os.path.join(work.transcripts_dir, "asr_local.json")
		write_json(asr_json_path, asr_json)
		write_srt(os.path.join(work.transcripts_dir, "asr_local.srt"), asr_json["segments"])
	else:
		from transcriber.transcribe_openai import transcribe_with_openai
		asr_json = transcribe_with_openai(audio_path)
		asr_json_path = os.path.join(work.transcripts_dir, "asr_openai.json")
		write_json(asr_json_path, asr_json)
		write_srt(os.path.join(work.transcripts_dir, "asr_openai.srt"), asr_json["segments"])

	# 4) Enhance transcript
	logger.info("Enhancing transcript with LLM")
	from enhancer.enhance_transcript import enhance_with_openai
	if llm_backend == "openai":
		enhanced_segments = enhance_with_openai(asr_json["segments"]) 
	else:
		enhanced_segments = asr_json["segments"]
	enhanced_json_path = os.path.join(work.enhanced_dir, "enhanced.json")
	write_json(enhanced_json_path, {"segments": enhanced_segments})
	enhanced_srt_path = os.path.join(work.enhanced_dir, "enhanced.srt")
	write_srt(enhanced_srt_path, enhanced_segments)

	# 5) Add subtitles
	logger.info(f"Adding subtitles ({subtitle_mode})")
	from utils.ffmpeg_utils import add_subtitles_soft, burn_subtitles
	if subtitle_mode == "soft":
		final_video = os.path.join(work.subtitled_dir, "with_subtitles_soft.mp4")
		add_subtitles_soft(video_path, enhanced_srt_path, final_video)
	else:
		final_video = os.path.join(work.subtitled_dir, "with_subtitles_burned.mp4")
		burn_subtitles(video_path, enhanced_srt_path, final_video)

	return {
		"video_id": video_id,
		"video": video_path,
		"audio": audio_path,
		"asr_json": asr_json_path,
		"enhanced_json": enhanced_json_path,
		"enhanced_srt": enhanced_srt_path,
		"final_video": final_video,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="End-to-end YouTube → Transcript → Subtitles pipeline")
	parser.add_argument("--url", required=True, help="YouTube URL")
	parser.add_argument("--asr_backend", choices=["local", "openai"], default="local")
	parser.add_argument("--whisper_model", default=os.environ.get("DEFAULT_WHISPER_MODEL", "base"))
	parser.add_argument("--llm_backend", choices=["openai"], default="openai")
	parser.add_argument("--subtitle_mode", choices=["soft", "burn"], default="soft")
	args = parser.parse_args()

	result = run_pipeline(
		url=args.url,
		asr_backend=args.asr_backend,
		whisper_model=args.whisper_model,
		llm_backend=args.llm_backend,
		subtitle_mode=args.subtitle_mode,
	)
	logger.info(json.dumps(result, indent=2))


if __name__ == "__main__":
	main() 