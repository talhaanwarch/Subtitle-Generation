import os
import sys
import json
import argparse
import shutil
from typing import Dict, Any

# ensure package path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.paths import ensure_workdirs
from utils.logging_utils import get_logger
from utils.ffmpeg_utils import extract_audio
from utils.io_utils import write_json, write_srt
from utils.config import Config, load_config
from downloader.download_youtube import download_youtube

logger = get_logger(__name__)


def run_pipeline_with_config(config: Config) -> Dict[str, Any]:
    """
    Run the complete pipeline using configuration object.

    Args:
        config: Configuration object containing all pipeline settings.

    Returns:
        Dictionary with paths to all generated files and metadata.
    """
    url = config.video.url

    # 1) Download video (or use existing if already downloaded)
    tmp_root = os.path.join(BASE_DIR, config.video.tmp_downloads_dir)
    os.makedirs(tmp_root, exist_ok=True)
    logger.info(f"Checking/downloading video: {url}")
    dl_info = download_youtube(url, tmp_root)
    video_id = dl_info["video_id"]
    title = dl_info.get("title")
    video_path = dl_info["video_path"]
    work = ensure_workdirs(video_id)

    # Check if video is already in the correct location
    dst_video_path = os.path.join(work.video_dir, os.path.basename(video_path))
    if os.path.abspath(video_path) != os.path.abspath(dst_video_path):
        # Video is in tmp_downloads, need to move it to the video directory
        if os.path.exists(video_path):
            shutil.move(video_path, dst_video_path)
        video_path = dst_video_path
    else:
        # Video is already in the correct location (existing video)
        logger.info(f"Using existing video at: {video_path}")

    # save/update metadata
    metadata_path = os.path.join(work.root, "metadata.json")
    if not os.path.exists(metadata_path) or title:
        write_json(metadata_path, {"video_id": video_id, "title": title})

    # 2) Extract audio
    audio_path = os.path.join(work.audio_dir, "audio.wav")
    logger.info("=====Extracting audio=====")
    logger.info(f"INPUT: {video_path}")
    logger.info(f"OUTPUT: {audio_path}")
    extract_audio(
        video_path,
        audio_path,
        sample_rate_hz=config.processing.audio.sample_rate,
        mono=config.processing.audio.mono,
    )

    # 2.5) Audio separation (optional)
    transcription_audio_path = audio_path  # Default to original audio
    separation_results = {}

    if config.processing.audio.separation.enabled:
        logger.info("=====Separating audio (vocals from music)=====")
        logger.info(f"INPUT: {audio_path}")
        logger.info(f"OUTPUT DIR: {work.separated_dir}")
        try:
            from separator.separate_audio import separate_audio_file

            separation_output_dir = work.separated_dir
            separation_results = separate_audio_file(
                audio_path=audio_path,
                output_dir=separation_output_dir,
                model_name=config.processing.audio.separation.model,
                output_format=config.processing.audio.separation.output_format,
                sample_rate=config.processing.audio.sample_rate,
                auto_select_best=config.processing.audio.separation.auto_select_best,
                stem_type=config.processing.audio.separation.stem_type,
            )

            # Use separated vocals for transcription if configured
            if (
                config.processing.audio.separation.enabled
                and "vocals" in separation_results
            ):
                transcription_audio_path = separation_results["vocals"]
                logger.info(
                    f"SUCCESS: Separated vocals saved to {transcription_audio_path}"
                )
                logger.info(
                    f"ASR WILL USE: {transcription_audio_path} (separated vocals)"
                )
            else:
                logger.info("No vocals stem found in separation results")
                logger.info(f"ASR WILL USE: {audio_path} (original audio)")

        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            logger.warning("Continuing with original audio for transcription")
            logger.info(f"ASR WILL USE: {audio_path} (original audio)")
    else:
        logger.info("=====Audio separation disabled=====")
        logger.info(f"ASR WILL USE: {audio_path} (original audio)")

    # 3) Transcribe
    logger.info(f"=====Transcribing audio with {config.asr.backend} backend=====")
    logger.info(f"INPUT: {transcription_audio_path}")
    if config.asr.backend == "local":
        from transcriber.transcribe_local import transcribe_with_whisper

        asr_json = transcribe_with_whisper(transcription_audio_path, config)
        asr_json_path = os.path.join(work.transcripts_dir, "asr_local.json")
        asr_srt_path = os.path.join(work.transcripts_dir, "asr_local.srt")
        write_json(asr_json_path, asr_json)
        write_srt(asr_srt_path, asr_json["segments"])
        logger.info(f"OUTPUT: {asr_json_path}")
        logger.info(f"OUTPUT: {asr_srt_path}")
    elif config.asr.backend == "groq":
        from transcriber.transcribe_groq import transcribe_with_groq

        asr_json = transcribe_with_groq(transcription_audio_path, config)
        asr_json_path = os.path.join(work.transcripts_dir, "asr_groq.json")
        asr_srt_path = os.path.join(work.transcripts_dir, "asr_groq.srt")
        write_json(asr_json_path, asr_json)
        write_srt(asr_srt_path, asr_json["segments"])
        logger.info(f"OUTPUT: {asr_json_path}")
        logger.info(f"OUTPUT: {asr_srt_path}")
    else:
        raise ValueError(f"Unknown ASR backend: {config.asr.backend}")

    # 4) Enhance transcript
    if config.llm.enhancer.enabled:
        logger.info("=====Enhancing transcript with LLM=====")
        logger.info(f"INPUT: ASR segments ({len(asr_json['segments'])} segments)")
        from enhancer.enhance_transcript import enhance_with_groq

        if config.llm.backend == "groq":
            enhanced_segments = enhance_with_groq(asr_json["segments"], config)
            logger.info(f"SUCCESS: Enhanced {len(enhanced_segments)} segments")
        else:
            logger.warning(
                f"Unknown LLM backend: {config.llm.backend}, skipping enhancement"
            )
            enhanced_segments = asr_json["segments"]
            logger.info("NEXT STEP WILL USE: ASR segments (unenhanced)")

        # Save enhanced segments
        enhanced_json_path = os.path.join(work.enhanced_dir, "enhanced.json")
        write_json(enhanced_json_path, {"segments": enhanced_segments})
        enhanced_srt_path = os.path.join(work.enhanced_dir, "enhanced.srt")
        write_srt(enhanced_srt_path, enhanced_segments)
        logger.info(f"OUTPUT: {enhanced_json_path}")
        logger.info(f"OUTPUT: {enhanced_srt_path}")

        # Use enhanced segments for next step
        segments_for_next_step = enhanced_segments
        srt_for_next_step = enhanced_srt_path
        logger.info("NEXT STEP WILL USE: Enhanced segments")
    else:
        logger.info("=====Enhancement disabled=====")
        # Use ASR segments directly, no enhancement processing
        enhanced_segments = asr_json["segments"]
        segments_for_next_step = asr_json["segments"]
        srt_for_next_step = asr_srt_path  # Use original ASR SRT
        # Initialize enhancement paths as None since enhancement is disabled
        enhanced_json_path = None
        enhanced_srt_path = None
        logger.info("NEXT STEP WILL USE: ASR segments (unenhanced)")

    # 5) Translate transcript (optional)
    final_segments = segments_for_next_step
    final_srt_path = srt_for_next_step
    translated_srt_path = None
    translated_json_path = None

    # Validate translation configuration
    translation_enabled = config.llm.translator.enabled
    target_language = config.llm.translator.target_language

    if translation_enabled and not target_language:
        logger.warning(
            "Translation is enabled but no target language is specified. Skipping translation."
        )
        input_type = "Enhanced" if config.llm.enhancer.enabled else "ASR"
        logger.info(f"SUBTITLES WILL USE: {input_type} segments")
    elif not translation_enabled and target_language:
        logger.warning(
            "Target language is specified but translation is disabled. Skipping translation."
        )
        input_type = "Enhanced" if config.llm.enhancer.enabled else "ASR"
        logger.info(f"SUBTITLES WILL USE: {input_type} segments")
    elif translation_enabled and target_language:
        logger.info(f"=====Translating transcript to {target_language}=====")
        input_type = "Enhanced" if config.llm.enhancer.enabled else "ASR"
        logger.info(
            f"INPUT: {input_type} segments ({len(segments_for_next_step)} segments)"
        )
        from translator.translate_transcript import translate_with_groq

        if config.llm.backend == "groq":
            translated_segments = translate_with_groq(
                segments_for_next_step, target_language, config
            )
            logger.info(
                f"SUCCESS: Translated {len(translated_segments)} segments to {target_language}"
            )
        else:
            logger.warning(
                f"Unknown LLM backend: {config.llm.backend}, skipping translation"
            )
            translated_segments = segments_for_next_step

        # Use language code for file naming
        lang_code = target_language.lower().replace(" ", "_")
        translated_json_path = os.path.join(
            work.translated_dir, f"translated_{lang_code}.json"
        )
        translated_srt_path = os.path.join(
            work.translated_dir, f"translated_{lang_code}.srt"
        )
        write_json(
            translated_json_path,
            {"segments": translated_segments, "target_language": target_language},
        )
        write_srt(translated_srt_path, translated_segments)
        logger.info(f"OUTPUT: {translated_json_path}")
        logger.info(f"OUTPUT: {translated_srt_path}")

        # Use translated subtitles for final video
        final_segments = translated_segments
        final_srt_path = translated_srt_path
        logger.info(f"SUBTITLES WILL USE: Translated segments ({target_language})")
    else:
        logger.info("=====Translation disabled=====")
        input_type = "Enhanced" if config.llm.enhancer.enabled else "ASR"
        logger.info(f"SUBTITLES WILL USE: {input_type} segments")

    # 6) Add subtitles
    logger.info(f"=====Adding subtitles ({config.subtitles.mode})=====")
    logger.info(f"INPUT VIDEO: {video_path}")
    logger.info(f"INPUT SUBTITLES: {final_srt_path}")
    from utils.ffmpeg_utils import add_subtitles_soft, burn_subtitles

    if config.subtitles.mode == "soft":
        final_video = os.path.join(work.subtitled_dir, "with_subtitles_soft.mp4")
        add_subtitles_soft(video_path, final_srt_path, final_video)
        logger.info(f"OUTPUT: {final_video}")
    else:
        final_video = os.path.join(work.subtitled_dir, "with_subtitles_burned.mp4")
        burn_subtitles(
            video_path, final_srt_path, final_video, config.subtitles.box_opacity
        )
        logger.info(f"OUTPUT: {final_video}")

    result = {
        "video_id": video_id,
        "video": video_path,
        "audio": audio_path,
        "transcription_audio": transcription_audio_path,
        "asr_json": asr_json_path,
        "enhanced_json": enhanced_json_path,
        "enhanced_srt": enhanced_srt_path,
        "final_video": final_video,
    }

    # Add separation info if separation was performed
    if config.processing.audio.separation.enabled and separation_results:
        result.update(
            {
                "separation_results": separation_results,
                "separation_output_dir": work.separated_dir,
            }
        )

    # Add translation info if translation was performed
    if translation_enabled and target_language:
        result.update(
            {
                "translated_json": translated_json_path,
                "translated_srt": translated_srt_path,
                "target_language": target_language,
            }
        )

    return result


def run_pipeline(
    url: str,
    asr_backend: str = "local",
    whisper_model: str = "base",
    llm_backend: str = "groq",
    subtitle_mode: str = "soft",
    target_lang: str = None,
    box_opacity: float = 0.6,
    input_language: str = None,
) -> dict:
    # 1) Download video (or use existing if already downloaded)
    logger.info("1. Fetching video.....")
    tmp_root = os.path.join(BASE_DIR, "tmp_downloads")
    os.makedirs(tmp_root, exist_ok=True)
    logger.info(f"Checking/downloading video: {url}")
    dl_info = download_youtube(url, tmp_root)
    video_id = dl_info["video_id"]
    title = dl_info.get("title")
    video_path = dl_info["video_path"]
    work = ensure_workdirs(video_id)

    # Check if video is already in the correct location
    dst_video_path = os.path.join(work.video_dir, os.path.basename(video_path))
    if os.path.abspath(video_path) != os.path.abspath(dst_video_path):
        # Video is in tmp_downloads, need to move it to the video directory
        if os.path.exists(video_path):
            shutil.move(video_path, dst_video_path)
        video_path = dst_video_path
    else:
        # Video is already in the correct location (existing video)
        logger.info(f"Using existing video at: {video_path}")

    # save/update metadata
    metadata_path = os.path.join(work.root, "metadata.json")
    if not os.path.exists(metadata_path) or title:
        write_json(metadata_path, {"video_id": video_id, "title": title})

    # 2) Extract audio
    logger.info("2. Extracting audio....")
    audio_path = os.path.join(work.audio_dir, "audio.wav")
    extract_audio(video_path, audio_path, sample_rate_hz=16000, mono=True)

    # 2.5) Audio separation (optional) - using default settings for legacy mode
    transcription_audio_path = audio_path  # Default to original audio
    separation_results = {}

    # Check if audio separation is enabled via environment variable
    separation_enabled = (
        os.environ.get("ENABLE_AUDIO_SEPARATION", "false").lower() == "true"
    )
    if separation_enabled:
        logger.info("2.5. Separating audio (vocals from music)....")
        try:
            from separator.separate_audio import (
                separate_audio_file,
                check_audio_separator_availability,
            )

            if not check_audio_separator_availability():
                logger.warning(
                    "Audio separator not available. Install with: pip install audio-separator[cpu] or audio-separator[gpu]"
                )
                logger.warning("Continuing with original audio for transcription")
            else:
                # Perform audio separation with default settings
                separation_output_dir = work.separated_dir
                separation_results = separate_audio_file(
                    audio_path=audio_path,
                    output_dir=separation_output_dir,
                    model_name="Roformer Model: BS-Roformer-Viperx-1297",
                    output_format="WAV",
                    sample_rate=16000,
                )

                # Use separated vocals for transcription
                if "vocals" in separation_results:
                    transcription_audio_path = separation_results["vocals"]
                    logger.info(
                        f"Using separated vocals for transcription: {transcription_audio_path}"
                    )
                else:
                    logger.info("Using original audio for transcription")

        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            logger.warning("Continuing with original audio for transcription")
    else:
        logger.info("Audio separation disabled, using original audio for transcription")

    # 3) Transcribe
    if asr_backend == "local":
        from transcriber.transcribe_local import transcribe_with_whisper

        asr_json = transcribe_with_whisper(
            transcription_audio_path, whisper_model, input_language
        )
        asr_json_path = os.path.join(work.transcripts_dir, "asr_local.json")
        write_json(asr_json_path, asr_json)
        write_srt(
            os.path.join(work.transcripts_dir, "asr_local.srt"), asr_json["segments"]
        )
    else:
        from transcriber.transcribe_groq import transcribe_with_groq

        # Create a minimal config for backward compatibility
        from utils.config import Config, ASRConfig, APIConfig

        temp_config = Config()
        temp_config.asr.groq_model = os.environ.get(
            "ASR_MODEL_NAME", "distil-whisper-large-v3-en"
        )
        temp_config.api.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        asr_json = transcribe_with_groq(transcription_audio_path, temp_config)
        asr_json_path = os.path.join(work.transcripts_dir, "asr_groq.json")
        write_json(asr_json_path, asr_json)
        write_srt(
            os.path.join(work.transcripts_dir, "asr_groq.srt"), asr_json["segments"]
        )

    # 4) Enhance transcript
    logger.info("3. Enhancing transcript with LLM")
    from enhancer.enhance_transcript import enhance_with_groq

    if llm_backend == "groq":
        # Create a minimal config for backward compatibility
        temp_config.llm.model = os.environ.get("LLM_MODEL_NAME", "llama3-8b-8192")
        enhanced_segments = enhance_with_groq(asr_json["segments"], temp_config)
    else:
        enhanced_segments = asr_json["segments"]
    enhanced_json_path = os.path.join(work.enhanced_dir, "enhanced.json")
    write_json(enhanced_json_path, {"segments": enhanced_segments})
    enhanced_srt_path = os.path.join(work.enhanced_dir, "enhanced.srt")
    write_srt(enhanced_srt_path, enhanced_segments)

    # 5) Translate transcript (optional)
    final_segments = enhanced_segments
    final_srt_path = enhanced_srt_path
    translated_srt_path = None
    translated_json_path = None

    if target_lang:
        logger.info(f"4. Translating transcript to {target_lang}")
        from translator.translate_transcript import translate_with_groq

        if llm_backend == "groq":
            translated_segments = translate_with_groq(
                enhanced_segments, target_lang, temp_config
            )
        else:
            translated_segments = enhanced_segments

        # Use language code for file naming
        lang_code = target_lang.lower().replace(" ", "_")
        translated_json_path = os.path.join(
            work.translated_dir, f"translated_{lang_code}.json"
        )
        translated_srt_path = os.path.join(
            work.translated_dir, f"translated_{lang_code}.srt"
        )
        write_json(
            translated_json_path,
            {"segments": translated_segments, "target_language": target_lang},
        )
        write_srt(translated_srt_path, translated_segments)

        # Use translated subtitles for final video
        final_segments = translated_segments
        final_srt_path = translated_srt_path

    # 6) Add subtitles
    logger.info(f"5. Adding subtitles to video ({subtitle_mode})")
    from utils.ffmpeg_utils import add_subtitles_soft, burn_subtitles

    if subtitle_mode == "soft":
        final_video = os.path.join(work.subtitled_dir, "with_subtitles_soft.mp4")
        add_subtitles_soft(video_path, final_srt_path, final_video)
    else:
        final_video = os.path.join(work.subtitled_dir, "with_subtitles_burned.mp4")
        burn_subtitles(video_path, final_srt_path, final_video, box_opacity)

    result = {
        "video_id": video_id,
        "video": video_path,
        "audio": audio_path,
        "transcription_audio": transcription_audio_path,
        "asr_json": asr_json_path,
        "enhanced_json": enhanced_json_path,
        "enhanced_srt": enhanced_srt_path,
        "final_video": final_video,
    }

    # Add separation info if separation was performed
    if separation_enabled and separation_results:
        result.update(
            {
                "separation_results": separation_results,
                "separation_output_dir": work.separated_dir,
            }
        )

    # Add translation info if translation was performed
    if target_lang:
        result.update(
            {
                "translated_json": translated_json_path,
                "translated_srt": translated_srt_path,
                "target_language": target_lang,
            }
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end YouTube → Transcript → Subtitles pipeline"
    )
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--url", help="YouTube URL (overrides config)")
    parser.add_argument(
        "--asr-backend",
        choices=["local", "groq"],
        help="ASR backend (overrides config)",
    )
    parser.add_argument(
        "--whisper-model", help="Whisper model for local ASR (overrides config)"
    )
    parser.add_argument(
        "--llm-backend", choices=["groq"], help="LLM backend (overrides config)"
    )
    parser.add_argument(
        "--subtitle-mode",
        choices=["soft", "burn"],
        help="Subtitle mode (overrides config)",
    )
    parser.add_argument(
        "--target-lang", help="Target language for translation (overrides config)"
    )
    parser.add_argument(
        "--input-lang", help="Input language for ASR (overrides config)"
    )
    parser.add_argument(
        "--box-opacity",
        type=float,
        help="Opacity of subtitle background box (overrides config)",
    )
    args = parser.parse_args()

    # Load configuration if provided, otherwise create default
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            sys.exit(1)

        # Override with command line arguments
        if args.url:
            config.video.url = args.url
        if getattr(args, "asr_backend"):
            config.asr.backend = getattr(args, "asr_backend")
        if getattr(args, "whisper_model"):
            config.asr.whisper_model = getattr(args, "whisper_model")
        if getattr(args, "llm_backend"):
            config.llm.backend = getattr(args, "llm_backend")
        if getattr(args, "subtitle_mode"):
            config.subtitles.mode = getattr(args, "subtitle_mode")
        if getattr(args, "target_lang"):
            config.llm.translator.target_language = getattr(args, "target_lang")
            config.llm.translator.enabled = True
        if getattr(args, "input_lang"):
            config.video.input_language = getattr(args, "input_lang")
        if getattr(args, "box_opacity") is not None:
            config.subtitles.box_opacity = getattr(args, "box_opacity")

        if not config.video.url:
            logger.error("YouTube URL is required")
            sys.exit(1)

        result = run_pipeline_with_config(config)
    else:
        # Legacy mode: use old function signature
        if not args.url:
            logger.error("YouTube URL is required")
            sys.exit(1)

        result = run_pipeline(
            url=args.url,
            asr_backend=getattr(args, "asr_backend", "local"),
            whisper_model=getattr(
                args, "whisper_model", os.environ.get("DEFAULT_WHISPER_MODEL", "base")
            ),
            llm_backend=getattr(args, "llm_backend", "groq"),
            subtitle_mode=getattr(args, "subtitle_mode", "soft"),
            target_lang=getattr(args, "target_lang"),
            box_opacity=getattr(args, "box_opacity", 0.6),
            input_language=getattr(args, "input_lang"),
        )

    logger.info(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
