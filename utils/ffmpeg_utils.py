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


def burn_subtitles(input_video: str, srt_file: str, output_path: str, box_opacity: float = 0.6) -> str:
    """
    Burn subtitles into video with a semi-transparent background overlay.
    
    Args:
        input_video: Path to input video file
        srt_file: Path to SRT subtitle file
        output_path: Path for output video file
        box_opacity: Opacity of the background box (0.0 to 1.0, default 0.6)
    
    Returns:
        Path to the output video file
    """
    import tempfile
    import os as path_os
    
    # Create a temporary ASS file with proper background styling
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8') as temp_ass:
        temp_ass_path = temp_ass.name
        
        # Read the SRT file
        with open(srt_file, 'r', encoding='utf-8') as srt:
            srt_content = srt.read()
        
        # Calculate alpha value for background (0-255, where 0=transparent, 255=opaque)
        alpha_value = int(box_opacity * 255)
        alpha_hex = f"{alpha_value:02X}"
        
        # Write ASS header with proper styling for unified background
        # Using BorderStyle=4 with proper settings for background box
        temp_ass.write(f"""[Script Info]
Title: Enhanced Subtitles
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&Hffffff,&Hffffff,&H00000000,&H{alpha_hex}000000,0,0,0,0,100,100,0,0,4,2,0,2,10,10,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text

""")
        
        # Convert SRT content to ASS format
        from utils.timecode import srt_time_to_ass_time
        
        lines = srt_content.strip().split('\n\n')
        for block in lines:
            if not block.strip():
                continue
            
            parts = block.split('\n')
            if len(parts) >= 3:
                # Skip subtitle number
                time_line = parts[1]
                text_lines = parts[2:]
                
                # Parse time codes
                if ' --> ' in time_line:
                    start_time, end_time = time_line.split(' --> ')
                    
                    # Convert SRT time format to ASS time format
                    start_ass = srt_time_to_ass_time(start_time)
                    end_ass = srt_time_to_ass_time(end_time)
                    
                    # Join all text lines with ASS line break
                    text = '\\N'.join(text_lines)
                    # Clean up any HTML-like tags
                    text = text.replace('<', '').replace('>', '')
                    
                    # Add background box using ASS tags
                    # {\1a&H00&} sets background to fully opaque
                    # {\3a&H00&} sets outline to fully opaque
                    # {\4a&H[alpha]&} sets shadow/background alpha
                    text_with_bg = f"{{\\1a&H00&\\3a&H00&\\4a&H{alpha_hex}&}}{text}"
                    
                    # Write ASS dialogue line
                    temp_ass.write(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text_with_bg}\n")
    
    try:
        # Use the ASS file with subtitles filter
        vf = f"ass={shlex.quote(temp_ass_path)}"
        cmd = (
            f"ffmpeg -y -i {shlex.quote(input_video)} -vf {vf} -c:a copy {shlex.quote(output_path)}"
        )
        run_cmd(cmd)
        return output_path
    finally:
        # Clean up temporary file
        if path_os.path.exists(temp_ass_path):
            path_os.unlink(temp_ass_path) 