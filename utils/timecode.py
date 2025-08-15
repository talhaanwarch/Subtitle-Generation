from typing import Tuple


def seconds_to_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3600000
    remainder = total_milliseconds % 3600000
    minutes = remainder // 60000
    remainder = remainder % 60000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def srt_time_to_seconds(time_str: str) -> float:
    # HH:MM:SS,mmm
    hh, mm, rest = time_str.split(":", 2)
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def srt_time_to_ass_time(srt_time: str) -> str:
    """Convert SRT time format (HH:MM:SS,mmm) to ASS time format (H:MM:SS.cc)"""
    # Remove any whitespace
    srt_time = srt_time.strip()
    
    # Parse SRT format: HH:MM:SS,mmm
    hh, mm, rest = srt_time.split(":", 2)
    ss, ms = rest.split(",")
    
    # Convert to ASS format: H:MM:SS.cc (centiseconds, not milliseconds)
    hours = int(hh)
    minutes = int(mm)
    seconds = int(ss)
    centiseconds = int(ms) // 10  # Convert milliseconds to centiseconds
    
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}" 