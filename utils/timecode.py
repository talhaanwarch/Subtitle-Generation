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