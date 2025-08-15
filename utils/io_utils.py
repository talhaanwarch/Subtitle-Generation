import json
from typing import Any, Dict, List
from .timecode import seconds_to_srt_time


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_srt(path: str, segments: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        lines.append(str(idx))
        lines.append(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        lines.append(text)
        lines.append("")
    write_text(path, "\n".join(lines)) 