# Subtitles

Generate, soft-embed, or burn-in subtitles using SRT files.

## Usage

Soft-embed subtitles (no re-encoding):

```bash
python subtitles/add_subtitles.py \
  --video-id ABC123 \
  --input-video outputs/ABC123/input.mp4 \
  --srt outputs/ABC123/enhanced/enhanced.srt \
  --mode soft
```

Burn-in subtitles (re-encode video):

```bash
python subtitles/add_subtitles.py \
  --video-id ABC123 \
  --input-video outputs/ABC123/input.mp4 \
  --srt outputs/ABC123/enhanced/enhanced.srt \
  --mode burn
```
