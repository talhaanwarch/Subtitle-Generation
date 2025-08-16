# Audio Separation Module

This module provides audio separation functionality to isolate vocals from instrumental music, which can significantly improve transcription quality when dealing with audio that contains background music or instrumental tracks.

## Overview

The audio separation module uses the [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) library, which is a wrapper around various AI models (MDX-Net, VR Arch, Demucs, MDXC) trained for audio source separation. This is particularly useful for:

- Improving transcription accuracy when there's background music
- Creating karaoke-style videos by separating vocals and instrumental tracks
- Processing audio with mixed speech and music content

## Installation

### Prerequisites

1. **Python 3.10+** is required
2. **FFmpeg** must be installed on your system

### Install audio-separator

Choose one of the following installation methods based on your hardware:

#### CPU-only installation (recommended for most users)
```bash
pip install "audio-separator[cpu]"
```

#### GPU installation (CUDA 11.8 or 12.2)
```bash
pip install "audio-separator[gpu]"
```

#### Apple Silicon (M1/M2) with CoreML acceleration
```bash
pip install "audio-separator[cpu]"
```

### Verify Installation

You can verify the installation by running:
```bash
python -c "from separator.separate_audio import check_audio_separator_availability; print(check_audio_separator_availability())"
```

This should return `True` if the installation was successful.

## Configuration

### Using Configuration File

Add audio separation settings to your `config.yaml`:

```yaml
processing:
  audio:
    separation:
      # Enable audio separation
      enabled: true
      # Model to use for separation (can be model name or filename)
      # Examples: "UVR_MDXNET_KARA_2", "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "htdemucs_6s"
      model: "UVR_MDXNET_KARA_2"
      # Whether to automatically select the best model for vocals (overrides model setting)
      auto_select_best: false
      # Stem type to separate (vocals, drums, bass, etc.)
      stem_type: "vocals"
      # Output format for separated files
      output_format: "WAV"
      # Use separated vocals for transcription
      use_separated_for_transcription: true
```

### Using Environment Variables (Legacy Mode)

For the legacy pipeline, you can enable audio separation using:
```bash
export ENABLE_AUDIO_SEPARATION=true
```

## Available Models

The module supports various pre-trained models for different separation tasks:

### Vocal/Instrumental Separation
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (default) - High-quality vocal separation
- `UVR_MDXNET_KARA_2.onnx` - Good for karaoke-style separation
- `model_bs_roformer_ep_368_sdr_12.9628.ckpt` - Alternative roformer model

### Multi-stem Separation (Demucs models)
- `htdemucs_6s.yaml` - Separates into 6 stems: vocals, drums, bass, other, guitar, piano
- `htdemucs.yaml` - Separates into 4 stems: vocals, drums, bass, other

### List Available Models

You can list all available models programmatically:
```python
from separator.separate_audio import list_available_models

models = list_available_models()
for model in models:
    print(f"Model: {model['name']}")
    print(f"  Filename: {model['filename']}")
    print(f"  Stems: {', '.join(model['stems'])}")
    print(f"  SDR: {model['sdr']}")
```

### Get Best Models for Specific Stems

You can get the best performing models for specific stem types:
```python
from separator.separate_audio import get_best_models_for_stem, get_best_model_for_stem

# Get top 5 best vocal separation models
best_vocal_models = get_best_models_for_stem("vocals", limit=5)
for model in best_vocal_models:
    print(f"Model: {model['name']} (SDR: {model['sdr']})")

# Get the single best model for vocals
best_model = get_best_model_for_stem("vocals")
if best_model:
    print(f"Best vocal model: {best_model['name']} (SDR: {best_model['sdr']})")
```

### Validate Model Availability

You can validate if a specific model is available:
```python
from separator.separate_audio import validate_model

# Check by model name
is_valid, model_info = validate_model("UVR_MDXNET_KARA_2")
if is_valid:
    print(f"Model found: {model_info['name']} (SDR: {model_info['sdr']})")
else:
    print("Model not found")
```

## Usage

### Basic Usage

```python
from separator.separate_audio import separate_audio_file
from utils.paths import ensure_workdirs

# Get working directories for a video
work = ensure_workdirs("VIDEO_ID")

# Separate audio file with specific model
result = separate_audio_file(
    audio_path="path/to/audio.wav",
    output_dir=work.separated_dir,
    model_name="MDX-Net Model: UVR-MDX-NET Karaoke 2",  # Can use model name or filename
    output_format="WAV",
    sample_rate=16000,

)

# Or automatically select the best model for vocals
result = separate_audio_file(
    audio_path="path/to/audio.wav",
    output_dir=work.separated_dir,
    auto_select_best=True,  # Automatically select best model
    stem_type="vocals",     # Specify stem type
    output_format="WAV",
    sample_rate=16000,
)

print(f"Vocals: {result['vocals']}")
print(f"Instrumental: {result['instrumental']}")
```

### Advanced Usage

```python
from separator.separate_audio import AudioSeparator
from utils.paths import ensure_workdirs

# Get working directories for a video
work = ensure_workdirs("VIDEO_ID")

# Initialize separator with custom settings
separator = AudioSeparator(
    output_dir=work.separated_dir,
    model_name="Demucs v4: htdemucs_6s",  # Multi-stem model
    output_format="FLAC",
    sample_rate=44100,
    auto_select_best=False,  # Use specified model
    stem_type="vocals"
)

# Or automatically select the best model
separator = AudioSeparator(
    output_dir=work.separated_dir,
    auto_select_best=True,  # Automatically select best model
    stem_type="vocals",     # For vocal separation
    output_format="WAV",
    sample_rate=16000,
)

# Separate audio with custom output names
output_names = {
    "Vocals": "my_vocals",
    "Drums": "my_drums",
    "Bass": "my_bass",
    "Other": "my_other"
}

result = separator.separate_audio("path/to/audio.wav", output_names)
```

### Integration with Pipeline

The audio separation is automatically integrated into the main pipeline when enabled in the configuration. The pipeline will:

1. Extract audio from the video
2. Separate vocals from instrumental (if enabled)
3. Use the separated vocals for transcription
4. Continue with the rest of the pipeline (enhancement, translation, subtitles)

## Output Structure

When audio separation is enabled, the following directory structure is created:

```
outputs/
└── {video_id}/
    ├── audio/
    │   └── audio.wav                    # Original extracted audio
    ├── separated/
    │   ├── {video_id}_vocals.wav        # Separated vocals
    │   └── {video_id}_instrumental.wav  # Separated instrumental
    ├── transcripts/
    │   └── asr_*.json                   # Transcription results
    └── ...
```

## New Features

### Model Validation and Auto-Selection

The audio separator now includes intelligent model management:

1. **Model Validation**: The system validates that the specified model is available before using it
2. **Automatic Fallback**: If the specified model is not found, the system automatically switches to the best available model for the specified stem type
3. **Model Switching Logging**: The system logs when models are switched and provides clear feedback to users
2. **Auto-Selection**: Can automatically select the best performing model for a specific stem type
3. **Flexible Naming**: Supports both model names and filenames for easier configuration

### Configuration Examples

**Use a specific model by name:**
```yaml
processing:
  audio:
    separation:
      enabled: true
      model: "UVR_MDXNET_KARA_2"  # Model name instead of filename
      auto_select_best: false
```

**Automatically select the best model:**
```yaml
processing:
  audio:
    separation:
      enabled: true
      auto_select_best: true      # Overrides model setting
      stem_type: "vocals"         # Select best model for vocals
```

**Use a specific model for drums:**
```yaml
processing:
  audio:
    separation:
      enabled: true
      model: "htdemucs_6s"        # Multi-stem model
      stem_type: "drums"          # Separate drums specifically
      auto_select_best: false
```

## Performance Considerations

### Processing Time
- **CPU processing**: Slower but works on all systems
- **GPU processing**: Much faster but requires CUDA-compatible GPU
- **Model selection**: Different models have different speed/quality trade-offs

### Memory Usage
- Larger models require more RAM
- Processing longer audio files may require more memory
- Consider using `use_soundfile=True` for very long audio files

### Quality vs Speed Trade-offs
- **High quality**: Use `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (slower)
- **Fast processing**: Use `UVR_MDXNET_KARA_2.onnx` (faster)
- **Multi-stem**: Use Demucs models for detailed separation

## Troubleshooting

### Common Issues

1. **Import Error**: `audio-separator not available`
   - Solution: Install the package with `pip install "audio-separator[cpu]"`

2. **CUDA/GPU Issues**:
   - Ensure you have the correct CUDA version installed
   - Try CPU-only installation if GPU issues persist

3. **FFmpeg not found**:
   - Install FFmpeg: `apt-get install ffmpeg` (Ubuntu/Debian)
   - Or: `brew install ffmpeg` (macOS)

4. **Out of Memory**:
   - Use CPU processing instead of GPU
   - Process shorter audio segments
   - Use `use_soundfile=True` parameter

### Model Download Issues

Models are automatically downloaded on first use. If download fails:
- Check internet connection
- Verify write permissions to the model directory
- Try downloading manually from the [UVR models repository](https://github.com/Anjok07/ultimatevocalremovergui/tree/master/models)

## Examples

### Example 1: Basic Pipeline with Audio Separation

```bash
# Run pipeline with audio separation enabled
python pipeline/run.py --config config.yaml
```

### Example 2: Standalone Audio Separation

```bash
# Separate a single audio file
python separator/separate_audio.py path/to/audio.wav --video-id VIDEO_ID

# List all available models
python separator/separate_audio.py --list-models

# List models filtered by stem type
python separator/separate_audio.py --list-models --filter-stem vocals

# List top 10 models
python separator/separate_audio.py --list-models --limit 10
```

### Example 3: Custom Model Usage

```python
from separator.separate_audio import separate_audio_file
from utils.paths import ensure_workdirs

# Get working directories for a video
work = ensure_workdirs("VIDEO_ID")

# Use a different model for better quality
result = separate_audio_file(
    audio_path="song.wav",
    output_dir=work.separated_dir,
    model_name="Demucs v4: htdemucs_6s",  # Multi-stem model
    output_format="FLAC"
)

# Access individual stems
vocals_path = result['vocals']
drums_path = result['drums']
bass_path = result['bass']
```

### Example 4: Automatic Model Fallback

If you specify a model that doesn't exist, the system automatically switches to the best available model:

```python
# This will automatically switch to the best vocal separation model
result = separate_audio_file(
    audio_path="song.wav",
    output_dir=work.separated_dir,
    model_name="NonExistentModel",  # This model doesn't exist
    stem_type="vocals"  # Will switch to best vocal model
)

# The system will log: "Model switched from 'NonExistentModel' to 'Roformer Model: BS-Roformer-Viperx-1297'"
```

## Credits

This module uses the [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) library, which is based on the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) project. The underlying models are trained by the UVR community and various AI researchers.

## License

The audio separation functionality is subject to the same license as the python-audio-separator library (MIT License). Please respect the original model licenses and provide appropriate attribution when using the separated audio in your projects.
