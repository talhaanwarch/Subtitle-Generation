"""Configuration management for subtitle generation pipeline."""

import os
import sys
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class VideoConfig:
    """Video processing configuration."""
    url: str = ""
    tmp_downloads_dir: str = "tmp_downloads"
    input_language: str = ""


@dataclass
class ASRConfig:
    """ASR (Automatic Speech Recognition) configuration."""
    backend: str = "local"  # "local" or "groq"
    whisper_model: str = "base"
    groq_model: str = "distil-whisper-large-v3-en"


@dataclass
class EnhancerConfig:
    """Enhancer configuration."""
    enabled: bool = True
    temperature: float = 0.0


@dataclass
class TranslatorConfig:
    """Translator configuration."""
    enabled: bool = False
    target_language: str = ""
    temperature: float = 0.1


@dataclass
class LLMConfig:
    """LLM (Large Language Model) configuration."""
    backend: str = "groq"
    model: str = "llama3-8b-8192"
    enhancer: EnhancerConfig = field(default_factory=EnhancerConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)


@dataclass
class SubtitleConfig:
    """Subtitle generation configuration."""
    mode: str = "soft"  # "soft" or "burn"
    box_opacity: float = 0.6


@dataclass
class APIConfig:
    """API configuration."""
    groq_api_key: str = ""


@dataclass
class AudioConfig:
    """Audio extraction configuration."""
    sample_rate: int = 16000
    mono: bool = True


# Remove old EnhancementConfig as it's now part of LLMConfig


@dataclass
class OutputConfig:
    """Output directory configuration."""
    audio_dir: str = "audio"
    transcripts_dir: str = "transcripts"
    enhanced_dir: str = "enhanced"
    translated_dir: str = "translated"
    subtitled_dir: str = "subtitled"


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class VADParameters:
    """Voice Activity Detection parameters."""
    min_silence_duration_ms: int = 500


@dataclass
class FasterWhisperConfig:
    """Faster-whisper configuration."""
    device: str = "auto"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: VADParameters = field(default_factory=VADParameters)
    word_timestamps: bool = False


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    delay_seconds: int = 1


@dataclass
class AdvancedConfig:
    """Advanced configuration."""
    faster_whisper: FasterWhisperConfig = field(default_factory=FasterWhisperConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class Config:
    """Main configuration class."""
    video: VideoConfig = field(default_factory=VideoConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    subtitles: SubtitleConfig = field(default_factory=SubtitleConfig)
    api: APIConfig = field(default_factory=APIConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    def __post_init__(self):
        """Post-initialization to handle environment variables and validation."""
        # Load API key from environment if not set in config
        if not self.api.groq_api_key:
            self.api.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        
        # Validate ASR backend
        if self.asr.backend not in ["local", "groq"]:
            raise ValueError(f"Invalid ASR backend: {self.asr.backend}. Must be 'local' or 'groq'")
        
        # Validate subtitle mode
        if self.subtitles.mode not in ["soft", "burn"]:
            raise ValueError(f"Invalid subtitle mode: {self.subtitles.mode}. Must be 'soft' or 'burn'")
        
        # Validate box opacity
        if not (0.0 <= self.subtitles.box_opacity <= 1.0):
            raise ValueError(f"Box opacity must be between 0.0 and 1.0, got: {self.subtitles.box_opacity}")
        
        # Note: Translation is now explicitly controlled by the enabled flag
        # Users must set both enabled=True and provide a target_language for translation to occur


def _nested_dict_to_dataclass(cls, data: Dict[str, Any]) -> Any:
    """Convert nested dictionary to dataclass instance."""
    if not data:
        return cls()
    
    # Get the field types for this dataclass
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            
            # Handle nested dataclasses
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = _nested_dict_to_dataclass(field_type, value)
            else:
                kwargs[key] = value
    
    return cls(**kwargs)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml in project root.
        
    Returns:
        Config instance with loaded configuration.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file has invalid YAML syntax.
        ValueError: If config contains invalid values.
    """
    if config_path is None:
        # Find project root (directory containing this file's parent's parent)
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    
    # Convert nested dictionaries to appropriate dataclass instances
    config_kwargs = {}
    for section_name, section_data in config_data.items():
        if section_name == "video":
            config_kwargs[section_name] = _nested_dict_to_dataclass(VideoConfig, section_data)
        elif section_name == "asr":
            config_kwargs[section_name] = _nested_dict_to_dataclass(ASRConfig, section_data)
        elif section_name == "llm":
            # Handle nested LLM config
            llm_kwargs = {k: v for k, v in section_data.items() if k not in ["enhancer", "translator"]}
            if "enhancer" in section_data:
                llm_kwargs["enhancer"] = _nested_dict_to_dataclass(EnhancerConfig, section_data["enhancer"])
            if "translator" in section_data:
                llm_kwargs["translator"] = _nested_dict_to_dataclass(TranslatorConfig, section_data["translator"])
            config_kwargs[section_name] = LLMConfig(**llm_kwargs)
        elif section_name == "subtitles":
            config_kwargs[section_name] = _nested_dict_to_dataclass(SubtitleConfig, section_data)
        elif section_name == "api":
            config_kwargs[section_name] = _nested_dict_to_dataclass(APIConfig, section_data)
        elif section_name == "processing":
            # Handle nested processing config
            processing_kwargs = {}
            if "audio" in section_data:
                processing_kwargs["audio"] = _nested_dict_to_dataclass(AudioConfig, section_data["audio"])
            if "output" in section_data:
                processing_kwargs["output"] = _nested_dict_to_dataclass(OutputConfig, section_data["output"])
            config_kwargs[section_name] = ProcessingConfig(**processing_kwargs)
        elif section_name == "advanced":
            # Handle nested advanced config
            advanced_kwargs = {}
            if "faster_whisper" in section_data:
                fw_data = section_data["faster_whisper"]
                fw_kwargs = {k: v for k, v in fw_data.items() if k != "vad_parameters"}
                if "vad_parameters" in fw_data:
                    fw_kwargs["vad_parameters"] = _nested_dict_to_dataclass(VADParameters, fw_data["vad_parameters"])
                advanced_kwargs["faster_whisper"] = FasterWhisperConfig(**fw_kwargs)
            if "retry" in section_data:
                advanced_kwargs["retry"] = _nested_dict_to_dataclass(RetryConfig, section_data["retry"])
            config_kwargs[section_name] = AdvancedConfig(**advanced_kwargs)
    
    return Config(**config_kwargs)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config instance to save.
        config_path: Path where to save the config file.
    """
    config_path = Path(config_path)
    
    # Convert dataclass to dictionary
    def dataclass_to_dict(obj) -> Dict[str, Any]:
        """Convert dataclass instance to dictionary."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_obj in obj.__dataclass_fields__.items():
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field_name] = dataclass_to_dict(value)
                else:
                    result[field_name] = value
            return result
        return obj
    
    config_dict = dataclass_to_dict(config)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)


# Global config instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to config file. If None, uses default path.
        
    Returns:
        Global Config instance.
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config(config_path)
    return _global_config


def set_config(config: Config) -> None:
    """
    Set global configuration instance.
    
    Args:
        config: Config instance to set as global.
    """
    global _global_config
    _global_config = config
