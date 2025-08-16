"""
Audio Separation Module

This module provides functionality to separate audio into different stems (vocals, instrumental, etc.)
using the python-audio-separator library. This is particularly useful for improving transcription
quality when there is background music or instrumental audio mixed with speech.
"""

import os
import sys
import json
import subprocess
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ensure package path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from audio_separator.separator import Separator
except ImportError:
    raise ImportError("Warning: audio-separator not available. Install with: pip install audio-separator[cpu] or audio-separator[gpu]")

from utils.logging_utils import get_logger
from utils.paths import ensure_workdirs

logger = get_logger(__name__)


def check_gpu_availability() -> bool:
    """
    Check if GPU is available using both PyTorch and ONNX Runtime.
    
    Returns:
        True if GPU is available, False otherwise
    """
    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU detected via PyTorch: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        logger.debug("PyTorch not available for GPU detection")
    except Exception as e:
        logger.debug(f"PyTorch GPU detection failed: {e}")
    
    # Try ONNX Runtime as fallback
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            logger.info("GPU detected via ONNX Runtime CUDA provider")
            return True
        elif 'ROCMExecutionProvider' in providers:
            logger.info("GPU detected via ONNX Runtime ROCm provider")
            return True
    except ImportError:
        logger.debug("ONNX Runtime not available for GPU detection")
    except Exception as e:
        logger.debug(f"ONNX Runtime GPU detection failed: {e}")
    
    logger.info("No GPU detected, will use CPU")
    return False


class AudioSeparator:
    """
    Audio separation class that handles separating audio into different stems.
    
    This class provides methods to separate audio files into vocals and instrumental
    tracks, which can improve transcription quality when dealing with music or
    background audio.
    """
    
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 model_name: str = "Roformer Model: BS-Roformer-Viperx-1297",
                 output_format: str = "WAV",
                 sample_rate: int = 16000,
                 auto_select_best: bool = False,
                 stem_type: str = "vocals"):
        """
        Initialize the AudioSeparator.
        
        Args:
            output_dir: Directory to save separated audio files
            model_name: Model name or filename to use for separation
            output_format: Output format for separated files (WAV, MP3, FLAC, etc.)
            sample_rate: Sample rate for output audio
            auto_select_best: Whether to automatically select the best model for the stem type
            stem_type: Stem type to use when auto_select_best is True (e.g., 'vocals', 'drums')
        """
      
        self.output_dir = output_dir
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.stem_type = stem_type
        self.requested_model_name = model_name
        self.actual_model_info = None
        self.model_switched = False
 
        # Handle model selection
        if auto_select_best:
            logger.info(f"Auto-selecting best model for stem type: {stem_type}")
            best_model = self.get_best_model_for_stem(stem_type)
            if best_model:
                self.model_filename = best_model["filename"]
                self.actual_model_info = best_model
                logger.info(f"Selected best model: {best_model['name']} (SDR: {best_model['sdr']})")
            else:
                logger.warning(f"No models found for stem type '{stem_type}', using default")
                self.model_filename = self._get_model_filename(model_name)
        else:
            # Get filename from model name
            self.model_filename = self._get_model_filename(model_name)
            
            # Validate the provided model
            is_valid, model_info = self.validate_model(self.model_filename)
            if is_valid:
                self.actual_model_info = model_info
                logger.info(f"Using model: {model_info['name']} (SDR: {model_info['sdr']})")
            else:
                logger.warning(f"Model '{model_name}' not found in available models")
                logger.info(f"Automatically switching to best model for stem type: {stem_type}")
                
                # Switch to best model for the specified stem type
                best_model = self.get_best_model_for_stem(stem_type)
                if best_model:
                    self.model_filename = best_model["filename"]
                    self.actual_model_info = best_model
                    self.model_switched = True
                    logger.info(f"Switched to best model: {best_model['name']} (SDR: {best_model['sdr']})")
                else:
                    logger.warning(f"No models found for stem type '{stem_type}', using default model")
                    # Keep the original model_filename as fallback
        
        # Initialize the separator
        self.separator = None
        self._initialize_separator()
    
    def _get_model_filename(self, model_identifier: str) -> str:
        """
        Get model filename from model name or return the identifier if it's already a filename.
        
        Args:
            model_identifier: Model name or filename
            
        Returns:
            Model filename
        """
        # Get available models
        available_models = self.get_available_models()
        
        # Check if it's a known model name
        for model in available_models:
            if model_identifier.lower() == model["name"].lower():
                return model["filename"]
        
        # If not found in available models, assume it's already a filename
        return model_identifier

    def _initialize_separator(self):
        """Initialize the audio separator with the specified configuration."""
        try:
            # Configure separator parameters
            separator_params = {
                "output_dir": self.output_dir,
                "output_format": self.output_format,
                "sample_rate": self.sample_rate,
                "use_autocast": check_gpu_availability(),  # Only use autocast for GPU
            }
            
            # Remove None values
            separator_params = {k: v for k, v in separator_params.items() if v is not None}
            
            self.separator = Separator(**separator_params)
            
            # Load the model
            logger.info(f"Loading audio separation model: {self.model_filename}")
            self.separator.load_model(model_filename=self.model_filename)
            logger.info("Audio separation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio separator: {e}")
            raise
    
    def get_available_models(self, filter_stem: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get list of available separation models using audio-separator CLI.
        
        Args:
            filter_stem: Filter models by stem type (e.g., 'vocals', 'drums')
            limit: Limit number of results returned
            
        Returns:
            List of dictionaries containing model information
        """
        try:
            # Build the command
            cmd = ["audio-separator", "-l", "--list_format=json"]
            
            if filter_stem:
                cmd.extend(["--list_filter", filter_stem])
            
            if limit:
                cmd.extend(["--list_limit", str(limit)])
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse JSON output
            models_data = json.loads(result.stdout)
            
            # Transform the data to match our expected format
            models = []
            
            # The JSON structure has models organized by architecture (VR, MDX, Demucs, MDXC)
            # Each architecture contains multiple models
            models_dict = {}
            
            # Flatten the nested structure
            for arch_name, arch_models in models_data.items():
                if isinstance(arch_models, dict):
                    for model_name, model_info in arch_models.items():
                        if isinstance(model_info, dict) and "filename" in model_info:
                            models_dict[model_name] = model_info
            
            for model_name, model_info in models_dict.items():
                # Extract stems from the model info
                stems = model_info.get("stems", [])
                
                # Extract SDR values from scores
                scores = model_info.get("scores", {})
                sdr_values = []
                
                for stem_type, score_data in scores.items():
                    if isinstance(score_data, dict) and "SDR" in score_data:
                        sdr_values.append(str(score_data["SDR"]))
                
                # Use the highest SDR value as the model's SDR
                best_sdr = max(sdr_values) if sdr_values else "N/A"
                
                models.append({
                    "filename": model_info.get("filename", ""),
                    "name": model_name,
                    "stems": stems,
                    "sdr": best_sdr,
                    "target_stem": model_info.get("target_stem", ""),
                    "download_files": model_info.get("download_files", [])
                })
            
            logger.info(f"Found {len(models)} available models")
            return models
            
        except subprocess.CalledProcessError as e:
            logger.error(f"audio-separator command failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output from audio-separator: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def get_best_models_for_stem(self, stem_type: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Get the best performing models for a specific stem type.
        
        Args:
            stem_type: The stem type to filter for (e.g., 'vocals', 'drums')
            limit: Number of top models to return
            
        Returns:
            List of dictionaries containing model information, sorted by SDR
        """
        models = self.get_available_models(filter_stem=stem_type, limit=limit)
        
        # Sort by SDR value (highest first)
        def sort_key(model):
            try:
                return float(model.get("sdr", "0"))
            except (ValueError, TypeError):
                return 0.0
        
        return sorted(models, key=sort_key, reverse=True)
    
    def validate_model(self, model_identifier: str) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Validate if a model is available by name or filename.
        
        Args:
            model_identifier: Model name or filename to validate
            
        Returns:
            Tuple of (is_valid, model_info) where model_info is None if not found
        """
        try:
            available_models = self.get_available_models()
            for model in available_models:
                # Check by name or filename
                if (model_identifier.lower() == model["name"].lower() or 
                    model_identifier == model["filename"]):
                    return True, model
            
            return False, None
            
        except Exception as e:
            logger.error(f"Failed to validate model {model_identifier}: {e}")
            return False, None
    
    def get_best_model_for_stem(self, stem_type: str = "vocals") -> Optional[Dict[str, str]]:
        """
        Get the best performing model for a specific stem type.
        
        Args:
            stem_type: The stem type to get best model for (e.g., 'vocals', 'drums')
            
        Returns:
            Best model information or None if no models found
        """
        try:
            best_models = self.get_best_models_for_stem(stem_type=stem_type, limit=1)
            return best_models[0] if best_models else None
        except Exception as e:
            logger.error(f"Failed to get best model for stem {stem_type}: {e}")
            return None

    def separate_audio(self, audio_path: str, output_names: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Separate audio into vocals and instrumental tracks.
        
        Args:
            audio_path: Path to the input audio file
            output_names: Custom names for output files
            
        Returns:
            Dictionary with paths to separated audio files
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.separator is None:
            raise RuntimeError("Audio separator not initialized")
        
        try:
            logger.info(f"Starting audio separation for: {audio_path}")
            
            # Set custom output names if provided
            if output_names is None:
                # Default output names
                base_name = Path(audio_path).stem
                output_names = {
                    "Vocals": f"{base_name}_vocals",
                    "Instrumental": f"{base_name}_instrumental"
                }
            
            # Perform separation
            output_files = self.separator.separate(audio_path, output_names)
            
            logger.info(f"Audio separation completed. Output files: {output_files}")
            
            # The separator returns a list of file paths, so we need to map them to stem types
            # Based on the output_names we provided, we can infer the stem types
            result = {}
            if isinstance(output_files, list):
                # Map the output files to their corresponding stem types
                for i, file_path in enumerate(output_files):
                    if i < len(output_names):
                        stem_type = list(output_names.keys())[i]
                        result[stem_type.lower()] = file_path
                    else:
                        # Fallback naming if we have more files than expected
                        result[f"stem_{i}"] = file_path
            elif isinstance(output_files, dict):
                # If it's already a dictionary, use it directly
                result = {k.lower(): v for k, v in output_files.items()}
            else:
                # Fallback for unexpected return types
                result = {"output": str(output_files)}
            
            return result
            
        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the model being used.
        
        Returns:
            Dictionary with model information including whether it was switched
        """
        return {
            "requested_model": self.requested_model_name,
            "actual_model": self.actual_model_info,
            "model_filename": self.model_filename,
            "model_switched": self.model_switched,
            "stem_type": self.stem_type
        }



def separate_audio_file(audio_path: str, 
                       output_dir: Optional[str] = None,
                       model_name: str = "Roformer Model: BS-Roformer-Viperx-1297",
                       output_format: str = "WAV",
                       sample_rate: int = 16000,
                       auto_select_best: bool = False,
                       stem_type: str = "vocals") -> Dict[str, str]:
    """
    Convenience function to separate a single audio file.
    
    Args:
        audio_path: Path to the input audio file
        output_dir: Directory to save separated audio files
        model_name: Model name or filename to use for separation
        output_format: Output format for separated files
        sample_rate: Sample rate for output audio
        auto_select_best: Whether to automatically select the best model for the stem type
        stem_type: Stem type to use when auto_select_best is True (e.g., 'vocals', 'drums')
        
    Returns:
        Dictionary with paths to separated audio files
    """
    separator = AudioSeparator(
        output_dir=output_dir,
        model_name=model_name,
        output_format=output_format,
        sample_rate=sample_rate,
        auto_select_best=auto_select_best,
        stem_type=stem_type
    )
    
    # Log model information
    model_info = separator.get_model_info()
    if model_info["model_switched"]:
        logger.info(f"Model switched from '{model_info['requested_model']}' to '{model_info['actual_model']['name']}'")
    
    return separator.separate_audio(audio_path)


def list_available_models(filter_stem: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    List available audio separation models.
    
    Args:
        filter_stem: Filter models by stem type (e.g., 'vocals', 'drums')
        limit: Limit number of results returned
        
    Returns:
        List of dictionaries containing model information
    """
    separator = AudioSeparator()
    return separator.get_available_models(filter_stem=filter_stem, limit=limit)


def print_available_models(filter_stem: Optional[str] = None, limit: Optional[int] = None):
    """
    Print available models in a formatted way.
    
    Args:
        filter_stem: Filter models by stem type
        limit: Limit number of results
    """
    models = list_available_models(filter_stem=filter_stem, limit=limit)
    
    if not models:
        print("No models found.")
        return
    
    print(f"\nAvailable Audio Separation Models{f' (filtered by stem: {filter_stem})' if filter_stem else ''}:")
    print("=" * 80)
    
    for i, model in enumerate(models, 1):
        print(f"{i:3d}. {model['name']}")
        print(f"     Filename: {model['filename']}")
        print(f"     Stems: {', '.join(model['stems']) if model['stems'] else 'N/A'}")
        print(f"     SDR: {model['sdr']}")
        print(f"     Target: {model['target_stem'] if model['target_stem'] else 'N/A'}")
        print()


if __name__ == "__main__":
    # use following for test
    #source ~/venv/bin/activate && python separator/separate_audio.py outputs/dDD6bmOmz8A/audio/audio.wav --video-id dDD6bmOmz8A
    # Test the audio separation functionality
    import argparse
    
    parser = argparse.ArgumentParser(description="Test audio separation")
    parser.add_argument("audio_file", nargs='?', help="Path to audio file to separate")
    parser.add_argument("--video-id", help="Video ID for output directory")
    parser.add_argument("--model", default="Roformer Model: BS-Roformer-Viperx-1297", 
                       help="Model to use for separation (can be model name or filename)")
    parser.add_argument("--format", default="WAV", help="Output format")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--filter-stem", help="Filter models by stem type (e.g., vocals, drums)")
    parser.add_argument("--limit", type=int, help="Limit number of models to show")
    
    args = parser.parse_args()
    
    # If --list-models is specified, show available models
    if args.list_models:
        print_available_models(filter_stem=args.filter_stem, limit=args.limit)
        sys.exit(0)
    
    # Check if audio_file is provided
    if not args.audio_file:
        print("Error: audio_file is required unless --list-models is specified")
        print("Usage examples:")
        print("  python separator/separate_audio.py audio.wav --video-id VIDEO_ID")
        print("  python separator/separate_audio.py --list-models")
        print("  python separator/separate_audio.py --list-models --filter-stem vocals")
        sys.exit(1)
    
    # Check if video-id is provided
    if not args.video_id:
        print("Error: --video-id is required")
        sys.exit(1)
    

    # Get working directories using video_id
    work = ensure_workdirs(args.video_id)
    logger.info(f"Separating audio {args.audio_file} → {work.separated_dir}")

    try:
        # Create separator to get model info
        separator = AudioSeparator(
            output_dir=work.separated_dir,
            model_name=args.model,
            output_format=args.format,
            sample_rate=args.sample_rate,
        )
        
        # Check if model was switched
        model_info = separator.get_model_info()
        if model_info["model_switched"]:
            print(f"⚠️  Model switched from '{model_info['requested_model']}' to '{model_info['actual_model']['name']}'")
        
        # Perform separation
        result = separator.separate_audio(args.audio_file)
        
        print("✅ Separation completed successfully!")
        print("Output files:")
        for stem_type, file_path in result.items():
            print(f"  {stem_type}: {file_path}")
            
    except Exception as e:
        print(f"❌ Error during separation: {e}")
        sys.exit(1)
