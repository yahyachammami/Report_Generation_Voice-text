"""
Audio processing configuration and environment setup.
"""

import os
import warnings
from typing import Dict, Any

def configure_audio_environment():
    """
    Configure environment variables and settings for optimal audio processing.
    """
    # Set environment variables to handle various audio processing issues
    
    # Disable problematic audio backends that cause MPEG errors
    os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
    
    # Configure FFmpeg for better audio format support
    os.environ["FFMPEG_BINARY"] = "ffmpeg"
    
    # Set audio processing to be more tolerant of metadata issues
    os.environ["SOUNDFILE_IGNORE_METADATA"] = "1"
    
    # Configure PyTorch for better audio processing
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Set cache directories
    cache_base = os.path.join(os.path.expanduser("~"), ".cache")
    os.makedirs(cache_base, exist_ok=True)
    
    cache_dirs = {
        "SPEECHBRAIN_CACHE_DIR": os.path.join(cache_base, "speechbrain"),
        "HF_CACHE_DIR": os.path.join(cache_base, "huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(cache_base, "huggingface"),
        "TRANSFORMERS_CACHE": os.path.join(cache_base, "transformers"),
        "TORCH_HOME": os.path.join(cache_base, "torch"),
    }
    
    for env_var, path in cache_dirs.items():
        os.environ[env_var] = path
        os.makedirs(path, exist_ok=True)

def get_audio_processing_config() -> Dict[str, Any]:
    """
    Get configuration for audio processing with fallback values.
    """
    return {
        "default_sample_rate": 16000,  # Whisper's preferred sample rate
        "max_audio_duration": 3600,    # 1 hour max
        "min_audio_duration": 0.1,     # 100ms minimum
        "supported_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
        "max_file_size_mb": 500,       # 500MB max file size
        "chunk_size_seconds": 30,      # Process in 30-second chunks
        "overlap_seconds": 2,          # 2-second overlap between chunks
    }

def setup_audio_logging():
    """
    Configure logging for audio processing to reduce noise.
    """
    # Suppress specific library warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
    
    # Suppress MPEG-related warnings
    warnings.filterwarnings("ignore", message=".*No comment text / valid description.*")
    warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III subtype is unknown.*")
    warnings.filterwarnings("ignore", message=".*process_comment.*")
    
    # Suppress deprecation warnings for known issues
    warnings.filterwarnings("ignore", message=".*torchaudio._backend.utils.info has been deprecated.*")
    warnings.filterwarnings("ignore", message=".*torchaudio._backend.common.AudioMetaData has been deprecated.*")
    warnings.filterwarnings("ignore", message=".*will be changed to use torchaudio.load_with_torchcodec.*")

def validate_audio_environment() -> bool:
    """
    Validate that the audio processing environment is properly configured.
    """
    try:
        import torch
        import torchaudio
        import whisper
        
        # Check if basic audio processing works
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU for audio processing")
        
        # Check cache directories
        cache_base = os.path.join(os.path.expanduser("~"), ".cache")
        if not os.path.exists(cache_base):
            print(f"Warning: Cache directory not found: {cache_base}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"Error: Missing required audio processing library: {e}")
        return False
    except Exception as e:
        print(f"Error: Audio environment validation failed: {e}")
        return False
