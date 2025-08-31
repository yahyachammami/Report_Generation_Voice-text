"""
Audio processing utilities with warning suppression and error handling.
"""

import warnings
import logging
import os
from typing import Optional, Dict, Any
import torch
import torchaudio

logger = logging.getLogger(__name__)

def setup_audio_warnings():
    """
    Configure warning filters for audio processing libraries.
    This suppresses non-critical warnings that don't affect functionality.
    """
    # Suppress torchaudio deprecation warnings
    warnings.filterwarnings("ignore", 
                          message=".*torchaudio._backend.utils.info has been deprecated.*",
                          category=UserWarning,
                          module="torchaudio.*")
    
    warnings.filterwarnings("ignore", 
                          message=".*torchaudio._backend.common.AudioMetaData has been deprecated.*",
                          category=UserWarning,
                          module="torchaudio.*")
    
    warnings.filterwarnings("ignore", 
                          message=".*this function's implementation will be changed to use torchaudio.load_with_torchcodec.*",
                          category=UserWarning,
                          module="torchaudio.*")
    
    # Suppress MPEG audio metadata warnings (these are non-critical)
    warnings.filterwarnings("ignore", 
                          message=".*No comment text / valid description.*",
                          category=UserWarning)
    
    # Suppress pyannote warnings about deprecated torchaudio functions
    warnings.filterwarnings("ignore", 
                          message=".*torchaudio._backend.utils.info has been deprecated.*",
                          category=UserWarning,
                          module="pyannote.*")
    
    # Suppress MPEG layer III subtype warnings
    warnings.filterwarnings("ignore", 
                          message=".*The MPEG_LAYER_III subtype is unknown to TorchAudio.*",
                          category=UserWarning,
                          module="torchaudio.*")

def configure_torch_for_audio():
    """
    Configure PyTorch settings for optimal audio processing.
    """
    # Set Windows symlink strategy to avoid permission issues
    os.environ["PYTORCH_WINDOWS_SYMLINK_STRATEGY"] = "copy"
    
    # Configure cache directories
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set cache directories for various libraries
    os.environ["SPEECHBRAIN_CACHE_DIR"] = os.path.join(cache_dir, "speechbrain")
    os.environ["HF_CACHE_DIR"] = os.path.join(cache_dir, "huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "huggingface")
    
    # Set torch to use CPU if CUDA is not available or causing issues
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU for audio processing")
        torch.set_num_threads(min(4, os.cpu_count() or 1))

def get_audio_info_safe(audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely get audio file information with fallback handling.
    Uses multiple approaches to handle different audio formats and library versions.
    """
    try:
        # Try to get info with the current method
        info = torchaudio.info(audio_path)
        return {
            "sample_rate": info.sample_rate,
            "num_frames": info.num_frames,
            "num_channels": info.num_channels,
            "bits_per_sample": getattr(info, 'bits_per_sample', 0)
        }
    except Exception as e:
        logger.warning(f"Could not get audio info with torchaudio.info for {audio_path}: {e}")
        
        # Fallback: try to load the audio and get info from the tensor
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            return {
                "sample_rate": sample_rate,
                "num_frames": waveform.shape[-1],
                "num_channels": waveform.shape[0],
                "bits_per_sample": 16  # Default assumption
            }
        except Exception as e2:
            logger.warning(f"Could not load audio for info extraction {audio_path}: {e2}")
            # Return default values
            return {
                "sample_rate": 16000,  # Default for Whisper
                "num_frames": 0,
                "num_channels": 1,
                "bits_per_sample": 16
            }

def load_audio_safe(audio_path: str, target_sr: int = 16000) -> Optional[torch.Tensor]:
    """
    Safely load audio file with error handling and resampling.
    """
    try:
        # Load audio with error handling
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
        
    except Exception as e:
        logger.error(f"Failed to load audio file {audio_path}: {e}")
        return None

def validate_audio_file(audio_path: str) -> bool:
    """
    Validate that an audio file can be processed.
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file does not exist: {audio_path}")
        return False
    
    # Check file size (avoid processing empty or corrupted files)
    file_size = os.path.getsize(audio_path)
    if file_size < 1024:  # Less than 1KB
        logger.error(f"Audio file too small: {audio_path} ({file_size} bytes)")
        return False
    
    # Try to get basic info
    info = get_audio_info_safe(audio_path)
    if info is None:
        return False
    
    # Check if audio has reasonable duration (at least 0.1 seconds)
    duration = info["num_frames"] / info["sample_rate"]
    if duration < 0.1:
        logger.error(f"Audio file too short: {audio_path} ({duration:.2f} seconds)")
        return False
    
    logger.info(f"Audio file validated: {audio_path} ({duration:.2f}s, {info['sample_rate']}Hz)")
    return True
