from typing import Dict, List
import logging
from pyannote.audio import Pipeline
import whisper
import torch
from app.utils.audio_utils import setup_audio_warnings, configure_torch_for_audio, validate_audio_file

logger = logging.getLogger(__name__)

class MeetingAnalyzer:
    def __init__(self, whisper_model: str = "base", huggingface_token: str = None):
        # Setup audio processing environment and suppress warnings
        setup_audio_warnings()
        configure_torch_for_audio()
        
        try:
            self.whisper_model = whisper.load_model(whisper_model)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
            
        self.diarization_pipeline = None
        if huggingface_token:
            try:
                # Configure environment for pyannote
                os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
                os.environ["HF_TOKEN"] = huggingface_token
                
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",  # Specify an older version for compatibility
                    use_auth_token=huggingface_token
                )
                logger.info("Successfully initialized speaker diarization")
            except Exception as e:
                logger.warning(f"Failed to initialize speaker diarization: {e}")
                logger.warning("Speaker diarization will be disabled. To enable it:")
                logger.warning("1. Visit https://huggingface.co/pyannote/segmentation")
                logger.warning("2. Accept the user conditions")
                logger.warning("3. Ensure your token has the required access")

    def analyze_meeting(self, audio_path: str) -> Dict:
        """
        Analyze meeting audio to produce structured data including transcription and speaker diarization.
        """
        if not self.whisper_model:
            raise ValueError("Whisper model is not initialized")

        # Validate audio file before processing
        if not validate_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")

        try:
            # Get the basic transcription
            result = self.whisper_model.transcribe(
                audio_path,
                initial_prompt="This is a business meeting transcript.",
                verbose=False
            )
            
            # Extract segments from the result
            segments = []
            for idx, segment in enumerate(result["segments"]):
                processed_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "speaker": f"Speaker {(idx % 2) + 1}"  # Alternate between Speaker 1 and 2
                }
                segments.append(processed_segment)

            # Try speaker diarization if available
            if self.diarization_pipeline:
                try:
                    diarization = self.diarization_pipeline(audio_path)
                    segments = self._merge_transcription_with_speakers(result["segments"], diarization)
                except Exception as e:
                    logger.warning(f"Speaker diarization failed: {e}")
                    logger.warning("Continuing with basic speaker separation")

            return {
                "language": result["language"],
                "segments": segments,
                "full_text": result["text"]
            }
        except Exception as e:
            logger.error(f"Error in meeting analysis: {str(e)}")
            raise

    def _merge_transcription_with_speakers(self, transcription_segments, diarization):
        """
        Merge whisper transcription segments with speaker diarization information.
        """
        merged_segments = []
        
        for segment in transcription_segments:
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Find the dominant speaker for this segment
            speaker_times = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= end_time and turn.end >= start_time:
                    overlap = min(turn.end, end_time) - max(turn.start, start_time)
                    speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap
            
            dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[0] if speaker_times else "Unknown Speaker"
            
            merged_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": dominant_speaker
            })
        
        return merged_segments
