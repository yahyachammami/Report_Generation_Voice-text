from typing import Dict, List
import logging
from groq import Groq
import os
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection, Resegmentation
from app.utils.audio_utils import setup_audio_warnings, configure_torch_for_audio

logger = logging.getLogger(__name__)

class MeetingSummarizer:
    def __init__(self, groq_api_key: str = None, huggingface_token: str = None):
        # Setup audio processing environment and suppress warnings
        setup_audio_warnings()
        configure_torch_for_audio()
        
        self.client = Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))
        
        # Initialize speech analysis models
        self.huggingface_token = huggingface_token or os.environ.get("HUGGINGFACE_TOKEN")
        try:
            self.segmentation_model = Model.from_pretrained("pyannote/segmentation", 
                                                          use_auth_token=self.huggingface_token)
            
            # Initialize pipelines with default parameters
            self.vad_pipeline = VoiceActivityDetection(segmentation=self.segmentation_model)
            self.osd_pipeline = OverlappedSpeechDetection(segmentation=self.segmentation_model)
            self.resegmentation_pipeline = Resegmentation(segmentation=self.segmentation_model,
                                                        diarization="baseline")
            
            # Set default hyper-parameters
            self.HYPER_PARAMETERS = {
                "onset": 0.5,
                "offset": 0.5,
                "min_duration_on": 0.0,
                "min_duration_off": 0.0
            }
            
            # Instantiate pipelines
            self.vad_pipeline.instantiate(self.HYPER_PARAMETERS)
            self.osd_pipeline.instantiate(self.HYPER_PARAMETERS)
            self.resegmentation_pipeline.instantiate(self.HYPER_PARAMETERS)
            
            logger.info("Successfully initialized speech analysis pipelines")
        except Exception as e:
            logger.warning(f"Failed to initialize speech analysis pipelines: {e}")
            self.segmentation_model = None

    def analyze_audio(self, audio_path: str, baseline_diarization: Dict = None) -> Dict:
        """
        Analyze the audio file for voice activity, overlapped speech, and perform resegmentation.
        
        Args:
            audio_path: Path to the audio file
            baseline_diarization: Optional baseline diarization data for resegmentation
        """
        if not self.segmentation_model:
            logger.warning("Speech analysis models not initialized. Skipping audio analysis.")
            return {}
            
        try:
            # Perform voice activity detection
            vad = self.vad_pipeline(audio_path)
            
            # Detect overlapped speech regions
            osd = self.osd_pipeline(audio_path)
            
            # Perform resegmentation if baseline diarization is provided
            if baseline_diarization:
                resegmented = self.resegmentation_pipeline({
                    "audio": audio_path,
                    "baseline": baseline_diarization
                })
            else:
                resegmented = None
            
            return {
                "voice_activity": vad,
                "overlapped_speech": osd,
                "resegmented_diarization": resegmented
            }
        except Exception as e:
            logger.error(f"Error in audio analysis: {str(e)}")
            return {}

    def generate_summary(self, meeting_data: Dict) -> Dict:
        """
        Generate a structured summary of the meeting using Groq's LLM.
        """
        try:
            # Create a structured prompt
            prompt = self._create_summary_prompt(meeting_data)
            
            # Get summary from Groq
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional meeting summarizer. 
                        Analyze the meeting transcript and provide a clear, structured summary including:
                        1. Main points and overview
                        2. Key topics discussed
                        3. All decisions made
                        4. Action items with assigned responsibilities
                        5. Follow-up items and deadlines"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=4000
            )

            # Extract and structure the response
            response = chat_completion.choices[0].message.content
            return self._structure_response(response)
        except Exception as e:
            logger.error(f"Error in summary generation: {str(e)}")
            raise

    def _create_summary_prompt(self, meeting_data: Dict) -> str:
        """
        Create a detailed prompt for the meeting summary.
        """
        segments = meeting_data["segments"]
        speakers_text = []
        
        for segment in segments:
            speakers_text.append(f"{segment['speaker']}: {segment['text']}")
        
        transcript = "\n".join(speakers_text)
        
        return f"""You are an assistant specialized in creating professional meeting summaries.
The following is a transcript of a meeting conducted in {meeting_data['language']}.

        Transcript:
        {transcript}

        Your task:
Carefully read the transcript and generate a structured meeting summary. 
Ensure clarity, conciseness, and completeness.

Please include the following sections in your response:

1. **Overall Summary**  
   - Provide a concise paragraph summarizing the purpose of the meeting and its main outcomes.

2. **Key Topics Discussed**  
   - List the main topics covered during the meeting (use bullet points).

3. **Decisions Made**  
   - Clearly list all decisions reached during the meeting (bullet points).

4. **Action Items**  
   - Provide a list of specific tasks, who is responsible, and any relevant deadlines.  
   Example: *[Task] → Assigned to [Person], Deadline: [Date]*

5. **Follow-ups / Next Steps**  
   - Mention any pending discussions, open questions, or future agenda items.

Formatting Guidelines:
- Use clear headings and bullet points.
- Keep summaries concise but detailed enough for someone who did not attend the meeting.
- Do not add information not present in the transcript.
"""

    def _structure_response(self, response: str) -> Dict:
        """
        Structure the LLM response into organized sections.
        """
        sections = response.split("\n\n")
        structured_summary = {
            "overview": "",
            "topics": [],
            "decisions": [],
            "action_items": [],
            "follow_up": []
        }

        current_section = None
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            if "main points" in section.lower() or "summary" in section.lower():
                current_section = "overview"
                structured_summary["overview"] = section
            elif "topics" in section.lower():
                current_section = "topics"
            elif "decision" in section.lower():
                current_section = "decisions"
            elif "action" in section.lower():
                current_section = "action_items"
            elif "follow" in section.lower():
                current_section = "follow_up"
            elif current_section and current_section != "overview":
                # Clean up bullet points and add to appropriate section
                items = [item.strip().strip('•').strip('-').strip() 
                        for item in section.split('\n') 
                        if item.strip()]
                structured_summary[current_section].extend(items)

        return structured_summary
