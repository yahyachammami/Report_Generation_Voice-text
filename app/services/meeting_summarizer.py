from typing import Dict, List, Optional
import logging
from datetime import datetime
import json

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
                        "content": """You are a professional meeting summarizer with expertise in extracting key insights, 
                        decisions, and action items from business meetings. Your summaries are always structured, 
                        comprehensive, and actionable. You pay special attention to:
                        
                        1. Identifying the core purpose and outcomes of the meeting
                        2. Extracting specific decisions with clear ownership
                        3. Documenting action items with assignees and deadlines
                        4. Noting important context and rationale for decisions
                        5. Highlighting risks, dependencies, and follow-up requirements
                        
                        Format your response using clear section headers and bullet points for easy parsing."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,  # Lower temperature for more consistent formatting
                max_tokens=5000,
                response_format={"type": "json_object"}  # Request JSON output for better structure
            )

            # Extract and structure the response
            response = chat_completion.choices[0].message.content
            
            # Parse JSON response or fall back to text parsing
            try:
                structured_response = json.loads(response)
                return self._enhance_summary_with_metadata(structured_response, meeting_data)
            except json.JSONDecodeError:
                logger.warning("LLM response not in JSON format, falling back to text parsing")
                return self._structure_response(response)
                
        except Exception as e:
            logger.error(f"Error in summary generation: {str(e)}")
            raise

    def _create_summary_prompt(self, meeting_data: Dict) -> str:
        """
        Create a detailed prompt for the meeting summary with enhanced context.
        """
        segments = meeting_data["segments"]
        speakers_text = []
        
        for segment in segments:
            speakers_text.append(f"{segment['speaker']}: {segment['text']}")
        
        transcript = "\n".join(speakers_text)
        
        # Extract meeting metadata for better context
        meeting_title = meeting_data.get("title", "Unknown Meeting")
        meeting_date = meeting_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        participants = meeting_data.get("participants", [])
        
        return f"""MEETING SUMMARY REQUEST

Meeting Title: {meeting_title}
Date: {meeting_date}
Participants: {', '.join(participants) if participants else 'Not specified'}
Language: {meeting_data.get('language', 'Unknown')}

TRANSCRIPT:
{transcript}

ANALYSIS REQUEST:

Please provide a comprehensive, structured summary in JSON format with the following sections:

1. "executive_summary": A concise overview of the meeting's purpose and key outcomes (2-3 paragraphs)
2. "key_decisions": List of decisions made with:
   - Decision description
   - Decision maker/owner
   - Rationale/context
   - Implementation timeline if available
3. "action_items": Specific tasks with:
   - Task description
   - Assignee(s)
   - Deadline (if specified)
   - Dependencies/constraints
4. "topics_discussed": Main discussion points with key insights
5. "follow_up_items": Items requiring further investigation or future discussion
6. "risks_issues": Any risks, concerns, or unresolved issues identified
7. "next_steps": Recommended actions and timeline for follow-up

Please ensure:
- Use clear, actionable language
- Extract specific names, dates, and commitments from the transcript
- Maintain neutral, professional tone
- Focus on business value and clarity
- Format all dates consistently (YYYY-MM-DD)
- Return ONLY valid JSON without any additional text"""

    def _structure_response(self, response: str) -> Dict:
        """
        Structure the LLM response into organized sections with enhanced parsing.
        """
        # Enhanced parsing for text responses
        sections = {
            "executive_summary": "",
            "key_decisions": [],
            "action_items": [],
            "topics_discussed": [],
            "follow_up_items": [],
            "risks_issues": [],
            "next_steps": []
        }
        
        # Try to parse section headers more intelligently
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            lower_line = line.lower()
            if "executive" in lower_line or "summary" in lower_line:
                current_section = "executive_summary"
            elif "decision" in lower_line:
                current_section = "key_decisions"
            elif "action" in lower_line and "item" in lower_line:
                current_section = "action_items"
            elif "topic" in lower_line or "discuss" in lower_line:
                current_section = "topics_discussed"
            elif "follow" in lower_line or "future" in lower_line:
                current_section = "follow_up_items"
            elif "risk" in lower_line or "issue" in lower_line:
                current_section = "risks_issues"
            elif "next" in lower_line or "step" in lower_line:
                current_section = "next_steps"
            elif current_section:
                # Clean and add content to current section
                if current_section == "executive_summary":
                    sections[current_section] += line + " "
                else:
                    # Handle bullet points and list items
                    if line.startswith(('•', '-', '*')) or line[0].isdigit():
                        clean_item = line.lstrip('•-* ').lstrip('0123456789.) ')
                        if clean_item and len(clean_item) > 3:  # Minimum meaningful length
                            sections[current_section].append(clean_item)
        
        # Clean up executive summary
        sections["executive_summary"] = sections["executive_summary"].strip()
        
        return self._enhance_summary_with_metadata(sections, {})

    def _enhance_summary_with_metadata(self, summary: Dict, meeting_data: Dict) -> Dict:
        """
        Add metadata and quality checks to the summary.
        """
        enhanced_summary = {
            "metadata": {
                "generation_date": datetime.now().isoformat(),
                "model_used": "llama-3.3-70b-versatile",
                "meeting_title": meeting_data.get("title", ""),
                "meeting_date": meeting_data.get("date", ""),
                "participants": meeting_data.get("participants", []),
                "language": meeting_data.get("language", "")
            },
            "summary": summary,
            "quality_metrics": {
                "decision_count": len(summary.get("key_decisions", [])),
                "action_item_count": len(summary.get("action_items", [])),
                "has_deadlines": any("deadline" in str(item).lower() for item in summary.get("action_items", [])),
                "has_assignees": any("assignee" in str(item).lower() for item in summary.get("action_items", []))
            }
        }
        
        return enhanced_summary

    def generate_detailed_report(self, meeting_data: Dict, audio_analysis: Dict = None) -> Dict:
        """
        Generate a comprehensive report including summary and audio insights.
        """
        summary = self.generate_summary(meeting_data)
        
        report = {
            "meeting_report": summary,
            "audio_analysis": audio_analysis or {},
            "transcript_metrics": {
                "total_segments": len(meeting_data.get("segments", [])),
                "total_speakers": len(set(segment.get("speaker", "") for segment in meeting_data.get("segments", []))),
                "transcript_length": sum(len(segment.get("text", "")) for segment in meeting_data.get("segments", []))
            }
        }
        
        # Add audio insights if available
        if audio_analysis:
            report["audio_insights"] = self._extract_audio_insights(audio_analysis)
        
        return report

    def _extract_audio_insights(self, audio_analysis: Dict) -> Dict:
        """
        Extract meaningful insights from audio analysis data.
        """
        insights = {
            "speaking_time_analysis": "Not available",
            "overlap_analysis": "Not available",
            "conversation_flow": "Not available"
        }
        
        # Add actual analysis extraction logic here based on audio_analysis content
        # This would parse the pyannote output to provide meaningful metrics
        
        return insights