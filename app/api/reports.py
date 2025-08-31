import logging
import os
import uuid
from fastapi import APIRouter, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List

from app.core.config import settings
from app.services.meeting_analyzer import MeetingAnalyzer
from app.services.meeting_summarizer import MeetingSummarizer
from app.services.report_generator import ReportGenerator
from app.utils.lru_cache import LRUCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
meeting_analyzer = None
meeting_summarizer = None
report_generator = ReportGenerator()

try:
    meeting_analyzer = MeetingAnalyzer(
        whisper_model="base",
        huggingface_token=settings.HUGGINGFACE_TOKEN
    )
    meeting_summarizer = MeetingSummarizer(
        groq_api_key=settings.GROQ_API_KEY
    )
except Exception as e:
    logger.error(f"Failed to initialize services: {e}")

# Create a temporary directory for file uploads
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize cache for results
cache = LRUCache(capacity=100)


@router.post("/upload")
async def upload_audio(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Upload an audio file, analyze it, and generate PDF and Markdown reports.
    """
    warnings = []
    
    if not meeting_analyzer:
        raise HTTPException(
            status_code=500,
            detail="Audio analysis service is not available."
        )
    
    if not meeting_summarizer:
        warnings.append("Summary service is not available. Basic summary will be used.")

    # Define paths to be used in the finally block
    temp_file_path = None
    output_files = []

    try:
        # Create a unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_file_path = os.path.join(TEMP_DIR, unique_filename)

        # Save the uploaded file to the temporary directory
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"Temporary file saved at: {temp_file_path}")

        # Check if the file was saved correctly
        if not os.path.exists(temp_file_path):
            logger.error(f"File not found after saving: {temp_file_path}")
            raise HTTPException(
                status_code=500, 
                detail="Failed to save uploaded file."
            )

        # Check cache for existing analysis
        cached_item = cache.get(temp_file_path)
        if cached_item:
            meeting_data = cached_item["meeting_data"]
            summary_data = cached_item["summary_data"]
        else:
            # Analyze the meeting audio
            logger.info(f"Analyzing file: {temp_file_path}")
            meeting_data = meeting_analyzer.analyze_meeting(temp_file_path)
            logger.info("Audio analysis successful.")

            # Generate meeting summary
            logger.info("Generating meeting summary")
            summary_data = meeting_summarizer.generate_summary(meeting_data)
            logger.info("Summary generation successful")

            # Cache the results
            cache.put(temp_file_path, {
                "meeting_data": meeting_data,
                "summary_data": summary_data
            })

        # Generate the reports
        pdf_path, md_path = report_generator.generate_report(
            meeting_data,
            summary_data,
            TEMP_DIR
        )
        output_files.extend([pdf_path, md_path])
        logger.info(f"Reports generated at: {pdf_path} and {md_path}")

        # Schedule cleanup in background
        background_tasks.add_task(cleanup_files, [temp_file_path] + output_files)

        # Return the PDF file directly for download
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=f"report_{file.filename}.pdf",
            headers={
                "Content-Disposition": f'attachment; filename="report_{file.filename}.pdf"'
            }
        )

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        # Clean up any files that were created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        for file_path in output_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


def cleanup_files(file_paths: List[str]):
    """Clean up temporary files after they've been sent to the client."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {e}")
