from typing import Dict
import os
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from datetime import datetime

class ReportGenerator:
    def generate_report(self, meeting_data: Dict, summary_data: Dict, output_dir: str) -> tuple:
        """
        Generate both PDF and Markdown reports from the meeting data.
        Returns tuple of (pdf_path, markdown_path)
        """
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"meeting_report_{timestamp}"
        
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        md_path = os.path.join(output_dir, f"{base_filename}.md")
        
        # Generate both formats
        self._generate_pdf(meeting_data, summary_data, pdf_path)
        self._generate_markdown(meeting_data, summary_data, md_path)
        
        return pdf_path, md_path

    def _generate_pdf(self, meeting_data: Dict, summary_data: Dict, output_path: str):
        """
        Generate a well-formatted PDF report.
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            spaceAfter=16,
            spaceBefore=16
        ))

        # Title and Meeting Info
        story.append(Paragraph("Meeting Report", styles['Heading1']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Paragraph(f"Language: {meeting_data['language']}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Summary sections
        story.append(Paragraph("Executive Summary", styles['SectionHeader']))
        story.append(Paragraph(summary_data['overview'], styles['Normal']))
        story.append(Spacer(1, 12))

        # Topics
        story.append(Paragraph("Topics Discussed", styles['SectionHeader']))
        for topic in summary_data['topics']:
            story.append(Paragraph(f"• {topic}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Decisions
        story.append(Paragraph("Decisions Made", styles['SectionHeader']))
        for decision in summary_data['decisions']:
            story.append(Paragraph(f"• {decision}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Action Items
        story.append(Paragraph("Action Items", styles['SectionHeader']))
        for action in summary_data['action_items']:
            story.append(Paragraph(f"• {action}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Follow-up Items
        if summary_data['follow_up']:
            story.append(Paragraph("Follow-up Items", styles['SectionHeader']))
            for item in summary_data['follow_up']:
                story.append(Paragraph(f"• {item}", styles['Normal']))
            story.append(Spacer(1, 12))

        # Full Transcript
        story.append(Paragraph("Full Transcript", styles['SectionHeader']))
        for segment in meeting_data['segments']:
            speaker_text = f"{segment['speaker']}: {segment['text']}"
            story.append(Paragraph(speaker_text, styles['Normal']))
            story.append(Spacer(1, 6))

        doc.build(story)

    def _generate_markdown(self, meeting_data: Dict, summary_data: Dict, output_path: str):
        """
        Generate a markdown version of the report.
        """
        md_content = [
            "# Meeting Report\n",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"Language: {meeting_data['language']}\n",
            "## Executive Summary\n",
            f"{summary_data['overview']}\n",
            "## Topics Discussed\n"
        ]

        for topic in summary_data['topics']:
            md_content.append(f"* {topic}\n")

        md_content.extend(["\n## Decisions Made\n"])
        for decision in summary_data['decisions']:
            md_content.append(f"* {decision}\n")

        md_content.extend(["\n## Action Items\n"])
        for action in summary_data['action_items']:
            md_content.append(f"* {action}\n")

        if summary_data['follow_up']:
            md_content.extend(["\n## Follow-up Items\n"])
            for item in summary_data['follow_up']:
                md_content.append(f"* {item}\n")

        md_content.extend(["\n## Full Transcript\n"])
        for segment in meeting_data['segments']:
            md_content.append(f"**{segment['speaker']}**: {segment['text']}\n\n")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("".join(md_content))
