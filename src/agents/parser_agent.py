import io
import fitz  # PyMuPDF
from typing import Any

from src.core.base_agent import BaseAgent
from src.models.medical_document import MedicalDocumentSchema


class MedicalParserAgent(BaseAgent):
    """
    Agent responsible for converting uploaded medical PDFs into structured text.
    """
    def __init__(self):
        super().__init__(name="MedicalParserAgent")

    async def run(self, file_content: bytes, filename: str) -> MedicalDocumentSchema:
        """
        Extracts text from a PDF byte stream and returns a structured document.
        
        Args:
            file_content: The bytes content of the PDF.
            filename: The original filename context.
            
        Returns:
            MedicalDocumentSchema containing the raw text and metadata.
        """
        self.logger.info("starting_parsing", filename=filename, size=len(file_content))
        
        raw_text = ""
        page_count = 0
        
        try:
            # We open the PDF from memory using PyMuPDF
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                page_count = doc.page_count
                for page_num in range(page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    raw_text += text + "\n"
            
            self.logger.info("parsing_complete", filename=filename, pages=page_count)
            
            return MedicalDocumentSchema(
                raw_text=raw_text.strip(),
                metadata={"filename": filename, "page_count": page_count}
            )

        except Exception as e:
            self.logger.error("parsing_failed", filename=filename, error=str(e))
            raise ValueError(f"Failed to parse PDF document: {str(e)}") from e
