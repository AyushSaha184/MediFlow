"""
src/agents/parser_agent.py
---------------------------
MedicalParserAgent — Phase 1 of the MediFlow pipeline.

Accepts raw bytes + filename for any supported format and returns a
fully-populated MedicalDocumentSchema ready for the next agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.core.base_agent import BaseAgent
from src.models.document_type_map import EXTENSION_TO_DOC_TYPE
from src.models.medical_document import DocumentType, MedicalDocumentSchema
from src.services.extractors import (
    extract_dicom,
    extract_image,
    extract_pdf,
    extract_xlsx,
    extract_zip,
)

# ── Extension routing (single source of truth in document_type_map.py) ──────
_EXTENSION_MAP = EXTENSION_TO_DOC_TYPE

_ALLOWED_MIME_TYPES: set[str] = {
    "application/pdf",
    "application/dicom",
    "image/jpeg",
    "image/png",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/zip",
    "application/x-zip-compressed",
    "application/octet-stream",  # browsers often send DICOM / ZIP as raw bytes
}


def detect_document_type(filename: str) -> DocumentType:
    """
    Infer the document type from its file extension.

    Args:
        filename: The original name of the uploaded file.

    Returns:
        A DocumentType enum value.

    Raises:
        ValueError: If the extension is not in the supported set.
    """
    suffix = Path(filename).suffix.lower()
    doc_type = _EXTENSION_MAP.get(suffix)
    if doc_type is None:
        supported = ", ".join(sorted(_EXTENSION_MAP.keys()))
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported extensions: {supported}"
        )
    return doc_type


class MedicalParserAgent(BaseAgent):
    """
    Entry-point agent for the MediFlow pipeline.

    Dispatches raw file bytes to the appropriate format-specific extractor
    and returns a MedicalDocumentSchema that all downstream agents consume.

    Supported inputs
    ----------------
    - PDF   — medical PDFs (text-based)
    - DICOM — MRI / CT / X-ray studies
    - JPG / PNG — scanned lab reports, handwritten notes (OCR)
    - XLSX  — structured lab-result spreadsheets
    """

    def __init__(self) -> None:
        super().__init__(name="MedicalParserAgent")

    async def run(
        self,
        file_content: bytes,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> MedicalDocumentSchema:
        """
        Parse any supported medical document into a MedicalDocumentSchema.

        Args:
            file_content: Raw bytes of the uploaded file.
            filename: Original filename (used for type detection).
            mime_type: Optional MIME type from the HTTP Content-Type header.

        Returns:
            A populated MedicalDocumentSchema.

        Raises:
            ValueError: On unsupported file type or extraction failure.
        """
        self.logger.info(
            "parsing_started",
            filename=filename,
            size_bytes=len(file_content),
            mime_type=mime_type,
        )

        # Determine document type from extension
        doc_type = detect_document_type(filename)

        # Dispatch to the correct extractor
        try:
            if doc_type == DocumentType.PDF:
                raw_text, metadata, tabular_data = extract_pdf(file_content, filename)

            elif doc_type == DocumentType.DICOM:
                raw_text, metadata, tabular_data = extract_dicom(file_content, filename)

            elif doc_type == DocumentType.IMAGE:
                raw_text, metadata, tabular_data = extract_image(file_content, filename)

            elif doc_type == DocumentType.SPREADSHEET:
                raw_text, metadata, tabular_data = extract_xlsx(file_content, filename)

            elif doc_type == DocumentType.ZIP_DICOM:
                raw_text, metadata, tabular_data = extract_zip(file_content, filename)

            else:
                # Defensive — should never reach here after detect_document_type
                raise ValueError(f"No extractor registered for type '{doc_type}'")

        except ValueError:
            raise
        except Exception as exc:
            self.logger.error("extraction_error", filename=filename, error=str(exc))
            raise ValueError(f"Unexpected extraction error: {exc}") from exc

        self.logger.info(
            "parsing_complete",
            filename=filename,
            doc_type=doc_type.value,
            text_length=len(raw_text),
        )

        return MedicalDocumentSchema(
            document_type=doc_type,
            raw_text=raw_text,
            metadata=metadata,
            tabular_data=tabular_data,
            processed_by=[self.name]
        )
