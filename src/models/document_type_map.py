"""
src/models/document_type_map.py
--------------------------------
Single source of truth for the file extension → DocumentType mapping.
Imported by both parser_agent.py and zip_intake.py to avoid duplication.
"""

from __future__ import annotations

from src.models.medical_document import DocumentType

EXTENSION_TO_DOC_TYPE: dict[str, DocumentType] = {
    ".pdf":   DocumentType.PDF,
    ".dcm":   DocumentType.DICOM,
    ".dicom": DocumentType.DICOM,
    ".jpg":   DocumentType.IMAGE,
    ".jpeg":  DocumentType.IMAGE,
    ".png":   DocumentType.IMAGE,
    ".xlsx":  DocumentType.SPREADSHEET,
    ".zip":   DocumentType.ZIP_DICOM,   # ZIP is a special routing type
}
