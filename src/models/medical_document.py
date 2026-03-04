"""
src/models/medical_document.py
-------------------------------
Pydantic schema representing a parsed medical document as it flows through
the MediFlow pipeline. Each downstream agent enriches the same schema object.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported medical document input types."""
    PDF = "pdf"
    DICOM = "dicom"
    IMAGE = "image"              # JPG / PNG lab scans, handwritten notes, etc.
    SPREADSHEET = "spreadsheet"  # XLSX lab reports
    ZIP_DICOM = "zip_dicom"      # ZIP archive containing one or more .dcm files


class MedicalDocumentSchema(BaseModel):
    """
    Canonical data structure that travels through the entire MediFlow pipeline.

    The parser agent populates *raw_text* and *metadata*.
    Downstream agents (Privacy, DataPrep, RAG, …) enrich the remaining fields
    in place without touching already-populated sections.
    """

    # ── Metadata and Lineage ────────────────────────────────────────────────
    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this document instance"
    )
    processed_by: List[str] = Field(
        default_factory=list,
        description="List of agents that have modified or enriched this document"
    )

    # ── Core fields populated by the Parser Agent ───────────────────────────
    document_type: DocumentType = Field(..., description="Detected type of the source document")
    raw_text: str = Field(..., description="Raw textual content extracted from the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Technical metadata (filename, page count, DICOM modality, etc.)"
    )

    # ── Fields populated / refined by downstream agents ─────────────────────
    patient_info: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Structured patient demographic information"
    )
    lab_results: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Extracted laboratory values and reference ranges"
    )
    clinical_notes: Optional[str] = Field(
        default="",
        description="Clinical notes or physician narrative"
    )

    # ── Tabular data for XLSX uploads ────────────────────────────────────────
    tabular_data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Row-level records extracted from spreadsheet uploads"
    )
