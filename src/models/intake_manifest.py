"""
src/models/intake_manifest.py
------------------------------
Pydantic schemas describing the output of the ZIP intake step.

After a user uploads a mixed ZIP archive, the ZipIntakeService
extracts every supported file and organises them on disk by type.
The resulting IntakeManifest is the entry-point for all downstream
agents (Privacy, DataPrep, RAG, …) in a multi-document session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field

from src.models.medical_document import DocumentType


class StagedFile(BaseModel):
    """Represents a single file that has been extracted and staged to disk."""

    original_name: str = Field(..., description="Name of the file inside the ZIP archive")
    staged_path: str = Field(..., description="Absolute path to the file on disk after staging")
    document_type: DocumentType = Field(..., description="Detected type based on file extension")
    size_bytes: int = Field(..., description="Size of the file in bytes")
    file_hash: str = Field(..., description="SHA-256 hash of the file contents to detect duplicates")


class IntakeManifest(BaseModel):
    """
    Full manifest produced by ZipIntakeService for a mixed ZIP upload.

    Downstream agents iterate over `files` (or `by_type`) to process
    each document independently.
    """

    session_id: str = Field(..., description="Unique ID for this intake session (UUID4 short)")
    source_files: List[str] = Field(..., description="Original filenames uploaded by the user")
    staged_root: str = Field(
        ...,
        description="Root directory where all extracted files live: data/<session_id>/"
    )
    total_files: int = Field(..., description="Total number of successfully staged files")
    skipped_files: List[str] = Field(
        default_factory=list,
        description="Files inside the ZIP that were skipped (unsupported extension)"
    )
    files: List[StagedFile] = Field(
        default_factory=list,
        description="All successfully staged files, in extraction order"
    )
    by_type: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Mapping of document type → list of staged file paths. "
            "Keys match DocumentType values (e.g. 'dicom', 'pdf', 'image', 'spreadsheet')"
        )
    )
