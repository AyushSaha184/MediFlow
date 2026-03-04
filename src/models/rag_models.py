"""
Pydantic request/response models for Phase 4 dual-store RAG endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.medical_document import MedicalDocumentSchema


class RAGIndexPatientRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Session identifier used for patient-store isolation.")
    documents: List[MedicalDocumentSchema] = Field(default_factory=list)


class RAGIndexPatientResponse(BaseModel):
    session_id: str
    chunks_seen: int
    chunks_embedded: int
    documents_indexed: int


class RAGRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default=None)
    top_k_patient: int = Field(default=5, ge=1, le=100)
    top_k_global: int = Field(default=5, ge=1, le=100)
    top_k_total: int = Field(default=8, ge=1, le=100)


class RAGRetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str = ""
    source_file: str
    origin: str
    section: str
    text: str
    session_id: Optional[str] = None
    distance: float
    l2_distance: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGRetrieveResponse(BaseModel):
    query: str
    results: List[RAGRetrievedChunk] = Field(default_factory=list)


class RAGCleanupResponse(BaseModel):
    session_id: str
    deleted: bool
