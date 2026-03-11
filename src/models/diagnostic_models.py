"""
src/models/diagnostic_models.py
-------------------------------
Pydantic schemas for the Diagnostic Prediction Agent (Phase 5).
Enforces structured LLM output, urgency levels, missing data lists,
and includes safety validators for critical contraindications.
"""

from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from src.models.medical_document import VisualFinding

class UrgencyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class StructuredDiagnosis(BaseModel):
    primary_diagnosis: str = Field(..., description="The most likely clinical diagnosis.")
    differential_diagnoses: List[str] = Field(..., description="Alternative plausible conditions.")
    supporting_evidence: List[str] = Field(..., description="Direct quotes or chunk references from the patient text.")
    visual_evidence: Optional[VisualFinding] = Field(default=None, description="Visual findings triangulated from Phase 1.5.")
    clinical_visual_congruence: Optional[bool] = Field(default=None, description="Flags True if the image findings match the text labs. False if discordant.")
    urgency_level: UrgencyLevel = Field(..., description="Clinical triage level.")
    missing_data_points: List[str] = Field(..., description="Information or lab codes (e.g., LOINC) needed to increase certainty.")
    contraindications: List[str] = Field(..., description="Critical 'Must Not Dos' based on the diagnosis (e.g., NSAIDs, IV Contrast, specific drugs).")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence metric between 0.0 and 1.0.")

    @model_validator(mode='after')
    def validate_safety_contraindications(self) -> 'StructuredDiagnosis':
        """
        Hard-coded Pydantic Safety Override for known high-risk diagnoses.
        """
        diag_lower = self.primary_diagnosis.lower()
        contra_lower = [c.lower() for c in self.contraindications]
        
        # 1. Acute Kidney Injury (AKI) -> NSAIDs or Contrast
        if "kidney injury" in diag_lower or "aki" in diag_lower:
            if not any("nsaid" in c or "contrast" in c or "ibuprofen" in c for c in contra_lower):
                raise ValueError("Safety Override: 'Acute Kidney Injury' diagnosed but 'NSAIDs' or 'IV Contrast' missing from contraindications.")
                
        # 2. Myocardial Infarction -> (Simple mock check for demonstration)
        # In a real system you would check against a massive rule engine.
        
        # 3. Confidence missing data check
        if self.confidence_score < 0.6 and len(self.missing_data_points) < 3:
            raise ValueError(f"Low confidence ({self.confidence_score}). Agent must list at least 3 missing_data_points to improve certainty.")

        return self


class EvidenceMap(BaseModel):
    statement: str = Field(description="The clinical claim being made.")
    source_chunk_ids: List[str] = Field(description="List of FAISS chunk IDs backing this claim, if any.")
    source_type: Literal["Patient_Record", "Global_Literature", "Inferred_Reasoning"] = Field(description="Where the knowledge came from.")
    is_contradictory: bool = Field(default=False, description="Flag for UI to highlight if this claim contradicts other findings.")
    confidence_of_mapping: float = Field(ge=0.0, le=1.0, description="Confidence that the source backs the statement.")


class FinalDiagnosticReport(BaseModel):
    session_id: str
    clinician_brief: str = Field(description="A highly technical summary for the physician.")
    patient_explanation: str = Field(description="A jargon-free, empathetic explanation for the patient.")
    evidence_table: List[EvidenceMap] = Field(description="Traceability matrix linking every claim to source chunks.")
    citations: List[str] = Field(description="List of AMA/APA formatted citations from Global Literature if used.")
    disclaimer: str = Field(
        default="This is an AI-generated clinical synthesis. Clinical correlation by a licensed physician is strictly required. Do not use this output for final medical decision-making.",
        description="Hard-coded medical legal safety disclaimer."
    )
    structured_diagnosis: StructuredDiagnosis = Field(description="The underlying Phase 5 mathematical/structured diagnosis output.")
    review_status: Literal["completed", "pending_clinician_review", "rejected_by_clinician"] = Field(
        default="completed",
        description="Human review status for this report lifecycle.",
    )
    hitl_review_id: Optional[str] = Field(
        default=None,
        description="Review record ID when clinician review is required.",
    )
    hitl_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons that triggered clinician review.",
    )
    clinician_review_notes: Optional[str] = Field(
        default=None,
        description="Optional reviewer notes captured during HITL approval/rejection.",
    )


class HITLReviewStatus(BaseModel):
    session_id: str
    status: Literal["none", "pending_clinician_review", "approved", "rejected"]
    review_id: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    reviewer_id: Optional[str] = None
    reviewer_notes: Optional[str] = None


class HITLReviewActionRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=1, description="Unique reviewer identifier.")
    notes: Optional[str] = Field(default=None, description="Optional reviewer notes.")
