from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class MedicalDocumentSchema(BaseModel):
    """
    Structured output schema for the medical document after parsing.
    Currently extracts raw text. Later agents will populate the specific fields.
    """
    patient_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extracted patient information")
    lab_results: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extracted laboratory results")
    clinical_notes: Optional[str] = Field(default="", description="Clinical notes or other medical context")
    raw_text: str = Field(..., description="The raw textual content extracted from the document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata such as page count, source filename")
