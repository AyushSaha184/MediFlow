"""
src/agents/diagnostic_agent.py
------------------------------
Phase 5: Diagnostic Prediction Agent.
Consumes Patient Info, Extracted Labs, and RAG context to formulate a diagnosis.
Implements the "Clinical Verification" Loop (Drafting + Devil's Advocate).
"""

import json
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from src.core.base_agent import BaseAgent
from src.models.medical_document import MedicalDocumentSchema
from src.models.diagnostic_models import StructuredDiagnosis
from src.services.llm_service import LLMService
from src.services.numerical_extractor import NumericalGuardrailsExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DiagnosticAgent(BaseAgent):
    def __init__(self, llm_service: LLMService, extractor: NumericalGuardrailsExtractor):
        super().__init__("DiagnosticAgent")
        self.llm = llm_service
        self.extractor = extractor

    def _build_system_prompt(self, pass_type: str) -> str:
        base = (
            "You are the MediFlow Diagnostic Prediction Agent, an expert clinical AI.\n"
            "You must synthesize patient data, numerical lab values, visual findings (Phase 1.5), and retrieved medical evidence into a strictly formatted JSON response.\n"
            "RULES:\n"
            "- If you see speculative language ('suspected', 'rule out'), DO NOT treat it as confirmed fact.\n"
            "- Pay attention to temporal trends (e.g., dropping vs. stable values).\n"
            "- If symptoms are severe but labs are normal (Symptom-Lab Mismatch), defaults Urgency to High.\n"
            "- **CONFLICT ALERT (Clinical-Visual Mismatch)**: If Phase 1.5 Visual Findings contradict Phase 3 Labs (e.g., Image says 'Pneumonia', Labs say 'Normal WBC'), you MUST set `clinical_visual_congruence=False` and prioritize exploring the discordance in your primary or differential diagnosis.\n"
            "- If your confidence is < 0.6, you MUST list at least 3 missing data points.\n"
            "- Ensure critical contraindications are listed (e.g., AKI -> NSAIDs).\n"
        )
        
        schema_def = (
            "Your output MUST be a JSON object matching this schema exactly:\n"
            "{\n"
            '  "primary_diagnosis": "string",\n'
            '  "differential_diagnoses": ["string", "string"],\n'
            '  "supporting_evidence": ["string"],\n'
            '  "visual_evidence": { /* ... */ } | null,\n'
            '  "clinical_visual_congruence": true | false | null,\n'
            '  "urgency_level": "Low" | "Medium" | "High" | "Critical",\n'
            '  "missing_data_points": ["string"],\n'
            '  "contraindications": ["string"],\n'
            '  "confidence_score": float\n'
            "}\n"
        )

        if pass_type == "draft":
            return base + schema_def + "\nGenerate the initial clinical hypothesis based on the provided evidence."
        elif pass_type == "devil_advocate":
            return base + schema_def + (
                "\nCRITICAL CORRECTION PASS:\n"
                "You are playing Devil's Advocate against a proposed preliminary diagnosis.\n"
                "Focus on finding evidence for the top two differential diagnoses that would DISPROVE the primary diagnosis.\n"
                "Output an updated, revised JSON object that incorporates your corrections and skepticism."
            )
        return ""

    def _build_user_prompt(self, doc: MedicalDocumentSchema, rag_context: List[Dict[str, Any]], draft_json: Optional[str] = None) -> str:
        # Pre-process numbers into Markdown table (multi-lab historical view)
        labs_md = self.extractor.process_historical_context(
            current_text=doc.normalized_text or doc.raw_text,
            current_ts=doc.document_timestamp,
            rag_chunks=rag_context
        )
        
        context_str = "\n".join([str(c) for c in rag_context])

        prompt = (
            f"--- PATIENT DEMOGRAPHICS & NOTES ---\n{doc.patient_info}\n\n"
            f"--- CLINICAL TEXT ---\n{doc.normalized_text or doc.raw_text}\n\n"
            f"--- NUMERICAL GUARDRAILS (LABS) ---\n{labs_md}\n\n"
            f"--- VISUAL FINDINGS (Phase 1.5) ---\n{doc.visual_findings.model_dump_json() if doc.visual_findings else 'None'}\n\n"
            f"--- RETRIEVED MEDICAL EVIDENCE & HISTORY (Decay Weighted) ---\n{context_str}\n\n"
        )

        if draft_json:
            prompt += (
                f"--- DRAFT DIAGNOSIS to CRITIQUE ---\n{draft_json}\n\n"
                "Review the above draft. Find flaws, contradictions, or missing contraindications. Provide the finalized JSON."
            )

        return prompt

    async def _safe_generate(self, doc: MedicalDocumentSchema, rag_context: List[Dict[str, Any]]) -> StructuredDiagnosis:
        # Pass 1: Drafting (Temp 0.1)
        self.logger.info("starting_pass_1_drafting")
        system_1 = self._build_system_prompt("draft")
        user_1 = self._build_user_prompt(doc, rag_context)
        
        try:
            draft_str = await self.llm.generate_json(system_1, user_1, temperature=0.1)
            # Basic validation check
            StructuredDiagnosis.model_validate_json(draft_str)
        except ValidationError as e:
            self.logger.warning("pass_1_validation_failed", error=str(e))
            # In production, we'd trigger a retry here. For now, we continue to Pass 2 to let it self-correct.
            draft_str = draft_str if draft_str else "{}"

        # Pass 2: Devil's Advocate (Temp 0.4)
        self.logger.info("starting_pass_2_devils_advocate")
        system_2 = self._build_system_prompt("devil_advocate")
        user_2 = self._build_user_prompt(doc, rag_context, draft_json=draft_str)
        
        final_str = await self.llm.generate_json(system_2, user_2, temperature=0.4)
        
        # Pydantic Structural Validation & Safety Overrides (e.g. AKI vs NSAIDs)
        final_obj = StructuredDiagnosis.model_validate_json(final_str)
        return final_obj

    async def run(self, document: MedicalDocumentSchema, rag_context: List[Dict[str, Any]] = None) -> StructuredDiagnosis:
        """
        Orchestrates the diagnostic verification loop.
        """
        if rag_context is None:
            rag_context = []
            
        try:
            diagnosis = await self._safe_generate(document, rag_context)
            
            # Lineage tracking
            if self.name not in document.processed_by:
                document.processed_by.append(self.name)
                
            return diagnosis
            
        except Exception as e:
            self.logger.error("diagnostic_agent_failed", error=str(e))
            raise
