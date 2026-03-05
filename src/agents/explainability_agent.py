"""
src/agents/explainability_agent.py
----------------------------------
Phase 6: Explainability Agent.
Transforms the Phase 5 StructuredDiagnosis into a highly verifiable,
trustworthy, patient-friendly FinalDiagnosticReport.
Applies Citation Validation and Traceability mappings.
"""

from typing import List, Dict, Any, Optional
import json
from pydantic import ValidationError

from src.core.base_agent import BaseAgent
from src.models.medical_document import MedicalDocumentSchema
from src.models.diagnostic_models import StructuredDiagnosis, FinalDiagnosticReport, EvidenceMap
from src.services.llm_service import LLMService
from src.services.explanation_service import ExplanationService


class ExplainabilityAgent(BaseAgent):
    def __init__(self, llm_service: LLMService, explanation_service: ExplanationService):
        super().__init__("ExplainabilityAgent")
        self.llm = llm_service
        self.explanation = explanation_service

    def _build_system_prompt(self, valid_chunk_ids: List[str]) -> str:
        # Pass the allowed chunks so LLM uses exactly these IDs
        valid_ids_str = ", ".join(valid_chunk_ids) if valid_chunk_ids else "None"
        
        return (
            "You are the MediFlow Explainability Agent. Your job is to translate technical Phase 5 predictions "
            "into a highly traceable, structured clinical report.\n\n"
            "CRITICAL PROTOCOLS:\n"
            "1. **Evidence Traceability**: Every claim you make in the `evidence_table` MUST cite at least one `source_chunk_id` "
            "from the provided context. If it is inferred, use an empty list for `source_chunk_ids` and set `source_type`="
            f"'Inferred_Reasoning'. Allowed FAISS Chunk IDs: [{valid_ids_str}]\n"
            "2. **Circular Citation Blocking**: Do NOT cite the Phase 5 output itself as evidence. You must cite the underlying RAG context.\n"
            "3. **Contradictory Findings**: If a RAG chunk challenges the diagnosis, you MUST include it in the `evidence_table` "
            "with `is_contradictory=True`.\n"
            "4. **Patient Explanation**: Must be empathetic, jargon-free point-of-view (use 'you', not 'the patient').\n\n"
            "OUTPUT SCHEMA:\n"
            "Return a strictly formatted JSON object matching this schema exactly:\n"
            "{\n"
            '  "session_id": "string",\n'
            '  "clinician_brief": "string",\n'
            '  "patient_explanation": "string",\n'
            '  "evidence_table": [\n'
            '    {\n'
            '      "statement": "string",\n'
            '      "source_chunk_ids": ["string"],\n'
            '      "source_type": "Patient_Record" | "Global_Literature" | "Inferred_Reasoning",\n'
            '      "is_contradictory": boolean,\n'
            '      "confidence_of_mapping": float\n'
            '    }\n'
            '  ],\n'
            '  "citations": ["string"]\n'
            "}"
        )

    def _build_user_prompt(self, diagnosis: StructuredDiagnosis, rag_context: List[Dict[str, Any]], session_id: str) -> str:
        # Format Phase 5 Output
        diag_json = diagnosis.model_dump_json(indent=2)
        
        # Format FAISS Chunks so LLM sees IDs
        chunks_md = ""
        for chunk in rag_context:
            c_id = chunk.get("chunk_id", "unknown")
            text = chunk.get("text", "")
            chunks_md += f"[{c_id}]: {text}\n"

        return (
            f"--- PHASE 5 STRUCTURED DIAGNOSIS ---\n{diag_json}\n\n"
            f"--- UNDERLYING EVIDENCE (FAISS CHUNKS) ---\n{chunks_md}\n\n"
            f"TASK: Generate the FinalDiagnosticReport for session: {session_id}.\n"
            "Ensure the patient_explanation is compassionate but highly accurate."
        )

    def _validate_citations(self, report_data: Dict[str, Any], valid_chunk_ids: List[str]) -> Dict[str, Any]:
        """
        Hard guardrail: Strips out any hallucinated chunk IDs from the evidence table.
        """
        evidence_table = report_data.get("evidence_table", [])
        for evidence in evidence_table:
            claimed_ids = evidence.get("source_chunk_ids", [])
            valid_claimed_ids = [cid for cid in claimed_ids if cid in valid_chunk_ids]
            
            # If all ids were hallucinated, flag it as inferred
            if claimed_ids and not valid_claimed_ids:
                self.logger.warning("explainability_hallucinated_citation_stripped", claimed=claimed_ids)
                evidence["source_type"] = "Inferred_Reasoning"
                
            evidence["source_chunk_ids"] = valid_claimed_ids
            
        return report_data

    async def run(self, diagnosis: StructuredDiagnosis, rag_context: List[Dict[str, Any]], session_id: str) -> FinalDiagnosticReport:
        self.logger.info("explainability_agent_started", session_id=session_id)
        
        # 1. Prepare valid chunk IDs for Citation Validator
        valid_chunk_ids = [c.get("chunk_id", "") for c in rag_context if "chunk_id" in c]
        
        system_prompt = self._build_system_prompt(valid_chunk_ids)
        user_prompt = self._build_user_prompt(diagnosis, rag_context, session_id)
        
        # 2. LLM Generation
        raw_json = await self.llm.generate_json(system_prompt, user_prompt, temperature=0.2)
        
        try:
            report_data = json.loads(raw_json)
        except Exception as e:
            self.logger.error("explainability_json_parse_failed", error=str(e))
            raise ValueError(f"Failed to parse LLM JSON output: {e}")
            
        # 3. Apply Safety Guardrails (Citation Validator)
        report_data = self._validate_citations(report_data, valid_chunk_ids)
        
        # 4. Post-Process Patient Explanation (Hedge-Words & Jargon)
        pat_exp = report_data.get("patient_explanation", "")
        # Optional: Add Narrative Sparklines if we had access to lab trends
        # For now, apply jargon substitution and hedge words
        pat_exp = self.explanation.inject_hedge_words(pat_exp)
        pat_exp = self.explanation.reverse_terminology_lookup(pat_exp)
        report_data["patient_explanation"] = pat_exp
        
        # Set required fields ensuring defaults
        report_data["session_id"] = session_id
        # We attach the original Phase 5 output as the backend truth
        report_data["structured_diagnosis"] = diagnosis.model_dump()
        
        try:
            final_report = FinalDiagnosticReport(**report_data)
            self.logger.info("explainability_agent_completed", session_id=session_id)
            return final_report
        except ValidationError as e:
            self.logger.error("explainability_validation_failed", error=str(e))
            raise
