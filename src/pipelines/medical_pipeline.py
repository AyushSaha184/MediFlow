"""
src/pipelines/medical_pipeline.py
---------------------------------
Phase 7: The Master Orchestrator.
Wires all agents together sequentially. Receives an IntakeManifest and 
returns a FinalDiagnosticReport.
"""

from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

from src.utils.logger import get_logger
from src.models.intake_manifest import IntakeManifest
from src.models.diagnostic_models import FinalDiagnosticReport, StructuredDiagnosis

# Agents
from src.agents.privacy_agent import PrivacyProtectionAgent
from src.agents.data_prep_agent import DataPrepAgent
from src.agents.medical_rag_agent import MedicalRAGAgent
from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.services.hitl_review_service import HITLReviewService

logger = get_logger(__name__)

# Basic retry policy for LLM / external API calls
# Retries up to 3 times on httpx.RequestError (timeouts, connection drops)
# Waits 2^x * 1 second between each retry
API_RETRY_POLICY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.RequestError),
    reraise=True
)

class MedicalPipeline:
    """
    Coordinates the full 8-phase lifecycle of a Patient Intake Session.
    """
    def __init__(
        self,
        privacy_agent: PrivacyProtectionAgent,
        data_prep_agent: DataPrepAgent,
        rag_agent: MedicalRAGAgent,
        diagnostic_agent: DiagnosticAgent,
        explainability_agent: ExplainabilityAgent,
        hitl_review_service: HITLReviewService,
    ):
        self.privacy_agent = privacy_agent
        self.data_prep_agent = data_prep_agent
        self.rag_agent = rag_agent
        self.diagnostic_agent = diagnostic_agent
        self.explainability_agent = explainability_agent
        self.hitl_review_service = hitl_review_service

    @API_RETRY_POLICY
    async def _safe_rag_retrieve(self, unified_text: str, session_id: str) -> List[Dict[str, Any]]:
        """Wraps RAG retrieval with resilience."""
        return await self.rag_agent.retrieve(query=unified_text, session_id=session_id)

    @API_RETRY_POLICY
    async def _safe_diagnostic_run(self, docs: List[Any], rag_context: List[Dict[str, Any]]) -> Any:
        """Wraps diagnostic prediction with resilience."""
        # The diagnostic agent currently takes a single document in scaffolding, 
        # but in a real pipeline we merge them or operate on a summary concept.
        # For this E2E, we'll run diagnosis on the primary (or first) clinical note,
        # but pass the combined tabular/visual findings.
        
        # Merge Visual Findings from all docs into the primary doc
        primary_doc = docs[0] 
        for doc in docs[1:]:
            if doc.visual_findings and not primary_doc.visual_findings:
                primary_doc.visual_findings = doc.visual_findings
            
            # Append tabular data
            if doc.tabular_data:
                for k, v in doc.tabular_data.items():
                    if k not in primary_doc.tabular_data:
                        primary_doc.tabular_data[k] = v
                        
        return await self.diagnostic_agent.run(document=primary_doc, rag_context=rag_context)

    @API_RETRY_POLICY
    async def _safe_explainability_run(self, diagnosis: Any, rag_context: List[Dict[str, Any]], session_id: str) -> FinalDiagnosticReport:
        return await self.explainability_agent.run(diagnosis=diagnosis, rag_context=rag_context, session_id=session_id)

    @staticmethod
    def _pending_review_report(
        session_id: str,
        diagnosis: StructuredDiagnosis,
        review_id: str,
        reasons: List[str],
    ) -> FinalDiagnosticReport:
        return FinalDiagnosticReport(
            session_id=session_id,
            clinician_brief="Awaiting clinician approval due to safety-triggered review conditions.",
            patient_explanation=(
                "Your case is being reviewed by a clinician before the final AI summary is released."
            ),
            evidence_table=[],
            citations=[],
            structured_diagnosis=diagnosis,
            review_status="pending_clinician_review",
            hitl_review_id=review_id,
            hitl_reasons=reasons,
        )

    async def finalize_after_hitl_approval(self, session_id: str) -> FinalDiagnosticReport:
        payload = self.hitl_review_service.get_approved_payload(session_id=session_id)
        if not payload:
            raise ValueError("No approved HITL payload found for this session.")

        diagnosis = StructuredDiagnosis.model_validate(payload.get("diagnosis", {}))
        rag_context = payload.get("rag_context", [])
        report = await self._safe_explainability_run(
            diagnosis=diagnosis,
            rag_context=rag_context,
            session_id=session_id,
        )
        report.review_status = "completed"
        report.hitl_review_id = payload.get("review_id")
        report.hitl_reasons = payload.get("reasons", [])
        report.clinician_review_notes = payload.get("reviewer_notes")
        return report
        
    async def analyze_session(self, manifest: IntakeManifest) -> FinalDiagnosticReport:
        """
        Executes the full pipeline for a given manifest.
        """
        session_id = manifest.session_id
        logger.info("pipeline_started", session_id=session_id)
        
        try:
            # 1. Phase 1 & 1.5 & 2: Extraction, Vision, Privacy
            logger.info("pipeline_phase_privacy_vision_started", session_id=session_id)
            clean_docs = await self.privacy_agent.run(manifest=manifest)
            
            if not clean_docs:
                raise ValueError("All documents failed during Privacy/Vision extraction.")
                
            # 2. Phase 3: Data Preparation (Normalization & Chunking)
            logger.info("pipeline_phase_data_prep_started", session_id=session_id)
            prepared_docs = await self.data_prep_agent.run(documents=clean_docs)
            
            # 3. Phase 4: RAG Ingestion & Retrieval
            logger.info("pipeline_phase_rag_started", session_id=session_id)
            # Ingest all prepared patient chunks into the session-local pgvector store
            await self.rag_agent.ingest_patient_documents(
                session_id=session_id,
                documents=prepared_docs,
            )
                    
            # Generate a unified query string from the normalized text to search FAISS
            unified_query = " ".join([d.normalized_text for d in prepared_docs if d.normalized_text])[:1000] # Limit query size
            rag_context = await self._safe_rag_retrieve(unified_text=unified_query, session_id=session_id)
            
            # 4. Phase 5: Diagnostic Prediction
            logger.info("pipeline_phase_diagnostic_started", session_id=session_id)
            structured_diagnosis = await self._safe_diagnostic_run(docs=prepared_docs, rag_context=rag_context)

            hitl_reasons = self.hitl_review_service.evaluate_gate(
                diagnosis=structured_diagnosis,
                rag_context=rag_context,
            )
            if hitl_reasons:
                status = self.hitl_review_service.create_pending_review(
                    session_id=session_id,
                    diagnosis=structured_diagnosis,
                    rag_context=rag_context,
                    reasons=hitl_reasons,
                )
                logger.warning(
                    "pipeline_paused_for_hitl_review",
                    session_id=session_id,
                    review_id=status.review_id,
                    reasons=hitl_reasons,
                )
                return self._pending_review_report(
                    session_id=session_id,
                    diagnosis=structured_diagnosis,
                    review_id=status.review_id or "",
                    reasons=hitl_reasons,
                )
            
            # 5. Phase 6: Explainability & Final Report
            logger.info("pipeline_phase_explainability_started", session_id=session_id)
            final_report = await self._safe_explainability_run(
                diagnosis=structured_diagnosis, 
                rag_context=rag_context, 
                session_id=session_id
            )
            final_report.review_status = "completed"
            
            logger.info("pipeline_completed_successfully", session_id=session_id)
            return final_report
            
        except Exception as e:
            logger.error("pipeline_failed_critical", session_id=session_id, error=str(e), exc_info=True)
            raise RuntimeError(f"Pipeline failed for session {session_id}: {str(e)}") from e
