"""
src/agents/privacy_agent.py
---------------------------
Phase 2 Agent. Consumes an IntakeManifest (files staged on disk),
extracts their contents using the Phase 1 parsers, and runs them
through the PrivacyService to mask PII/PHI.

Outputs a list of fully populated, anonymised MedicalDocumentSchema objects.
"""

from typing import List, Optional
import time

from src.core.base_agent import BaseAgent
from src.utils.logger import get_logger
from src.models.intake_manifest import IntakeManifest
from src.models.medical_document import MedicalDocumentSchema, DocumentType
from src.services.privacy_service import PrivacyService
from src.agents.parser_agent import MedicalParserAgent
from src.agents.vision_perception_agent import VisionPerceptionAgent

logger = get_logger(__name__)


class PrivacyProtectionAgent(BaseAgent):
    """
    Agent responsible for transforming raw staged files into clean,
    PII-free structured medical documents.
    """

    def __init__(
        self, 
        parser_agent: MedicalParserAgent, 
        privacy_service: PrivacyService,
        vision_agent: Optional[VisionPerceptionAgent] = None
    ):
        super().__init__(
            name="PrivacyAgent"
        )
        self.parser = parser_agent
        self.privacy_service = privacy_service
        self.vision_agent = vision_agent

    async def _process_single_file(self, staged_path: str, original_name: str, doc_type: str, session_id: str) -> Optional[MedicalDocumentSchema]:
        """
        Extracts, validates, and anonymizes a single file.
        """
        try:
            # 1. Read bytes from disk
            with open(staged_path, "rb") as f:
                content = f.read()
                
            # 2. Extract raw data using Phase 1 agent
            # We bypass the /upload endpoint and use the agent's internal .run() method directly
            raw_doc = await self.parser.run(
                file_content=content, 
                filename=original_name, 
                mime_type="application/octet-stream" # The parser doesn't strictly need mime if it has the extension
            )
            
            # 3. Anonymize all textual fields
            start_time = time.time()
            clean_text = self.privacy_service.anonymize_text(raw_doc.raw_text)
            clean_meta = self.privacy_service.anonymize_metadata(raw_doc.metadata)
            clean_tabular = self.privacy_service.anonymize_tabular_data(raw_doc.tabular_data)
            
            anon_time_ms = int((time.time() - start_time) * 1000)
            
            # Anonymize filename (keep extension)
            import os
            name_part, ext_part = os.path.splitext(original_name)
            
            # Presidio NER often misses names connected by underscores or hyphens.
            # Convert them to spaces to give the analyzer the best chance, then scrub.
            name_spaced = name_part.replace("_", " ").replace("-", " ")
            clean_name_spaced = self.privacy_service.anonymize_text(name_spaced)
            
            # Re-join with underscores for a safe filename format
            clean_name_part = clean_name_spaced.replace(" ", "_")
            clean_filename = f"{clean_name_part}{ext_part}"
            
            logger.debug(
                "privacy_agent_file_clean",
                file=original_name,
                clean_file=clean_filename,
                time_ms=anon_time_ms
            )
            
            # 4. Phase 1 Modality Multiplexer -> Phase 1.5 (Vision)
            visual_findings_obj = None
            if self.vision_agent and raw_doc.document_type in [DocumentType.DICOM, DocumentType.IMAGE]:
                # Heuristic: If it's an image but has lots of OCR text, it might just be a scanned lab
                # For this implementation, we route DICOMs and Images to Vision Agent.
                try:
                    vf_data = await self.vision_agent.analyze_image(
                        file_path=staged_path,
                        session_id=session_id
                    )
                    from src.models.medical_document import VisualFinding
                    if "error" not in vf_data:
                        visual_findings_obj = VisualFinding(
                            modality=vf_data.get("modality", "Unknown"),
                            ai_generated_preliminary_report=vf_data.get("ai_generated_preliminary_report", ""),
                            key_observations=vf_data.get("key_observations", []),
                            confidence_score=vf_data.get("confidence_score", 0.5)
                        )
                except Exception as ve:
                    logger.error("vision_agent_failed", file=original_name, error=str(ve))
            
            # 5. Reconstruct clean schema
            # Overwrite the filename in metadata if it exists, or add it.
            clean_meta["source_filename"] = clean_filename
            
            clean_doc = MedicalDocumentSchema(
                document_id=raw_doc.document_id,
                document_type=raw_doc.document_type,
                raw_text=clean_text,
                metadata=clean_meta,
                tabular_data=clean_tabular,
                processed_by=[*raw_doc.processed_by, self.name]
            )
            
            if visual_findings_obj:
                clean_doc.visual_findings = visual_findings_obj
                
            return clean_doc

        except Exception as exc:
            logger.error("privacy_agent_file_failed", file=original_name, error=str(exc))
            # Depending on business logic, we could halt the entire batch, 
            # but usually we want to skip the corrupted file and process the rest.
            return None

    async def run(self, manifest: IntakeManifest) -> List[MedicalDocumentSchema]:
        """
        Process the entire intake batch.
        
        Args:
            manifest: The IntakeManifest produced by Phase 1 (UnifiedBatchIntakeService)
            
        Returns:
            A list of perfectly clean Microsoft Presidio-scrubbed documents.
        """
        self.logger.info(
            "privacy_agent_started",
            session_id=manifest.session_id,
            total_files=manifest.total_files
        )
        
        cleaned_documents: List[MedicalDocumentSchema] = []
        failed_files = 0
        
        for staged_file in manifest.files:
            clean_doc = await self._process_single_file(
                staged_path=staged_file.staged_path,
                original_name=staged_file.original_name,
                doc_type=staged_file.document_type.value,
                session_id=manifest.session_id
            )
            
            if clean_doc:
                cleaned_documents.append(clean_doc)
            else:
                failed_files += 1

        self.logger.info(
            "privacy_agent_completed",
            session_id=manifest.session_id,
            successful=len(cleaned_documents),
            failed=failed_files
        )
        
        return cleaned_documents
