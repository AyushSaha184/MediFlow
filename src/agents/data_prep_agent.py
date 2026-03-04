"""
src/agents/data_prep_agent.py
-----------------------------
Phase 3 Agent. Consumes a list of MedicalDocumentSchema objects
(already parsed and privacy-scrubbed) and prepares them for RAG and
LLM inference.

1. Expands medical acronyms (TerminologyService)
2. Standardizes lab units (TerminologyService)
3. Chunks the text logically (ChunkingService)
"""

from typing import List

from src.core.base_agent import BaseAgent
from src.utils.logger import get_logger
from src.models.medical_document import MedicalDocumentSchema
from src.services.terminology_service import TerminologyService
from src.services.chunking_service import ChunkingService

logger = get_logger(__name__)


class DataPrepAgent(BaseAgent):
    """
    Transforms clean, parsed text into LLM-ready standardized strings
    and embedding-ready semantic chunks.
    """

    def __init__(self, terminology: TerminologyService, chunker: ChunkingService):
        super().__init__(name="DataPrepAgent")
        self.terminology = terminology
        self.chunker = chunker

    async def run(self, documents: List[MedicalDocumentSchema]) -> List[MedicalDocumentSchema]:
        """
        Processes a batch of MedicalDocumentSchema objects in place.
        """
        self.logger.info("data_prep_started", target_count=len(documents))
        
        for doc in documents:
            # Document must have raw_text populated by Phase 1
            if not doc.raw_text:
                self.logger.warning("data_prep_skip_empty_doc", document_id=doc.document_id)
                continue
                
            try:
                # 1. Normalize Terminology & Units
                norm_text = self.terminology.normalize(doc.raw_text)
                doc.normalized_text = norm_text
                
                # 2. Smart Chunks
                chunks = self.chunker.chunk_document(norm_text)
                doc.chunks = chunks
                
                # 3. Append to lineage
                if self.name not in doc.processed_by:
                    doc.processed_by.append(self.name)
                
                self.logger.debug(
                    "data_prep_doc_complete",
                    document_id=doc.document_id,
                    chunks_created=len(chunks)
                )
                
            except Exception as exc:
                self.logger.error(
                    "data_prep_failed",
                    document_id=doc.document_id,
                    error=str(exc)
                )
                # Fail gracefully for individual docs
                continue
                
        self.logger.info("data_prep_completed", successful_count=len(documents))
        return documents
