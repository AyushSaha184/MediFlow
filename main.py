from contextlib import asynccontextmanager
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from src.core.config import settings
from src.utils.logger import setup_logging, get_logger
from src.agents.parser_agent import MedicalParserAgent, _ALLOWED_MIME_TYPES
from src.models.medical_document import MedicalDocumentSchema
from src.models.intake_manifest import IntakeManifest
from src.services.zip_intake import UnifiedBatchIntakeService, UploadedItem
from src.services.privacy_service import PrivacyService
from src.agents.privacy_agent import PrivacyProtectionAgent
from src.services.terminology_service import TerminologyService
from src.services.chunking_service import ChunkingService
from src.services.session_context import atomic_session_lock
from src.agents.vision_perception_agent import VisionPerceptionAgent
from src.agents.data_prep_agent import DataPrepAgent
from src.rag.embedding_service import EmbeddingService
from src.agents.medical_rag_agent import MedicalRAGAgent
from src.models.rag_models import (
    RAGCleanupResponse,
    RAGIndexPatientRequest,
    RAGIndexPatientResponse,
    RAGRetrieveRequest,
    RAGRetrieveResponse,
)
from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.services.llm_service import LLMService
from src.services.explanation_service import ExplanationService
from src.services.numerical_extractor import NumericalGuardrailsExtractor
from src.pipelines.medical_pipeline import MedicalPipeline
from src.models.diagnostic_models import FinalDiagnosticReport

setup_logging()
logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE: int = 1 * 1024 * 1024 * 1024  # 1 GB

# ── Service / Agent singletons ────────────────────────────────────────────────
parser_agent = MedicalParserAgent()
batch_intake_service = UnifiedBatchIntakeService()

privacy_service: PrivacyService = None
privacy_agent: PrivacyProtectionAgent = None

terminology_service: TerminologyService = None
chunking_service: ChunkingService = None
data_prep_agent: DataPrepAgent = None
rag_embedding_service: EmbeddingService = None
medical_rag_agent: MedicalRAGAgent = None
vision_agent: VisionPerceptionAgent = None
llm_service: LLMService = None
diagnostic_agent: DiagnosticAgent = None
explanation_service: ExplanationService = None
explainability_agent: ExplainabilityAgent = None
medical_pipeline: MedicalPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global privacy_service, privacy_agent, terminology_service, chunking_service, data_prep_agent
    global rag_embedding_service, medical_rag_agent, vision_agent
    global llm_service, diagnostic_agent, explanation_service, explainability_agent, medical_pipeline
    logger.info("Application starting up", project=settings.project_name, env=settings.environment)
    # Lazy load the heavy NLP presidio models on startup
    privacy_service = PrivacyService()
    vision_agent = VisionPerceptionAgent()
    privacy_agent = PrivacyProtectionAgent(parser_agent=parser_agent, privacy_service=privacy_service, vision_agent=vision_agent)
    
    # Initialize Data Prep services
    terminology_service = TerminologyService()
    chunking_service = ChunkingService(target_chunk_size=1500, overlap=200)
    data_prep_agent = DataPrepAgent(terminology=terminology_service, chunker=chunking_service)

    rag_embedding_service = EmbeddingService(
        provider=settings.rag_embedding_provider,
        model_name=settings.rag_embedding_model_name,
        fallback_dimension=settings.rag_embedding_fallback_dimension,
        local_files_only=settings.rag_embedding_local_files_only,
        nvidia_api_url=settings.rag_embedding_nvidia_api_url,
        nvidia_api_key=settings.rag_embedding_nvidia_api_key,
        nvidia_truncate=settings.rag_embedding_nvidia_truncate,
        request_timeout_seconds=settings.rag_embedding_request_timeout_seconds,
        nvidia_max_batch_size=settings.rag_embedding_nvidia_max_batch_size,
    )
    llm_service = LLMService(api_key=settings.cerebras_api_key)
    medical_rag_agent = MedicalRAGAgent(
        embedder=rag_embedding_service,
        global_store_dir=Path(settings.rag_global_store_dir),
        patient_data_root=Path(settings.rag_patient_data_root),
        llm_service=llm_service,
    )
    
    # Initialize Phase 5 & 6 Agents — reuse the same llm_service instance already created above
    numerical_extractor = NumericalGuardrailsExtractor()
    diagnostic_agent = DiagnosticAgent(llm_service=llm_service, extractor=numerical_extractor)
    explanation_service = ExplanationService(terminology_service=terminology_service)
    explainability_agent = ExplainabilityAgent(llm_service=llm_service, explanation_service=explanation_service)
    
    # Initialize Phase 7 Pipeline Orchestrator
    medical_pipeline = MedicalPipeline(
        privacy_agent=privacy_agent,
        data_prep_agent=data_prep_agent,
        rag_agent=medical_rag_agent,
        diagnostic_agent=diagnostic_agent,
        explainability_agent=explainability_agent
    )
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title=settings.project_name,
    description="Multi-agent Medical AI system pipeline.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class HealthCheckResponse(BaseModel):
    status: str
    environment: str

class SessionCreateResponse(BaseModel):
    session_id: str


# ── Routes ───────────────────────────────────────────────────────────────────

def _require_rag_agent() -> MedicalRAGAgent:
    if medical_rag_agent is None:
        raise HTTPException(status_code=503, detail="MedicalRAGAgent is not initialized.")
    return medical_rag_agent

def _require_pipeline() -> MedicalPipeline:
    if medical_pipeline is None:
        raise HTTPException(status_code=503, detail="MedicalPipeline is not initialized.")
    return medical_pipeline


@app.post("/session/create", response_model=SessionCreateResponse, tags=["Session"])
async def create_session() -> SessionCreateResponse:
    """
    Create a new isolated session. Returns a UUID that the frontend
    must attach to every subsequent request as the session_id.
    """
    return SessionCreateResponse(session_id=str(uuid.uuid4()))


@app.get("/health", response_model=HealthCheckResponse, tags=["Utility"])
async def health_check() -> HealthCheckResponse:
    """
    Basic health check. Returns 200 + status when the service is alive.
    """
    logger.debug("Health check requested")
    return HealthCheckResponse(status="ok", environment=settings.environment)

@app.post("/analyze-medical-session/{session_id}", response_model=FinalDiagnosticReport, tags=["Pipeline"])
async def analyze_medical_session(session_id: str) -> FinalDiagnosticReport:
    """
    Phase 7 Orchestrator: Runs the full 8-phase lifecycle for a staged Intake session.
    Takes a session_id (e.g., from /intake-batch), loads the manifest, and executes:
    Privacy (Phase 2), Data Prep (Phase 3), RAG (Phase 4), Diagnosis (Phase 5), and Explainability (Phase 6).
    """
    pipeline = _require_pipeline()
    
    # Reconstruct the manifest from the staging directory logic
    # (In a real app, you'd load this from a DB or session store)
    manifest_path = batch_intake_service.data_root / session_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or expired.")
        
    try:
        manifest = IntakeManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        final_report = await pipeline.analyze_session(manifest)
        return final_report
    except Exception as e:
        logger.error("analyze_session_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/upload", response_model=MedicalDocumentSchema, tags=["Pipeline"])
async def upload_document(file: UploadFile = File(...)) -> MedicalDocumentSchema:
    """
    Secure medical document upload endpoint.

    Accepted formats
    ----------------
    - **PDF**  — text-based medical PDFs (.pdf)
    - **DICOM** — MRI / CT / X-ray studies (.dcm, .dicom)
    - **Image** — scanned lab reports, handwritten notes OCR'd via Tesseract (.jpg, .jpeg, .png)
    - **Spreadsheet** — structured lab results (.xlsx)

    Limits
    ------
    - Max file size: **1 GB**

    Returns
    -------
    MedicalDocumentSchema with raw_text, metadata,
    document_type, and (for XLSX) tabular_data.
    """
    # ── MIME type guard ──────────────────────────────────────────────────────
    if file.content_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{file.content_type}'. "
                f"Accepted types: {sorted(_ALLOWED_MIME_TYPES)}"
            ),
        )

    try:
        content = await file.read()
    except Exception as exc:
        logger.error("file_read_error", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to read uploaded file.") from exc

    # ── Size guard (post-read; streaming chunked guard can be added later) ───
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the 1 GB limit ({len(content):,} bytes received).",
        )

    try:
        result = await parser_agent.run(
            file_content=content,
            filename=file.filename or "unknown",
            mime_type=file.content_type,
        )
        return result

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("upload_processing_failed", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error during document processing.")


@app.post("/intake", response_model=IntakeManifest, tags=["Pipeline"])
async def intake_batch(
    files: List[UploadFile] = File(...),
    session_id: str = Query(..., description="Session UUID obtained from POST /session/create"),
) -> IntakeManifest:
    """
    Unified batch intake endpoint.

    Accepts a list of mixed files (standalone PDFs, DCMs, images, XLSXs, 
    and/or ZIP archives) and organises them all into a unified `data/User/<session_id>` 
    directory tree before AI processing starts.

    What happens
    ------------
    1. ZIP files in the batch are extracted in memory.
    2. All supported standalone files AND extracted ZIP contents are written to:
       - `data/User/<session_id>/dicom/`
       - `data/User/<session_id>/pdf/`
       - `data/User/<session_id>/images/`
       - `data/User/<session_id>/spreadsheets/`
    3. An IntakeManifest is returned describing the entire batch.
    """
    uploaded_items: List[UploadedItem] = []
    total_size = 0

    for file in files:
        if not file.filename:
            continue
            
        try:
            content = await file.read()
            total_size += len(content)
            
            if total_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Batch exceeds the 1 GB total limit."
                )
                
            uploaded_items.append(UploadedItem(filename=file.filename, content=content))
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("batch_read_error", filename=file.filename, error=str(exc))
            raise HTTPException(status_code=500, detail=f"Failed to read '{file.filename}'.") from exc

    if not uploaded_items:
        raise HTTPException(status_code=400, detail="No valid files provided.")

    try:
        manifest = batch_intake_service.process_batch(items=uploaded_items, session_id=session_id)
        
        # 1. Initialize atomic session lock (Metadata Drift Safety)
        async with atomic_session_lock(manifest.session_id, caller_name="intake_batch") as session:
            # At this early stage, we set the initial lock if possible, 
            # though patient IDs are mostly parsed during phase 1.
            pass
            
        return manifest
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        logger.error("batch_intake_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error during batch intake.")


@app.post("/privacy-scan", response_model=List[MedicalDocumentSchema], tags=["Pipeline"])
async def privacy_scan(manifest: IntakeManifest) -> List[MedicalDocumentSchema]:
    """
    Phase 2 Privacy Scanner Endpoint.
    
    Takes an `IntakeManifest` JSON object (typically returned by `/intake`),
    reads the staged files off the disk, extracts their text/metadata using 
    the parser agent, and safely scrubs all PII/PHI using Microsoft Presidio.
    
    Returns a list of fully anonymised `MedicalDocumentSchema` objects.
    """
    if not manifest or not manifest.files:
        raise HTTPException(status_code=400, detail="Manifest contains no files to process.")
        
    try:
        # Run the agent over the entire batch
        clean_docs = await privacy_agent.run(manifest=manifest)
        
        if not clean_docs:
            raise HTTPException(
                status_code=422, 
                detail="All files failed privacy extraction. Check server logs."
            )
            
        return clean_docs
        
    except Exception as exc:
        logger.error("privacy_scan_failed", session_id=manifest.session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error during privacy scanning.")


@app.post("/prepare-data", response_model=List[MedicalDocumentSchema], tags=["Pipeline"])
async def prepare_data(documents: List[MedicalDocumentSchema]) -> List[MedicalDocumentSchema]:
    """
    Phase 3 Data Preparation Endpoint.
    
    Accepts PII-scrubbed `MedicalDocumentSchema` objects (typically from Phase 2),
    expands medical terminology, standardizes units, and cleanly chunks the text 
    by clinical headers or overlap constraints for vector ingestion.
    
    Returns the enriched documents with `normalized_text` and `chunks`.
    """
    if not documents:
        raise HTTPException(status_code=400, detail="No documents provided for data preparation.")
        
    try:
        enriched_docs = await data_prep_agent.run(documents=documents)
        return enriched_docs
    except Exception as exc:
        logger.error("data_prep_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error during data preparation.")


@app.post("/rag/index-patient", response_model=RAGIndexPatientResponse, tags=["RAG"])
async def rag_index_patient(payload: RAGIndexPatientRequest) -> RAGIndexPatientResponse:
    """
    Build a session-scoped patient FAISS store from already prepared documents.
    """
    if not payload.documents:
        raise HTTPException(status_code=400, detail="No documents provided for patient indexing.")

    try:
        rag_agent = _require_rag_agent()
        stats = await rag_agent.ingest_patient_documents(
            session_id=payload.session_id,
            documents=payload.documents,
        )
        return RAGIndexPatientResponse(session_id=payload.session_id, **stats)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("rag_index_patient_failed", session_id=payload.session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error during patient indexing.")


@app.post("/rag/retrieve", response_model=RAGRetrieveResponse, tags=["RAG"])
async def rag_retrieve(payload: RAGRetrieveRequest) -> RAGRetrieveResponse:
    """
    Retrieve evidence from the patient session store and global medical store.
    """
    try:
        rag_agent = _require_rag_agent()
        results = await rag_agent.retrieve(
            query=payload.query,
            session_id=payload.session_id,
            top_k_patient=payload.top_k_patient,
            top_k_global=payload.top_k_global,
            top_k_total=payload.top_k_total,
        )
        return RAGRetrieveResponse(query=payload.query, results=results)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("rag_retrieve_failed", session_id=payload.session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error during retrieval.")


@app.delete("/session/{session_id}", response_model=RAGCleanupResponse, tags=["Session"])
async def delete_session(session_id: str) -> RAGCleanupResponse:
    """
    Full session teardown: deletes pgvector embeddings AND all files from
    data/User/<session_id>/ on disk.  Called automatically by the frontend
    on page unload via navigator.sendBeacon.
    """
    try:
        # 1. Remove pgvector rows
        rag_agent = _require_rag_agent()
        deleted = rag_agent.cleanup_session(session_id=session_id)

        # 2. Remove staged files from disk
        session_dir = Path(settings.rag_patient_data_root) / session_id
        if session_dir.exists() and session_dir.is_dir():
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info("session_disk_cleanup", session_id=session_id, path=str(session_dir))

        return RAGCleanupResponse(session_id=session_id, deleted=deleted)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("session_cleanup_failed", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error during session cleanup.")


# Keep the old RAG-scoped route as an alias so existing integrations don't break
@app.delete("/rag/session/{session_id}", response_model=RAGCleanupResponse, tags=["RAG"], include_in_schema=False)
async def rag_cleanup_session(session_id: str) -> RAGCleanupResponse:
    return await delete_session(session_id)


@app.post("/session/{session_id}/cleanup", tags=["Session"])
async def session_cleanup_beacon(session_id: str) -> dict:
    """
    Beacon-compatible cleanup endpoint (POST) used by navigator.sendBeacon on
    page unload.  Performs the same full teardown as DELETE /session/{session_id}.
    """
    await delete_session(session_id)
    return {"ok": True}
