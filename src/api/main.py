from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
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

setup_logging()
logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE: int = 1 * 1024 * 1024 * 1024  # 1 GB

# ── Service / Agent singletons ────────────────────────────────────────────────
parser_agent = MedicalParserAgent()
batch_intake_service = UnifiedBatchIntakeService()

privacy_service: PrivacyService = None
privacy_agent: PrivacyProtectionAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global privacy_service, privacy_agent
    logger.info("Application starting up", project=settings.project_name, env=settings.environment)
    # Lazy load the heavy NLP presidio models on startup
    privacy_service = PrivacyService()
    privacy_agent = PrivacyProtectionAgent(parser_agent=parser_agent, privacy_service=privacy_service)
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title=settings.project_name,
    description="Multi-agent Medical AI system pipeline.",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class HealthCheckResponse(BaseModel):
    status: str
    environment: str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheckResponse, tags=["Utility"])
async def health_check() -> HealthCheckResponse:
    """
    Basic health check. Returns 200 + status when the service is alive.
    """
    logger.debug("Health check requested")
    return HealthCheckResponse(status="ok", environment=settings.environment)


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
async def intake_batch(files: List[UploadFile] = File(...)) -> IntakeManifest:
    """
    Unified batch intake endpoint.

    Accepts a list of mixed files (standalone PDFs, DCMs, images, XLSXs, 
    and/or ZIP archives) and organises them all into a unified `data/<session_id>` 
    directory tree before AI processing starts.

    What happens
    ------------
    1. ZIP files in the batch are extracted in memory.
    2. All supported standalone files AND extracted ZIP contents are written to:
       - `data/<session_id>/dicom/`
       - `data/<session_id>/pdf/`
       - `data/<session_id>/images/`
       - `data/<session_id>/spreadsheets/`
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
        manifest = batch_intake_service.process_batch(items=uploaded_items)
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
