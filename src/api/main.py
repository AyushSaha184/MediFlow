from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.core.config import settings
from src.utils.logger import setup_logging, get_logger
from src.agents.parser_agent import MedicalParserAgent
from src.models.medical_document import MedicalDocumentSchema

setup_logging()
logger = get_logger(__name__)

# Initialize agents
parser_agent = MedicalParserAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Application starting up", project=settings.project_name, env=settings.environment)
    yield
    # Shutdown actions
    logger.info("Application shutting down")

app = FastAPI(
    title=settings.project_name,
    description="Multi-agent Medical AI system pipeline.",
    lifespan=lifespan,
)

class HealthCheckResponse(BaseModel):
    status: str
    environment: str

@app.get("/health", response_model=HealthCheckResponse, tags=["Utility"])
async def health_check():
    """
    Basic health check endpoint to verify API is running.
    """
    logger.debug("Health check requested")
    return HealthCheckResponse(status="ok", environment=settings.environment)

@app.post("/upload", response_model=MedicalDocumentSchema, tags=["Pipeline"])
async def upload_document(file: UploadFile = File(...)):
    """
    Secure file upload endpoint.
    Accepts a medical PDF document, extracts text using PyMuPDF, and returns structured data.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # 5MB limit for testing context. Reality might be larger, but this provides large-file handling context.
    MAX_FILE_SIZE = 5 * 1024 * 1024
    
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
             raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_FILE_SIZE} bytes.")
             
        # Run Parser Agent
        result = await parser_agent.run(file_content=content, filename=file.filename)
        return result
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("upload_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error during processing")

