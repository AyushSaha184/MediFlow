import pytest
from src.agents.parser_agent import MedicalParserAgent
from src.models.medical_document import MedicalDocumentSchema

import fitz  # PyMuPDF

@pytest.fixture
def sample_pdf_bytes():
    """Generates a simple PDF in memory for testing."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Patient Name: John Doe\nDOB: 01/01/1980\nDiagnosis: Hypertension")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

@pytest.mark.asyncio
async def test_parser_agent_success(sample_pdf_bytes):
    agent = MedicalParserAgent()
    result = await agent.run(file_content=sample_pdf_bytes, filename="test.pdf")
    
    assert isinstance(result, MedicalDocumentSchema)
    assert "John Doe" in result.raw_text
    assert "Hypertension" in result.raw_text
    assert result.metadata["page_count"] == 1
    assert result.metadata["filename"] == "test.pdf"

@pytest.mark.asyncio
async def test_parser_agent_invalid_pdf():
    agent = MedicalParserAgent()
    invalid_bytes = b"Not a real PDF content"
    
    with pytest.raises(ValueError, match="Failed to parse PDF"):
        await agent.run(file_content=invalid_bytes, filename="bad.pdf")
