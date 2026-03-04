"""
tests/test_parser_agent.py
---------------------------
Unit tests for MedicalParserAgent covering all supported file formats.
"""

from __future__ import annotations

import io
import zipfile
import openpyxl
import pytest
import fitz  # PyMuPDF — used to create in-memory test PDFs
import pydicom
import pydicom.data

from src.agents.parser_agent import MedicalParserAgent, detect_document_type
from src.models.medical_document import DocumentType, MedicalDocumentSchema


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a minimal text-based PDF in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Patient Name: John Doe\nDOB: 01/01/1980\nDiagnosis: Hypertension")
    data = doc.write()
    doc.close()
    return data


@pytest.fixture
def sample_xlsx_bytes() -> bytes:
    """Generate a minimal XLSX file in memory."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Lab Results"
    ws.append(["Test", "Value", "Unit", "Reference Range"])
    ws.append(["Hemoglobin", "14.2", "g/dL", "13.5–17.5"])
    ws.append(["WBC", "7.5", "x10³/µL", "4.5–11.0"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture
def sample_dicom_bytes() -> bytes:
    """Return bytes of a known-good DICOM test file shipped with pydicom."""
    path = pydicom.data.get_testdata_files("CT_small.dcm")[0]
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def sample_zip_with_dicom(sample_dicom_bytes: bytes) -> bytes:
    """Build an in-memory ZIP containing one .dcm file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("study/slice_001.dcm", sample_dicom_bytes)
    return buf.getvalue()


@pytest.fixture
def sample_zip_no_dicom() -> bytes:
    """Build a ZIP that has no .dcm files (should trigger ValueError)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("readme.txt", "no DICOM here")
    return buf.getvalue()


# ── Extension detection ────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename,expected", [
    ("report.pdf",    DocumentType.PDF),
    ("scan.dcm",      DocumentType.DICOM),
    ("scan.dicom",    DocumentType.DICOM),
    ("labs.jpg",      DocumentType.IMAGE),
    ("labs.jpeg",     DocumentType.IMAGE),
    ("labs.png",      DocumentType.IMAGE),
    ("results.xlsx",  DocumentType.SPREADSHEET),
    ("study.zip",     DocumentType.ZIP_DICOM),
])
def test_detect_document_type(filename: str, expected: DocumentType):
    assert detect_document_type(filename) == expected


def test_detect_document_type_unsupported():
    with pytest.raises(ValueError, match="Unsupported file type"):
        detect_document_type("notes.doc")


# ── PDF parsing ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parser_pdf_success(sample_pdf_bytes: bytes):
    agent = MedicalParserAgent()
    result: MedicalDocumentSchema = await agent.run(
        file_content=sample_pdf_bytes, filename="test.pdf"
    )
    assert result.document_type == DocumentType.PDF
    assert "John Doe" in result.raw_text
    assert "Hypertension" in result.raw_text
    assert result.metadata["page_count"] == 1


@pytest.mark.asyncio
async def test_parser_pdf_invalid():
    agent = MedicalParserAgent()
    with pytest.raises(ValueError, match="PDF extraction failed"):
        await agent.run(file_content=b"not a real pdf", filename="bad.pdf")


# ── XLSX parsing ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parser_xlsx_success(sample_xlsx_bytes: bytes):
    agent = MedicalParserAgent()
    result: MedicalDocumentSchema = await agent.run(
        file_content=sample_xlsx_bytes, filename="labs.xlsx"
    )
    assert result.document_type == DocumentType.SPREADSHEET
    assert "Hemoglobin" in result.raw_text
    assert result.tabular_data is not None
    assert len(result.tabular_data) == 2         # 2 data rows (header excluded by pandas)
    assert result.metadata["sheets"] == ["Lab Results"]


# ── Extension guard ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parser_unsupported_extension():
    agent = MedicalParserAgent()
    with pytest.raises(ValueError, match="Unsupported file type"):
        await agent.run(file_content=b"data", filename="report.docx")


# ── ZIP parsing ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parser_zip_with_dicom(sample_zip_with_dicom: bytes):
    agent = MedicalParserAgent()
    result: MedicalDocumentSchema = await agent.run(
        file_content=sample_zip_with_dicom, filename="study.zip"
    )
    assert result.document_type == DocumentType.ZIP_DICOM
    # The metadata should list the DICOM file found inside
    assert result.metadata["dicom_files_found"] == 1
    assert result.metadata["dicom_files"][0]["entry"] == "study/slice_001.dcm"
    assert result.tabular_data is None


@pytest.mark.asyncio
async def test_parser_zip_no_dicom_files(sample_zip_no_dicom: bytes):
    agent = MedicalParserAgent()
    with pytest.raises(ValueError, match="no DICOM files"):
        await agent.run(file_content=sample_zip_no_dicom, filename="empty.zip")


@pytest.mark.asyncio
async def test_parser_zip_invalid_bytes():
    agent = MedicalParserAgent()
    with pytest.raises(ValueError, match="Invalid ZIP archive"):
        await agent.run(file_content=b"not a zip", filename="corrupt.zip")
