"""
tests/test_zip_intake.py
------------------------
Tests for ZipIntakeService — ZIP extraction and file staging.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import fitz
import openpyxl
import pytest

from src.services.zip_intake import UnifiedBatchIntakeService, UploadedItem
from src.models.medical_document import DocumentType


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_zip(files: dict[str, bytes]) -> bytes:
    """Build an in-memory ZIP from a dict of {filename: content}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Patient: Jane Smith\nDiagnosis: Diabetes Type 2")
    data = doc.write()
    doc.close()
    return data


def _make_xlsx() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Test", "Value"])
    ws.append(["HbA1c", "7.2"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_jpg() -> bytes:
    from PIL import Image
    img = Image.new('RGB', (10, 10), color='white')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_service(tmp_path: Path) -> UnifiedBatchIntakeService:
    """Service that stages to a pytest-managed temp directory."""
    return UnifiedBatchIntakeService(data_root=tmp_path)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_intake_mixed_batch(tmp_service: UnifiedBatchIntakeService):
    """A batch with a standalone PDF, standalone XLSX, and a ZIP is staged into type folders."""
    zip_bytes = _make_zip({
        "notes/readme.txt": b"ignore me",   # unsupported inside zip
        "scan.jpg": _make_jpg(),            # valid fake image
    })
    
    items = [
        UploadedItem("history.pdf", _make_pdf()),
        UploadedItem("results.xlsx", _make_xlsx()),
        UploadedItem("archive.zip", zip_bytes),
        UploadedItem("bad_file.doc", b"unsupported standalone"),
    ]

    manifest = tmp_service.process_batch(items)

    assert len(manifest.source_files) == 4
    assert manifest.total_files == 3  # history.pdf, results.xlsx, archive_scan.jpg
    
    assert "bad_file.doc" in manifest.skipped_files
    assert "archive_readme.txt" in manifest.skipped_files

    # Check by_type keys
    assert DocumentType.PDF.value in manifest.by_type
    assert DocumentType.SPREADSHEET.value in manifest.by_type
    assert DocumentType.IMAGE.value in manifest.by_type

    # Verify files land in the right folders
    pdf_path = Path(manifest.by_type[DocumentType.PDF.value][0])
    xlsx_path = Path(manifest.by_type[DocumentType.SPREADSHEET.value][0])
    img_path = Path(manifest.by_type[DocumentType.IMAGE.value][0])
    
    assert "pdf" in pdf_path.parts
    assert "spreadsheets" in xlsx_path.parts
    assert "images" in img_path.parts
    
    assert pdf_path.exists()
    assert xlsx_path.exists()
    assert img_path.exists()


def test_intake_collision_handling(tmp_service: UnifiedBatchIntakeService):
    """Two PDFs with the same base filename but different contents don't overwrite each other."""
    pdf1 = _make_pdf()
    
    # Make a slightly different PDF to avoid SHA256 deduplication
    doc = fitz.open()
    doc.new_page().insert_text((0, 0), "Different text")
    pdf2 = doc.write()
    doc.close()
    
    items = [
        UploadedItem("report.pdf", pdf1),
        UploadedItem("report.pdf", pdf2),
    ]
    manifest = tmp_service.process_batch(items)
    assert manifest.total_files == 2
    paths = manifest.by_type[DocumentType.PDF.value]
    # Paths should be different on disk
    assert len(set(paths)) == 2


def test_intake_invalid_zip(tmp_service: UnifiedBatchIntakeService):
    """An invalid zip in a batch is logged and skipped, but doesn't crash the whole batch."""
    items = [
        UploadedItem("report.pdf", _make_pdf()),
        UploadedItem("bad.zip", b"not really a zip"),
    ]
    manifest = tmp_service.process_batch(items)
    
    assert manifest.total_files == 1  # the pdf still passes
    assert "bad.zip" in manifest.skipped_files


def test_intake_all_unsupported(tmp_service: UnifiedBatchIntakeService):
    """A batch with only unsupported files stages zero files."""
    zip_bytes = _make_zip({"notes.doc": b"word doc", "data.csv": b"a,b,c"})
    items = [
        UploadedItem("unsupported.zip", zip_bytes),
        UploadedItem("random.exe", b"bad"),
    ]
    manifest = tmp_service.process_batch(items)
    
    assert manifest.total_files == 0
    assert len(manifest.skipped_files) == 3  # random.exe + notes.doc + data.csv
