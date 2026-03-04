"""
src/services/extractors.py
---------------------------
Stateless extraction utilities, one per supported file format.
Each extractor receives raw bytes + filename and returns a
(raw_text, metadata, tabular_data) tuple so the caller can
build a MedicalDocumentSchema.

Supported formats:
  - PDF       → PyMuPDF (fitz)
  - DICOM     → pydicom  (extracts pixel data description + embedded tags)
  - JPG / PNG → Pillow + pytesseract OCR
  - XLSX      → openpyxl / pandas  (sheets → text + tabular records)
  - ZIP       → stdlib zipfile  (discovers + merges all .dcm files inside)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pydicom
from PIL import Image
import pytesseract
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Type alias ───────────────────────────────────────────────────────────────
ExtractionResult = Tuple[
    str,                          # raw_text
    Dict[str, Any],               # metadata
    Optional[List[Dict[str, Any]]]  # tabular_data  (only for XLSX)
]


# ── PDF ───────────────────────────────────────────────────────────────────────

def extract_pdf(content: bytes, filename: str) -> ExtractionResult:
    """
    Extract full text from a PDF using PyMuPDF.
    Works for text-based PDFs; falls back to empty string for scanned PDFs
    (OCR on scanned PDFs should be a separate phase).
    """
    raw_text = ""
    page_count = 0
    logger.info("extract_pdf_started", filename=filename, size_bytes=len(content))

    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            page_count = doc.page_count
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                raw_text += page.get_text() + "\n"
    except Exception as exc:
        logger.error("extract_pdf_failed", filename=filename, error=str(exc))
        raise ValueError(f"PDF extraction failed: {exc}") from exc

    cleaned_text = raw_text.strip()
    if not cleaned_text:
        logger.warning("extract_pdf_empty_text", filename=filename, page_count=page_count)
        raise ValueError("Scanned PDF with no text detected. OCR fallback required.")

    logger.info("extract_pdf_completed", filename=filename, page_count=page_count, text_length=len(cleaned_text))
    return (
        cleaned_text,
        {"filename": filename, "page_count": page_count},
        None,
    )


# ── DICOM ─────────────────────────────────────────────────────────────────────

def extract_dicom(content: bytes, filename: str) -> ExtractionResult:
    """
    Parse a DICOM file and extract its textual metadata tags.

    Relevant tags include patient name, modality, study/series description,
    institution name, etc. Pixel data is intentionally excluded from raw_text
    (it will be handled by the Diagnostic Agent later).
    """
    TEXT_TAGS = [
        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
        "StudyDate", "StudyDescription", "SeriesDescription",
        "Modality", "InstitutionName", "ReferringPhysicianName",
        "StudyInstanceUID", "Manufacturer",
    ]
    logger.info("extract_dicom_started", filename=filename, size_bytes=len(content))

    try:
        ds = pydicom.dcmread(io.BytesIO(content))
    except Exception as exc:
        logger.error("extract_dicom_failed", filename=filename, error=str(exc))
        raise ValueError(f"DICOM extraction failed: {exc}") from exc

    lines: List[str] = []
    meta: Dict[str, Any] = {"filename": filename}

    for tag_name in TEXT_TAGS:
        value = getattr(ds, tag_name, None)
        if value is not None:
            str_val = str(value)
            lines.append(f"{tag_name}: {str_val}")
            meta[tag_name] = str_val

    # Capture modality at the top level for downstream agents
    meta["modality"] = getattr(ds, "Modality", "UNKNOWN")
    meta["has_pixel_data"] = hasattr(ds, "PixelData")

    raw_text = "\n".join(lines)
    logger.info(
        "extract_dicom_completed",
        filename=filename,
        modality=meta.get("modality"),
        tags_extracted=len(lines),
        has_pixel_data=meta["has_pixel_data"],
    )
    return raw_text, meta, None


# ── IMAGE (JPG / PNG) ─────────────────────────────────────────────────────────

def extract_image(content: bytes, filename: str) -> ExtractionResult:
    """
    Run Tesseract OCR on a JPEG or PNG to extract embedded text.
    Resizes extremely large images to prevent OOM/slow OCR.
    """
    MAX_IMAGE_PIXELS = 16_000_000  # e.g., 4000x4000
    logger.info("extract_image_started", filename=filename, size_bytes=len(content))
    
    try:
        image = Image.open(io.BytesIO(content))
        # Ensure we don't blow up memory on crazy resolutions
        if image.width * image.height > MAX_IMAGE_PIXELS:
            image.thumbnail((4000, 4000), Image.Resampling.LANCZOS)
    except Exception as exc:
        logger.error("extract_image_open_failed", filename=filename, error=str(exc))
        raise ValueError(f"Image open failed: {exc}") from exc

    try:
        raw_text = pytesseract.image_to_string(image)
    except Exception as exc:
        logger.error("extract_image_ocr_failed", filename=filename, error=str(exc))
        raise ValueError(
            f"OCR failed for '{filename}'. "
            f"Ensure Tesseract is installed and on PATH. Detail: {exc}"
        ) from exc

    cleaned_text = raw_text.strip()
    if not cleaned_text:
        logger.warning("extract_image_empty_text", filename=filename)
        raise ValueError("OCR generated no text. Image may be empty or blurry.")

    meta: Dict[str, Any] = {
        "filename": filename,
        "image_format": image.format or Path(filename).suffix.lstrip(".").upper(),
        "image_size": image.size,   # (width, height) in pixels
        "image_mode": image.mode,
    }
    logger.info(
        "extract_image_completed",
        filename=filename,
        image_format=meta["image_format"],
        image_size=meta["image_size"],
        text_length=len(cleaned_text),
    )
    return cleaned_text, meta, None


# ── XLSX ──────────────────────────────────────────────────────────────────────

def extract_xlsx(content: bytes, filename: str) -> ExtractionResult:
    """
    Extract tabular lab-result data from an Excel spreadsheet.
    Each sheet is converted to:
      - tabular_data (list of row dicts) for structured downstream processing
      - raw_text (CSV-style dump) so text-based agents can still work on it
    """
    logger.info("extract_xlsx_started", filename=filename, size_bytes=len(content))
    try:
        xl = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
    except Exception as exc:
        logger.error("extract_xlsx_failed", filename=filename, error=str(exc))
        raise ValueError(f"XLSX extraction failed: {exc}") from exc

    all_rows: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name).fillna("")
        if df.empty:
            logger.debug("extract_xlsx_empty_sheet_skipped", filename=filename, sheet=sheet_name)
            continue
            
        # Prefix rows with their sheet name so multiple sheets stay distinct
        df["__sheet__"] = sheet_name
        rows = df.to_dict(orient="records")
        all_rows.extend(rows)
        text_parts.append(f"=== Sheet: {sheet_name} ===")
        text_parts.append(df.to_csv(index=False))

    if not all_rows:
        logger.warning("extract_xlsx_empty_workbook", filename=filename, sheet_count=len(xl.sheet_names))
        raise ValueError("Empty spreadsheet: No data found in any sheet.")

    meta: Dict[str, Any] = {
        "filename": filename,
        "sheets": xl.sheet_names,
        "total_rows": len(all_rows),
    }
    logger.info(
        "extract_xlsx_completed",
        filename=filename,
        sheet_count=len(xl.sheet_names),
        total_rows=len(all_rows),
    )
    return "\n".join(text_parts).strip(), meta, all_rows


# ── ZIP (containing DICOM series) ─────────────────────────────────────────────

_DICOM_EXTENSIONS: frozenset[str] = frozenset({".dcm", ".dicom"})


def extract_zip(content: bytes, filename: str) -> ExtractionResult:
    """
    Unpack a ZIP archive and extract all DICOM files found inside.

    Behaviour
    ---------
    - Recursively discovers every entry whose extension is .dcm or .dicom.
    - Runs extract_dicom on each file and merges text + metadata.
    - Files with other extensions are logged in metadata but skipped.
    - Raises ValueError if the archive contains no recognisable DICOM files.

    Returns
    -------
    ExtractionResult where:
      - raw_text  : per-file DICOM tag dumps separated by a header banner
      - metadata  : archive-level info + list of per-file metadata dicts
      - tabular_data : None  (DICOM series carry no tabular payload here)
    """
    logger.info("extract_zip_started", filename=filename, size_bytes=len(content))
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        logger.error("extract_zip_invalid_archive", filename=filename, error=str(exc))
        raise ValueError(f"Invalid ZIP archive '{filename}': {exc}") from exc

    all_names: List[str] = zf.namelist()
    dicom_entries: List[str] = [
        name for name in all_names
        if Path(name).suffix.lower() in _DICOM_EXTENSIONS
    ]
    skipped: List[str] = [n for n in all_names if n not in dicom_entries]

    if not dicom_entries:
        logger.warning("extract_zip_no_dicom_files", filename=filename, total_entries=len(all_names))
        raise ValueError(
            f"ZIP archive '{filename}' contains no DICOM files (.dcm / .dicom). "
            f"Found: {all_names[:10]}"
        )

    text_parts: List[str] = []
    per_file_meta: List[Dict[str, Any]] = []

    for entry_name in dicom_entries:
        try:
            dcm_bytes = zf.read(entry_name)
            entry_text, entry_meta, _ = extract_dicom(dcm_bytes, entry_name)
        except ValueError as exc:
            # Skip corrupt individual files but keep going
            logger.warning("extract_zip_entry_failed", filename=filename, entry=entry_name, error=str(exc))
            text_parts.append(f"--- {entry_name} [FAILED: {exc}] ---")
            per_file_meta.append({"entry": entry_name, "error": str(exc)})
            continue

        text_parts.append(f"=== DICOM: {entry_name} ===")
        text_parts.append(entry_text)
        per_file_meta.append({"entry": entry_name, **entry_meta})

    zf.close()

    meta: Dict[str, Any] = {
        "filename": filename,
        "total_entries": len(all_names),
        "dicom_files_found": len(dicom_entries),
        "skipped_files": skipped,
        "dicom_files": per_file_meta,
    }
    logger.info(
        "extract_zip_completed",
        filename=filename,
        total_entries=len(all_names),
        dicom_files_found=len(dicom_entries),
        skipped_files=len(skipped),
    )
    return "\n".join(text_parts).strip(), meta, None
