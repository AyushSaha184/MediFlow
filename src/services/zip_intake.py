"""
src/services/zip_intake.py
---------------------------
ZipIntakeService — extracts a mixed ZIP archive and organises its
contents into type-specific subdirectories under data/User/<session_id>/.

Folder structure created on disk
---------------------------------
data/
└── <session_id>/
    ├── dicom/          ← .dcm, .dicom
    ├── pdf/            ← .pdf
    ├── images/         ← .jpg, .jpeg, .png
    ├── spreadsheets/   ← .xlsx
    └── unknown/        ← anything else (kept for audit, not processed)

The service is purposely stateless — it takes bytes in, writes files to
disk, and returns an IntakeManifest.  Cleanup of the staged directory is
the responsibility of the caller (or a background task in production).
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
import hashlib
import httpx
import filetype
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.models.document_type_map import EXTENSION_TO_DOC_TYPE
from src.models.intake_manifest import IntakeManifest, StagedFile
from src.models.medical_document import DocumentType
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Validation Constants ─────────────────────────────────────────────────────
MAX_FILES_PER_BATCH = 500
MAX_UNCOMPRESSED_TOTAL_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB

# ── Folder name for each DocumentType ────────────────────────────────────────
_TYPE_FOLDER: Dict[DocumentType, str] = {
    DocumentType.DICOM:        "dicom",
    DocumentType.PDF:          "pdf",
    DocumentType.IMAGE:        "images",
    DocumentType.SPREADSHEET:  "spreadsheets",
}

# Root data directory (project root/data/)
_DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "User"

from dataclasses import dataclass

@dataclass
class UploadedItem:
    filename: str
    content: bytes


class UnifiedBatchIntakeService:
    """
    Accepts a batch of uploaded files. 
    - Standalone supported files are staged directly to disk by document type.
    - ZIP archives are extracted, and their supported contents staged similarly.
    
    All files in the batch share the same session_id and output directory tree.

    Usage
    -----
    service = UnifiedBatchIntakeService()
    manifest = service.process_batch(items=[UploadedItem("scan.jpg", b"..."), UploadedItem("study.zip", b"...")])
    """

    def __init__(self, data_root: Path = _DATA_ROOT) -> None:
        self.data_root = data_root

    def _stage_file(
        self, 
        session_root: Path, 
        filename: str, 
        file_bytes: bytes, 
        staged_files: List[StagedFile], 
        skipped: List[str], 
        by_type: Dict[str, List[str]],
        seen_hashes: Set[str]
    ) -> None:
        """Helper to classify, validate, and write a single file to disk."""
        suffix = Path(filename).suffix.lower()
        doc_type = EXTENSION_TO_DOC_TYPE.get(suffix)

        if doc_type is None or doc_type == DocumentType.ZIP_DICOM:
            # Unsupported or nested ZIP
            skipped.append(filename)
            logger.warning(
                "batch_intake_skipped",
                entry=filename,
                reason="unsupported extension or nested ZIP",
            )
            return

        # ── Duplicate Detection ──
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if file_hash in seen_hashes:
            skipped.append(filename)
            logger.warning("batch_intake_skipped_duplicate", entry=filename, sha256=file_hash)
            return
        
        # ── Magic MIME Validation ──
        # Certain extensions like .dcm, .dicom aren't universally supported by 'filetype',
        # but standard files like PDF, JPG, PNG, XLSX can be easily verified.
        kind = filetype.guess(file_bytes)
        if doc_type in (DocumentType.PDF, DocumentType.IMAGE, DocumentType.SPREADSHEET):
            if kind is None:
                skipped.append(filename)
                logger.warning("batch_intake_skipped_magic", entry=filename, reason="could not detect valid file signature")
                return

        folder_name = _TYPE_FOLDER.get(doc_type, "unknown")
        dest_dir = session_root / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / Path(filename).name

        if dest_path.exists():
            stem = dest_path.stem
            ext = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{ext}"
                counter += 1

        dest_path.write_bytes(file_bytes)
        seen_hashes.add(file_hash)

        staged = StagedFile(
            original_name=filename,
            staged_path=str(dest_path),
            document_type=doc_type,
            size_bytes=len(file_bytes),
            file_hash=file_hash,
        )
        staged_files.append(staged)

        type_key = doc_type.value
        by_type.setdefault(type_key, []).append(str(dest_path))

        logger.debug(
            "batch_intake_staged",
            original=filename,
            dest=str(dest_path),
            type=doc_type.value,
        )

    def _upload_to_supabase(self, session_id: str, staged_files: List[StagedFile]) -> None:
        """Upload staged files to the Supabase med-docs storage bucket."""
        from src.core.config import settings
        if not settings.supabase_url or not settings.supabase_service_key:
            logger.warning("supabase_storage_skipped", reason="SUPABASE_URL or SUPABASE_SERVICE_KEY not configured")
            return
        base_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/med-docs"
        headers = {
            "Authorization": f"Bearer {settings.supabase_service_key}",
            "apikey": settings.supabase_service_key,
        }
        with httpx.Client(timeout=120.0) as client:
            for sf in staged_files:
                bucket_path = f"{session_id}/{Path(sf.staged_path).name}"
                try:
                    file_bytes = Path(sf.staged_path).read_bytes()
                    resp = client.post(f"{base_url}/{bucket_path}", headers=headers, content=file_bytes)
                    resp.raise_for_status()
                    logger.info("supabase_upload_ok", file=sf.original_name, path=bucket_path)
                except Exception as exc:
                    logger.warning("supabase_upload_failed", file=sf.original_name, error=str(exc))

    def process_batch(self, items: List[UploadedItem], session_id: Optional[str] = None) -> IntakeManifest:
        """
        Process a mixed batch of ZIPs and standalone files.
        """
        total_size = sum(len(item.content) for item in items)
        source_names = [item.filename for item in items]
        logger.info("batch_intake_started", sources=source_names, total_size=total_size)

        session_id = session_id or uuid.uuid4().hex[:12]
        session_root = self.data_root / session_id
        session_root.mkdir(parents=True, exist_ok=True)

        staged_files: List[StagedFile] = []
        skipped: List[str] = []
        by_type: Dict[str, List[str]] = {}
        seen_hashes: Set[str] = set()

        total_uncompressed = 0
        total_files_count = 0

        for item in items:
            suffix = Path(item.filename).suffix.lower()
            doc_type = EXTENSION_TO_DOC_TYPE.get(suffix)

            if doc_type == DocumentType.ZIP_DICOM:
                # Unpack ZIP inline
                try:
                    zf = zipfile.ZipFile(io.BytesIO(item.content))
                    for entry in zf.infolist():
                        if entry.filename.endswith("/"):
                            continue
                        
                        total_files_count += 1
                        if total_files_count > MAX_FILES_PER_BATCH:
                            raise ValueError(f"Batch exceeds maximum file count ({MAX_FILES_PER_BATCH}).")
                            
                        total_uncompressed += entry.file_size
                        if total_uncompressed > MAX_UNCOMPRESSED_TOTAL_SIZE:
                            raise ValueError("ZIP bomb detected: Extracted files exceed size limit.")

                        entry_bytes = zf.read(entry.filename)
                        
                        # Strip directory parts to flatten depth, neutralizing Zip-Slip and deep nest structures.
                        safe_entry_name = Path(entry.filename).name
                        virtual_name = f"{Path(item.filename).stem}_{safe_entry_name}"
                        
                        self._stage_file(session_root, virtual_name, entry_bytes, staged_files, skipped, by_type, seen_hashes)
                    zf.close()
                except zipfile.BadZipFile as exc:
                    logger.error("batch_intake_bad_zip", filename=item.filename, error=str(exc))
                    skipped.append(item.filename)
            else:
                # Standalone file
                total_files_count += 1
                if total_files_count > MAX_FILES_PER_BATCH:
                    raise ValueError(f"Batch exceeds maximum file count ({MAX_FILES_PER_BATCH}).")
                
                total_uncompressed += len(item.content)
                if total_uncompressed > MAX_UNCOMPRESSED_TOTAL_SIZE:
                    raise ValueError("Batch exceeds uncompressed size limit.")
                    
                self._stage_file(session_root, item.filename, item.content, staged_files, skipped, by_type, seen_hashes)

        manifest = IntakeManifest(
            session_id=session_id,
            source_files=source_names,
            staged_root=str(session_root),
            total_files=len(staged_files),
            skipped_files=skipped,
            files=staged_files,
            by_type=by_type,
        )

        logger.info(
            "batch_intake_complete",
            session_id=session_id,
            staged=len(staged_files),
            skipped=len(skipped),
            types=list(by_type.keys()),
        )

        # Persist manifest to disk so analyze-medical-session can reload it
        manifest_path = session_root / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

        # Upload staged files to Supabase med-docs bucket
        self._upload_to_supabase(session_id, staged_files)

        return manifest
