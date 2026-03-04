"""
tests/test_privacy_agent.py
---------------------------
Tests to ensure the PrivacyAgent and PrivacyService correctly 
scrub PII/PHI (Names, Phones, SSNs, Emails, etc.) from raw text, 
metadata, and tabular data using Microsoft Presidio.
"""

from __future__ import annotations

import io
import fitz
import zipfile
from pathlib import Path

import pytest

from src.models.intake_manifest import IntakeManifest, StagedFile
from src.models.medical_document import DocumentType
from src.services.privacy_service import PrivacyService
from src.agents.parser_agent import MedicalParserAgent
from src.agents.privacy_agent import PrivacyProtectionAgent


def _make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    data = doc.write()
    doc.close()
    return data


@pytest.fixture
def privacy_service() -> PrivacyService:
    # Initialize once per test session (Presidio models take a few ms)
    return PrivacyService()


@pytest.fixture
def privacy_agent(privacy_service: PrivacyService) -> PrivacyProtectionAgent:
    parser = MedicalParserAgent()
    return PrivacyProtectionAgent(parser_agent=parser, privacy_service=privacy_service)


# ─-----------------------------------------------------------------------------
# 1. PrivacyService Tests (Unit)
# ------------------------------------------------------------------------------

def test_privacy_service_scrubs_text(privacy_service: PrivacyService):
    """Test the raw text anonymiser against common PII targets."""
    dirty_text = (
        "Patient Jane Smith (DOB 1985) is seen today. "
        "Her phone number is 555-0199 and her email is jane.smith@example.com. "
        "SSN: ***-**-****."
    )
    # The SSN won't trigger if masked, let's use a full fake one to test SSN logic
    # Presidio US_SSN recogniser prefers standard 9 digit with dashes context
    dirty_text_full = (
        "Patient John Doe. Contact: 123-456-7890. Email: john@doe.com. My Social is 234-56-7890."
    )
    
    clean_text = privacy_service.anonymize_text(dirty_text_full)
    
    assert "John Doe" not in clean_text
    assert "<PERSON>" in clean_text
    
    assert "123-456-7890" not in clean_text
    assert "<PHONE_NUMBER>" in clean_text
    
    assert "john@doe.com" not in clean_text
    assert "<EMAIL_ADDRESS>" in clean_text
    
    assert "234-56-7890" not in clean_text
    assert "<US_SSN>" in clean_text


def test_privacy_service_scrubs_metadata(privacy_service: PrivacyService):
    """Deeply nested metadata dictionaries must also be scrubbed."""
    dirty_meta = {
        # 'john_doe_report.pdf' is hard for off-the-shelf NER to catch as a PERSON usually.
        # So we test a more realistic metadata field like author or explicit patient name.
        "document_author": "Gregory House",
        "nested": {
            "referring_physician": "Dr. Lisa Cuddy",
            "contact": "Call 987-654-3210 for details",
        },
        "tags": ["urgent", "patient: James Wilson"]
    }
    
    clean_meta = privacy_service.anonymize_metadata(dirty_meta)
    
    assert "Gregory House" not in clean_meta["document_author"]
    assert "Lisa Cuddy" not in clean_meta["nested"]["referring_physician"]
    assert "987-654-3210" not in clean_meta["nested"]["contact"]
    assert "James Wilson" not in clean_meta["tags"][1]
    
    # Structure remains identical
    assert isinstance(clean_meta["nested"], dict)
    assert isinstance(clean_meta["tags"], list)


def test_privacy_service_safely_ignores_empty(privacy_service: PrivacyService):
    assert privacy_service.anonymize_text("") == ""
    assert privacy_service.anonymize_text("   ") == "   "
    assert privacy_service.anonymize_text(None) is None


# ─-----------------------------------------------------------------------------
# 2. PrivacyAgent Tests (Integration with Parser)
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_privacy_agent_cleans_staged_pdf(
    tmp_path: Path, 
    privacy_agent: PrivacyProtectionAgent
):
    """
    Simulates writing a PII-heavy PDF to disk (as if from Phase 1),
    building a fake IntakeManifest, and running the PrivacyAgent over it.
    """
    pdf_bytes = _make_pdf(
        "CLINICAL NOTE\n"
        "Patient Alice Wonderland walked in complaining of headache.\n"
        "Callback num: 555-867-5309"
    )
    
    staged_file_path = tmp_path / "alice_note.pdf"
    staged_file_path.write_bytes(pdf_bytes)
    
    fake_staged = StagedFile(
        original_name="alice_note.pdf",
        staged_path=str(staged_file_path),
        document_type=DocumentType.PDF,
        size_bytes=len(pdf_bytes),
        file_hash="fakehash123"
    )
    
    manifest = IntakeManifest(
        session_id="test-privacy-session",
        source_files=["upload.zip"],
        staged_root=str(tmp_path),
        total_files=1,
        skipped_files=[],
        files=[fake_staged],
        by_type={"pdf": [str(staged_file_path)]}
    )
    
    # Act
    clean_docs = await privacy_agent.run(manifest)
    
    # Assert
    assert len(clean_docs) == 1
    doc = clean_docs[0]
    
    assert doc.document_type == DocumentType.PDF
    assert "Alice Wonderland" not in doc.raw_text
    assert "<PERSON>" in doc.raw_text
    assert "555-867-5309" not in doc.raw_text
    assert "<PHONE_NUMBER>" in doc.raw_text
    
    # Verify the agent name was appended to the lineage
    assert "MedicalParserAgent" in doc.processed_by
    assert "PrivacyAgent" in doc.processed_by
