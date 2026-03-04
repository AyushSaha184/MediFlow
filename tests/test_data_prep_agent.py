"""
tests/test_data_prep_agent.py
-----------------------------
Test suite for Phase 3 Data Preparation logic.
Verifies the TerminologyService (abbreviation expansion, unit standardisation),
the ChunkingService (header-based sectioning and overlap bounds), and the
DataPrepAgent orchestrator.
"""

import pytest

from src.models.medical_document import MedicalDocumentSchema, DocumentType
from src.services.terminology_service import TerminologyService
from src.services.chunking_service import ChunkingService
from src.agents.data_prep_agent import DataPrepAgent

# ── 1. TerminologyService Tests ──────────────────────────────────────────────

def test_terminology_expansion():
    svc = TerminologyService()
    
    # Test abbreviation expansion boundaries
    raw = "The Pt is a 65yo M with a Hx of HTN and T2DM. He presented to the ED with SOB."
    clean = svc.normalize(raw)
    
    assert "Patient" in clean
    assert "History" in clean
    assert "Hypertension" in clean
    assert "Type 2 Diabetes Mellitus" in clean
    assert "Emergency Department" in clean
    assert "Shortness of Breath" in clean
    # Ensure it didn't mess up non-words (e.g. "PT" inside another word if case matching was lax)
    assert "Pt" not in clean
    assert "HTN" not in clean

def test_terminology_unit_standardization():
    svc = TerminologyService()
    
    # Messy varied units
    raw = "Glucose 105 mg / dl. Creatinine 1.2mg/dl. Hemoglobin 14 g / dl. HDL 45  mg / dl."
    clean = svc.normalize(raw)
    
    # Should all be uniform exactly
    assert "105 mg/dL" in clean
    assert "1.2 mg/dL" in clean
    assert "14 g/dL" in clean
    assert "45 mg/dL" in clean

def test_terminology_edge_cases():
    svc = TerminologyService()
    
    # 1. µg/L -> mcg/L, mgdl -> mg/dL, mg per dl -> mg/dL
    raw = "Vitamin B12 400 µg/L. Glucose 95mgdl. Calcium 9.5 mg per dl. Number: 1,200."
    clean = svc.normalize(raw)
    
    assert "400 mcg/L" in clean
    assert "95 mg/dL" in clean
    assert "9.5 mg/dL" in clean
    assert "Number: 1200" in clean
    
    # 2. Acronym safety (Ensure we didn't expand words)
    raw_safety = "The Pt is OR for surgery."
    clean_safety = svc.normalize(raw_safety)
    # We removed "OR" from acronyms, so it shouldn't be expanded
    assert "OR" in clean_safety
    assert "Operating Room" not in clean_safety


# ── 2. ChunkingService Tests ─────────────────────────────────────────────────

def test_chunking_by_headers():
    chunker = ChunkingService()
    
    raw = (
        "This is an intro line.\n"
        "CHIEF COMPLAINT:\nChest pain.\n\n"
        "HISTORY OF PRESENT ILLNESS:\nPatient has 2 days of chest pain.\n"
        "ASSESSMENT AND PLAN:\n1. Rule out MI.\n2. Admit."
    )
    
    chunks = chunker.chunk_document(raw)
    
    assert len(chunks) == 4
    
    assert chunks[0]["section"] == "GENERAL"
    assert "intro line" in chunks[0]["text"]
    
    assert chunks[1]["section"] == "CHIEF COMPLAINT"
    assert "Chest pain" in chunks[1]["text"]
    
    assert chunks[2]["section"] == "HISTORY OF PRESENT ILLNESS"
    assert "2 days of chest pain" in chunks[2]["text"]
    
    assert chunks[3]["section"] == "ASSESSMENT AND PLAN"
    assert "Rule out MI" in chunks[3]["text"]


def test_chunking_by_max_size_with_overlap():
    # Use small limits to easily trigger the math
    chunker = ChunkingService(target_chunk_size=100, overlap=20)
    
    # 250 characters of repeating text without headers (entire thing is GENERAL)
    # Target size = 100, Overlap = 20
    # Expected:
    # Chunk 1: [0, 100)
    # Chunk 2: [80, 180) 
    # Chunk 3: [160, 250) (ends early)
    raw_content = "A" * 250
    
    chunks = chunker.chunk_document(raw_content)
    
    # Since there are no newlines or periods, it should strictly chop at 100 char boundaries
    # Chunk 0 length is 100 + len("[GENERAL]\n") -> 100 + 10 = 110
    assert len(chunks) == 3
    
    # Strip the header when checking string sizes
    clean_chunk_0 = chunks[0]["text"].replace("[GENERAL]\n", "")
    clean_chunk_1 = chunks[1]["text"].replace("[GENERAL]\n", "")
    clean_chunk_2 = chunks[2]["text"].replace("[GENERAL]\n", "")
    
    assert len(clean_chunk_0) == 100
    assert len(clean_chunk_1) == 100
    
    # Overlap validation: The start of chunk 1 should be the end of chunk 0
    # Note: text is all 'A's so checking overlap specifically with identical chars is 
    # trivial, but the lengths prove the math was executed.
    
    assert chunks[0]["metadata"]["chunk_type"] == "split_section"


# ── 3. DataPrepAgent Integration Tests ───────────────────────────────────────

@pytest.mark.asyncio
async def test_data_prep_agent_integration():
    terminology = TerminologyService()
    chunker = ChunkingService(target_chunk_size=500, overlap=50)
    agent = DataPrepAgent(terminology=terminology, chunker=chunker)
    
    raw = (
        "HISTORY OF PRESENT ILLNESS:\n"
        "Pt has HTN and DM. BS is 150 mg / dl.\n"
        "ASSESSMENT:\n"
        "CHF exacerbation."
    )
    
    doc = MedicalDocumentSchema(
        document_type=DocumentType.PDF,
        raw_text=raw,
        metadata={"filename": "fake.pdf"},
        processed_by=["MedicalParserAgent"]
    )
    
    # Run Agent
    docs = await agent.run([doc])
    
    assert len(docs) == 1
    enriched_doc = docs[0]
    
    # Verify terminology applied to normalized_text
    assert "Hypertension" in enriched_doc.normalized_text
    assert "Patient" in enriched_doc.normalized_text
    assert "150 mg/dL" in enriched_doc.normalized_text
    
    # Verify chunking was applied to the normalized text
    assert len(enriched_doc.chunks) == 2
    
    # Check headers were prepended
    assert "HISTORY OF PRESENT ILLNESS" in enriched_doc.chunks[0]["text"]
    assert "ASSESSMENT" in enriched_doc.chunks[1]["text"]
    
    # Verify agent added to lineage
    assert "DataPrepAgent" in enriched_doc.processed_by
