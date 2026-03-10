Currently under work!

Browser (App.jsx)
     │ POST /intake-batch (ZIP)
     ▼
FastAPI (main.py)
     │
     ▼
UnifiedBatchIntakeService  ← stages files to disk  data/User/<session_id>/
     │ POST /analyze-medical-session/{session_id}
     ▼
MedicalPipeline.analyze_session()
     │
     ├─ Phase 1+2 ──▶ PrivacyProtectionAgent
     │                 ├─ MedicalParserAgent  (PDF/DICOM/XLSX/image → raw_text)
     │                 ├─ VisionPerceptionAgent  (DICOM/image → visual_findings)
     │                 └─ PrivacyService  (Presidio PII/PHI masking)
     │                         ↓  List[MedicalDocumentSchema]  (anonymised)
     │
     ├─ Phase 3 ──▶ DataPrepAgent
     │                 ├─ TerminologyService  (acronym expansion, unit normalisation)
     │                 └─ ChunkingService  (semantic text → chunks[])
     │                         ↓  List[MedicalDocumentSchema]  (enriched)
     │
     ├─ Phase 4 ──▶ MedicalRAGAgent
     │                 ├─ EmbeddingService  (NVIDIA / sentence-transformers / hashing)
     │                 ├─ PGVectorStore — global_store  (Supabase/pgvector)
     │                 └─ PGVectorStore — patient_store  (session-scoped namespace)
     │                         ↓  List[RAG context hits]
     │
     ├─ Phase 5 ──▶ DiagnosticAgent
     │                 ├─ LLMService (Cerebras — Llama 3.1-8b)
     │                 ├─ Draft pass  → Devil's Advocate pass  (2-shot verification loop)
     │                 └─ NumericalGuardrailsExtractor  (validates lab values in JSON)
     │                         ↓  StructuredDiagnosis
     │
     └─ Phase 6 ──▶ ExplainabilityAgent
                       ├─ LLMService (Cerebras)
                       ├─ ExplanationService  (citation traceability mapping)
                       └─ CitationValidator  (blocks circular citations, flags contradictions)
                               ↓  FinalDiagnosticReport  → JSON response