# MediFlow - Multi-Agent Medical AI Pipeline

MediFlow is an AI-powered, enterprise-grade **Multi-Agent Medical Analysis Pipeline** that transforms raw patient medical documents into structured, explainable diagnostic reports. It routes every patient session through a sequential **8-phase agent pipeline**: Document Parsing → Vision Perception → Privacy Scrubbing → Data Preparation → Corrective RAG → Diagnostic Prediction → Explainability — all wired together by a master `MedicalPipeline` orchestrator.

The system features a **Corrective RAG (CRAG)** engine powered by LangGraph that self-grades retrieval quality and rewrites queries when evidence is insufficient, a **two-pass Clinical Verification Loop** (Draft + Devil's Advocate) for hallucination-resistant diagnosis, a **Vision-Language Model** integration for image analysis with Side-Swap safety validation, and a **Microsoft Presidio** PII/PHI anonymization layer with custom medical recognizers to ensure HIPAA-aligned data handling.

**[🚀 Live Demo](https://medi-flow-pied.vercel.app)**

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [How MediFlow Works](#how-mediflow-works)
  - [The 8-Phase Pipeline](#the-8-phase-pipeline)
  - [Corrective RAG (CRAG)](#corrective-rag-crag)
  - [Two-Pass Clinical Verification](#two-pass-clinical-verification)
  - [Vision-Language Model Integration](#vision-language-model-integration)
  - [PII/PHI Anonymization](#piiphi-anonymization)
  - [Numerical Guardrails](#numerical-guardrails)
  - [Clinical Chunking](#clinical-chunking)
  - [Session Isolation & Metadata Drift Prevention](#session-isolation--metadata-drift-prevention)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Overview

MediFlow enables healthcare organizations to upload a mixed batch of patient medical records — PDFs, DICOMs, scanned images, and XLSX lab reports — and receive a fully structured `FinalDiagnosticReport` with clinician-facing briefs, patient-friendly explanations, and a traceable evidence table linking every clinical claim back to the source chunk that supports it.

At the core is a **LangGraph CRAG StateGraph** that replaces single-shot RAG retrieval with a self-correcting loop: it embeds the query, retrieves from both a global clinical knowledge base and a per-session patient store (via pgvector), grades each chunk for relevance using the LLM, and rewrites the query up to two times if the graded context is insufficient. A **two-pass DiagnosticAgent** first drafts a hypothesis, then plays Devil's Advocate against it before committing to a `StructuredDiagnosis`. The **ExplainabilityAgent** then validates every citation — stripping any hallucinated chunk IDs from the evidence table before the report reaches the user.

Specialized for:

- Clinical notes and discharge summaries (PDF/text)
- Lab result spreadsheets (XLSX)
- Medical imaging studies (DICOM — X-ray, MRI, CT)
- Scanned handwritten notes and lab printouts (JPG/PNG via OCR)
- NIfTI neuroimaging volumes (MRI/CT)
- ZIP archives of mixed multi-modal patient files
- Medical literature and clinical guidelines (knowledge base ingestion)

## Features

### 🔬 8-Phase Multi-Agent Pipeline

Every `POST /analyze-medical-session/{session_id}` call passes through `MedicalPipeline.analyze_session()`, which wires six specialized agents in strict sequential order:

| Phase | Agent | Responsibility |
|---|---|---|
| **Phase 1** | `MedicalParserAgent` | Parses raw bytes into `MedicalDocumentSchema` — PDF (PyMuPDF), DICOM (pydicom), images (OCR), XLSX (openpyxl), ZIP archives |
| **Phase 1.5** | `VisionPerceptionAgent` | Routes DICOM/NIfTI/images to the NVIDIA VLM; applies orientation safety checks; returns `VisualFinding` objects |
| **Phase 2** | `PrivacyProtectionAgent` | Anonymizes all text, metadata, and filenames via Presidio; routes images to Vision before scrubbing DICOM tags |
| **Phase 3** | `DataPrepAgent` | Expands medical abbreviations, standardizes lab units, and splits text into clinical header-aware chunks |
| **Phase 4** | `MedicalRAGAgent` | Embeds chunks; runs CRAG retrieval against global knowledge + patient stores; returns graded context |
| **Phase 5** | `DiagnosticAgent` | Two-pass LLM: Draft hypothesis → Devil's Advocate critique → `StructuredDiagnosis` with urgency, contraindications, and confidence |
| **Phase 6** | `ExplainabilityAgent` | Generates clinician brief + patient explanation; validates all citations; strips hallucinated chunk IDs from evidence table |

**Pipeline Guardrails:**

- Tenacity retry policy (3 attempts, exponential backoff) wraps every LLM and external API call.
- Session-scoped atomic locks prevent Metadata Drift (cross-patient data contamination) during parallel processing.
- Pydantic model validators enforce hard safety rules (e.g., AKI diagnosis must list NSAIDs/IV contrast as contraindications).

### 🔄 Corrective RAG (CRAG)

MediFlow's RAG layer uses a **LangGraph `StateGraph`** instead of a single retrieval call:

1. **Embed & Retrieve**: Query is embedded and searched against both the `mediflow_knowledge` table (permanent global clinical literature) and the `mediflow_patient_vectors` table (session-scoped patient records) via pgvector.
2. **Grade Documents**: The LLM scores each retrieved chunk as `relevant` or `not relevant`. Irrelevant chunks are moved to `rejected_hits`.
3. **Rewrite Query** (on failure): If graded context is insufficient, the LLM rewrites the query with different clinical phrasing and loops back. Maximum 2 rewrites.
4. **Flag Low Confidence**: If retrieval quality is still poor after all retries, `low_confidence=True` is propagated to the `DiagnosticAgent` to increase `missing_data_points` verbosity.
5. **Temporal Decay**: More recent patient records receive a score boost via a recency decay function — preventing old labs from outranking fresh findings.

### 🧠 Two-Pass Clinical Verification Loop

`DiagnosticAgent` runs a deliberate **Draft → Devil's Advocate** two-shot verification:

1. **Draft Pass** (`temperature=0.3`): Generates an initial `StructuredDiagnosis` JSON from patient text, numerical guardrails, and RAG context.
2. **Devil's Advocate Pass** (`temperature=0.5`): Critically attacks the draft, specifically seeking evidence for the top two differential diagnoses that would *disprove* the primary diagnosis. Outputs a revised, corrected JSON.
3. **Clinical-Visual Congruence**: If Phase 1.5 visual findings contradict Phase 3 lab data (e.g., imaging suggests pneumonia but WBC is normal), `clinical_visual_congruence=False` is forced and the discordance is surfaced in the primary or differential diagnosis.
4. **Pydantic Safety Validators**: Hard-coded model validators reject structurally unsafe outputs — e.g., an AKI diagnosis that doesn't list NSAIDs as a contraindication will raise a `ValueError` before the report is built.

### 👁️ Vision-Language Model Integration

`VisionPerceptionAgent` routes medical images through a dedicated processing pipeline before sending them to the NVIDIA VLM (Gemma-3-27b-it):

- **DICOM Processing**: `DicomProcessor` extracts pixel arrays, applies Hounsfield Unit rescaling (RescaleSlope × pixel + RescaleIntercept), and normalizes volumetric data to [0, 1].
- **NIfTI Support**: Full 3D neuroimaging volumes (`.nii`, `.nii.gz`) are loaded via NiBabel, normalized, and middle-representative slices are sampled for VLM inference.
- **Image Quality Gate**: Standard JPG/PNG inputs pass through a `validate_image_quality()` blur and resolution check — blurry or corrupted images are rejected with a `LOW_IMAGE_QUALITY` error before the API call is made.
- **Longitudinal Context**: A `ComparisonBuffer` allows historical image findings to be passed alongside the current scan for trend-aware analysis.
- **Side-Swap Safety Check**: `DicomProcessor.validate_side_orientation()` cross-references the VLM's claimed laterality (Left/Right) against the DICOM `ImageOrientationPatient` header — raising a `CriticalOrientationMismatch` exception if a discrepancy is detected.
- **Confidence Threshold**: VLM responses with confidence below 0.6 are flagged in `VisualFinding` for downstream uncertainty propagation.

### 🔒 PII/PHI Anonymization

`PrivacyProtectionAgent` + `PrivacyService` run every document through **Microsoft Presidio** before any data touches the LLM or vector store:

- **spaCy NER**: Uses `en_core_web_sm` (12 MB) to detect names, locations, and identifiers in running text.
- **Custom Medical Recognizers**:
  - `MEDICAL_ID`: Regex recognizer for MRN, Patient ID, Case No., Accession No., and Report No. patterns.
  - `PERSON` prefix booster: Explicit patterns for "Dr.", "Patient:", "Mrs.", "Pt:", etc. to improve detection of clinical name patterns that NER misses.
  - **Medical Allow-list**: Drugs and clinical terms (Aspirin, Creatinine, Lisinopril, etc.) are whitelisted to prevent over-anonymization — a common failure mode of generic Presidio deployments.
- **Scope**: Anonymizes raw text, metadata dictionaries (preserving safe DICOM keys like `Modality`, `BodyPartExamined`), tabular lab data, and even filenames (underscores/hyphens converted to spaces for better NER recall first).
- **DICOM Tag Scrubbing**: Only non-identifying technical tags (pixel spacing, window levels, modality, body part) are preserved; all patient-identifying DICOM tags are removed.

### 📊 Numerical Guardrails

`NumericalGuardrailsExtractor` creates a "hard attention" Markdown table of lab values to prevent the LLM from hallucinating numbers:

- Extracts 6 key biomarkers from free text via compiled regex: Glucose, Creatinine, Hemoglobin, WBC, Potassium, Sodium.
- Normalizes units (e.g., Glucose mmol/L → mg/dL via ×18.0182).
- Flags out-of-range values with `!! [value] (High/Low) !!` notation in the Markdown output.
- Supports **temporal trend analysis**: Multiple values for the same biomarker are preserved with timestamps and sorted chronologically — allowing the LLM to reason about trends (e.g., rising creatinine over 3 days).
- Output is injected as a structured table into both the Draft and Devil's Advocate prompts to anchor numerical reasoning.

### ✂️ Clinical Chunking

`ChunkingService` applies **clinical header-aware splitting** before embedding:

- Detects 20+ standard clinical section headers (CHIEF COMPLAINT, HPI, ASSESSMENT AND PLAN, LABORATORY DATA, MEDICATIONS, etc.) using a compiled multi-line regex.
- Splits text at section boundaries to preserve clinical context integrity — lab values stay with their section, not split across chunk boundaries.
- Falls back to character-count overlapping chunks (default: 1500 chars, 200 overlap) for sections without standard headers.

### 🔐 Session Isolation & Metadata Drift Prevention

- Every patient upload creates a fully isolated session directory: `data/User/{session_id}/` with subfolders for `dicom/`, `pdf/`, `images/`, `spreadsheets/`.
- `SessionManager` + `PatientSessionState` enforce **atomic session locks** (`asyncio.Lock`) that verify the patient ID on every file within a session — rejecting any file whose patient ID doesn't match the session owner and raising a `MetadataDrift` error.
- pgvector patient store is namespaced per session: `mediflow_patient_vectors` rows are scoped by `session_id`, and `delete_all()` atomically clears stale data at session start to prevent leftover vectors from a prior upload.
- Session cleanup fires on tab close via `navigator.sendBeacon` to `DELETE /session/{session_id}/cleanup`.

### 📋 Multi-Format Document Intake

`UnifiedBatchIntakeService` + `MedicalParserAgent` handle every supported medical document format:

| Format | Extension | Extractor | Notes |
|---|---|---|---|
| Medical PDFs | `.pdf` | PyMuPDF (fitz) | Text-based; raises error on scanned PDFs to trigger OCR path |
| DICOM Studies | `.dcm` | pydicom | Pixel array + metadata tags extraction |
| Scanned/Photos | `.jpg`, `.png` | Pillow + pytesseract OCR | Quality gate applied before OCR |
| Lab Spreadsheets | `.xlsx` | openpyxl / pandas | All sheets to text + tabular records |
| DICOM Archives | `.zip` | stdlib zipfile | Discovers + stages all `.dcm` files inside |
| Mixed Batches | `.zip` | `UnifiedBatchIntakeService` | Separates by type into subfolders |
| NIfTI Volumes | `.nii`, `.nii.gz` | NiBabel | 3D neuroimaging; slice-sampled for VLM |

**Validation:** Max 500 files per batch, max 2 GB uncompressed total. Duplicate files (by SHA-256 hash) are skipped. Unsupported formats are staged to an `unknown/` folder for audit without blocking the rest of the batch.

### 🏥 Citation-Validated Explainability

`ExplainabilityAgent` generates a `FinalDiagnosticReport` with full **citation traceability**:

- Every claim in the `evidence_table` must cite at least one `source_chunk_id` from the RAG context. Claims without valid chunk IDs are re-classified as `Inferred_Reasoning`.
- **Hallucination Blocking**: A `_validate_citations()` hard guardrail strips any chunk ID not present in the actual retrieval context — the LLM cannot invent citations.
- **Circular Citation Blocking**: The system prompt explicitly forbids citing Phase 5 output as evidence; only the underlying RAG chunks are valid sources.
- **Contradictory Finding Surfacing**: If a retrieved chunk contradicts the diagnosis, the evidence entry is flagged `is_contradictory=True` for UI highlighting.
- Two output tracks: a **clinician brief** (technical, structured) and a **patient explanation** (empathetic, jargon-free, second-person "you").
- Hard-coded legal disclaimer appended to every report.

### 📦 Global Knowledge Base Ingestion

`ingest_global.py` populates the permanent `mediflow_knowledge` pgvector table from `data/knowledge_base/`:

- Supported formats: `.md`, `.txt`, `.pdf`, `.csv`, `.docx`
- Uses the same `ChunkingService` + `EmbeddingService` as the patient pipeline for consistent vector space alignment.
- Idempotent: can be re-run safely; designed to be triggered manually when new clinical guidelines are added.

## Project Structure

```text
.
├── main.py                         # FastAPI app entrypoint: lifespan, CORS, all route definitions
├── requirements.txt                # Full Python dependencies
├── Dockerfile                      # Python 3.11-slim + Tesseract OCR
├── docker-compose.yml              # PostgreSQL 16 + pgvector + FastAPI backend
├── server.bat                      # Windows one-command local launcher
├── render.yaml                     # Render.com deployment blueprint (Docker)
├── .env.example                    # Environment variable template
├── src/
│   ├── agents/
│   │   ├── parser_agent.py         # Phase 1: Multi-format parser; routes bytes to extractors
│   │   ├── vision_perception_agent.py  # Phase 1.5: NVIDIA VLM wrapper; Side-Swap safety check
│   │   ├── privacy_agent.py        # Phase 2: Presidio anonymizer; Vision routing for images
│   │   ├── data_prep_agent.py      # Phase 3: Terminology normalization + clinical chunking
│   │   ├── medical_rag_agent.py    # Phase 4: CRAG retrieval; dual global+patient pgvector stores
│   │   ├── diagnostic_agent.py     # Phase 5: Draft+Advocate 2-pass; Numerical Guardrails
│   │   ├── explainability_agent.py # Phase 6: Evidence traceability; citation validation
│   │   └── vision/
│   │       ├── router.py           # Modality router: DICOM / standard image / NIfTI
│   │       ├── dicom_processor.py  # DICOM pixel extraction; HU rescaling; orientation validation
│   │       └── quality_gate.py     # Image blur/resolution quality gate for JPG/PNG
│   ├── core/
│   │   ├── base_agent.py           # Abstract BaseAgent with structured logger
│   │   └── config.py               # Pydantic Settings: all env vars with validation aliases
│   ├── models/
│   │   ├── medical_document.py     # MedicalDocumentSchema: canonical pipeline data structure
│   │   ├── diagnostic_models.py    # StructuredDiagnosis, FinalDiagnosticReport, EvidenceMap
│   │   ├── rag_models.py           # RAG request/response Pydantic schemas
│   │   ├── intake_manifest.py      # IntakeManifest + StagedFile models
│   │   └── document_type_map.py    # Extension → DocumentType single source of truth
│   ├── pipelines/
│   │   └── medical_pipeline.py     # Phase 7 Master Orchestrator: wires all agents; retry policy
│   ├── rag/
│   │   ├── embedding_service.py    # NVIDIA API embedder for clinical RAG vectors
│   │   ├── pgvector_store.py       # PostgreSQL + pgvector vector store; dual-table architecture
│   │   ├── crag_graph.py           # LangGraph CRAG StateGraph: retrieve → grade → rewrite loop
│   │   ├── ingest_global.py        # CLI: ingests data/knowledge_base/ into mediflow_knowledge table
│   │   ├── common.py               # Shared helpers: chunk_id builder, flatten_record, utc_now_iso
│   │   └── global_store/           # Runtime directory for FAISS-compat metadata (legacy path)
│   ├── services/
│   │   ├── llm_service.py          # Cerebras Cloud SDK wrapper; JSON-structured generation
│   │   ├── privacy_service.py      # Presidio engine: custom medical recognizers + allow-list
│   │   ├── chunking_service.py     # Clinical header-aware text splitter; overlapping fallback
│   │   ├── terminology_service.py  # 100+ medical abbreviation expansions; lab unit normalization
│   │   ├── numerical_extractor.py  # Lab value extractor; reference ranges; Markdown table formatter
│   │   ├── explanation_service.py  # Citation traceability helpers for ExplainabilityAgent
│   │   ├── extractors.py           # Stateless format extractors: PDF, DICOM, image OCR, XLSX, ZIP
│   │   ├── session_context.py      # PatientSessionState + SessionManager: atomic Metadata Drift guard
│   │   └── zip_intake.py           # UnifiedBatchIntakeService: stages files by type; ZIP extraction
│   └── utils/
│       └── logger.py               # structlog-based rotating logger
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main UI: session lifecycle, file upload, analysis trigger,
│   │   │                           #   real-time status badges, diagnostic report rendering
│   │   ├── index.css               # Tailwind CSS entry
│   │   └── main.jsx                # Vite entry point
│   ├── index.html
│   ├── vite.config.js              # Vite dev proxy → localhost:8000
│   ├── tailwind.config.js
│   └── vercel.json                 # Vercel SPA routing config
├── data/
│   ├── knowledge_base/             # Clinical guidelines / medical literature (gitignored)
│   └── User/                       # Per-session patient files (gitignored)
│       └── {session_id}/
│           ├── dicom/
│           ├── pdf/
│           ├── images/
│           ├── spreadsheets/
│           └── manifest.json       # IntakeManifest written by UnifiedBatchIntakeService
└── tests/
```
## Installation

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 16 with the `pgvector` extension (or Docker)
- Tesseract OCR binary (for scanned image/PDF support)
- NVIDIA API Key (embeddings + vision VLM)
- Cerebras API Key (LLM — diagnosis, CRAG grading, explainability)

### Steps

1. **Clone & Setup Backend**:

   ```bash
   git clone <repo_url>
   cd MediFlow
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Setup Frontend**:

   ```bash
   cd frontend
   npm install
   ```

3. **Database Setup** (manual):

   ```sql
   CREATE DATABASE mediflow;
   -- Connect to mediflow, then:
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Environment Configuration**:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URL
   ```

5. **Ingest Global Knowledge Base** (run once after setup):

   ```bash
   # Place clinical guidelines / medical literature in data/knowledge_base/
   python -m src.rag.ingest_global
   ```

6. **Quick Start (Windows)**:

   ```bash
   .\server.bat
   ```

7. **Quick Start (Docker)**:

   ```bash
   docker compose up --build
   # The backend starts at http://localhost:8000
   # Start the frontend separately:
   cd frontend && npm run dev   # http://localhost:5173
   ```

## Usage

### Workflow

1. **Create a session** — `POST /session/create` → returns `session_id`
2. **Upload files** — `POST /intake` or `POST /intake-batch` (ZIP) with `session_id`
3. **Run analysis** — `POST /analyze-medical-session/{session_id}` → returns `FinalDiagnosticReport`
4. **Ingest global knowledge** (admin) — `python -m src.rag.ingest_global`

### Frontend

Open `http://localhost:5173` after running `server.bat` or `npm run dev`. The React UI allows:

- Drag-and-drop / multi-file upload (PDFs, DICOMs, ZIPs, images, XLSX)
- Per-file real-time status badges (Uploading → Scanning → Scrubbed → Analyzed / Error)
- Urgency-color-coded diagnostic report panel (Low / Medium / High / Critical)
- Clinician brief, patient explanation, and full evidence table rendering
- Session auto-cleanup on tab close

### Run Tests

```bash
pytest tests/ -v
```

## Configuration

### Environment Variables

```bash
# ── NVIDIA (embeddings + vision VLM) ──────────────────────────────────────
NVIDIA_EMBED_API_KEY=nvapi-xxxxxxxxxxxx   # For nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx         # For google/gemma-3-27b-it vision model

# ── Cerebras (LLM — diagnosis, CRAG, explainability) ──────────────────────
CEREBRAS_API_KEY=csapi-xxxxxxxxxxxx       # Llama 3.1-8b via Cerebras Cloud SDK

# ── PostgreSQL + pgvector ──────────────────────────────────────────────────
# Local: postgresql://postgres:postgres@localhost:5432/mediflow
# Supabase: postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres
PGVECTOR_DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/mediflow

# ── Supabase Storage (optional — for patient file persistence) ────────────
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.example

# ── Application ───────────────────────────────────────────────────────────
ENVIRONMENT=production                    # dev | production
LOG_LEVEL=INFO
ALLOWED_ORIGIN=https://your-app.vercel.app  # Added to CORS allow_origins
```

### Docker Compose

```bash
docker compose up --build
```

Starts PostgreSQL 16 + pgvector on port 5432 and the FastAPI backend on port 8000. The frontend still runs via Vite (`cd frontend && npm run dev`).

### Render.com

Deploy via `render.yaml` (blueprint):

- Uses the `Dockerfile` (Python 3.11-slim + Tesseract; no PyTorch — embeddings use NVIDIA API).
- Set `NVIDIA_EMBED_API_KEY`, `NVIDIA_API_KEY`, `CEREBRAS_API_KEY`, `PGVECTOR_DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, and `ALLOWED_ORIGIN` as Render environment variables.
- Frontend is deployed separately on Vercel (`frontend/vercel.json`).

## How MediFlow Works

### The 8-Phase Pipeline

Every `POST /analyze-medical-session/{session_id}` call passes through `MedicalPipeline.analyze_session()`:

```
Browser (App.jsx)
     │ POST /intake-batch (ZIP or multi-file)
     ▼
FastAPI (main.py)
     │
     ▼
UnifiedBatchIntakeService  ← stages files to disk  data/User/<session_id>/
     │
     │ POST /analyze-medical-session/{session_id}
     ▼
MedicalPipeline.analyze_session()
     │
     ├─ Phase 1+1.5+2 ──▶ PrivacyProtectionAgent
     │                      ├─ MedicalParserAgent  (PDF/DICOM/XLSX/image → raw_text)
     │                      ├─ VisionPerceptionAgent  (DICOM/NIfTI/image → visual_findings)
     │                      └─ PrivacyService  (Presidio PII/PHI masking)
     │                              ↓  List[MedicalDocumentSchema]  (anonymized)
     │
     ├─ Phase 3 ──▶ DataPrepAgent
     │                ├─ TerminologyService  (acronym expansion, unit normalization)
     │                └─ ChunkingService  (clinical header-aware → chunks[])
     │                        ↓  List[MedicalDocumentSchema]  (enriched + chunked)
     │
     ├─ Phase 4 ──▶ MedicalRAGAgent
     │                ├─ EmbeddingService  (NVIDIA API)
     │                ├─ PGVectorStore — mediflow_knowledge  (global clinical literature)
     │                └─ PGVectorStore — mediflow_patient_vectors  (session-scoped)
     │                     └─ CRAG StateGraph  (retrieve → grade → rewrite → flag)
     │                        ↓  List[RAG context hits]
     │
     ├─ Phase 5 ──▶ DiagnosticAgent
     │                ├─ LLMService (Cerebras — Llama 3.1-8b)
     │                ├─ Draft pass  →  Devil's Advocate pass  (2-shot loop)
     │                └─ NumericalGuardrailsExtractor  (anchors lab values for LLM)
     │                        ↓  StructuredDiagnosis
     │
     └─ Phase 6 ──▶ ExplainabilityAgent
                      ├─ LLMService (Cerebras)
                      ├─ ExplanationService  (citation traceability mapping)
                      └─ CitationValidator  (strips hallucinated chunk IDs)
                              ↓  FinalDiagnosticReport  → JSON response
```

### Corrective RAG (CRAG)

`MedicalRAGAgent` builds and runs a LangGraph `StateGraph` with four nodes:

1. **`embed_and_retrieve`**: Embeds the query, runs pgvector cosine similarity search against both the global knowledge table and the per-session patient table. Merges results. Applies a recency decay function to boost fresher patient records.
2. **`grade_documents`**: Calls the LLM with each chunk to classify it as `relevant` or `not relevant`. Passes only high-quality chunks to the next stage.
3. **`rewrite_query`**: If no relevant chunks were found, sends the query + rejected chunks back to the LLM to generate a clinically rephrased alternative. Loops back to `embed_and_retrieve`. Max 2 rewrites (`MAX_RETRIES`).
4. **`flag_low_confidence`**: If all retries are exhausted, sets `low_confidence=True` in the state — the `DiagnosticAgent` uses this flag to increase `missing_data_points` output.

Graph output: `{"results": [...], "low_confidence": bool, "rewrite_count": int}`.

### Two-Pass Clinical Verification

`DiagnosticAgent.run()` executes two sequential LLM calls:

1. **Draft Pass**: System prompt = clinical AI expert + strict JSON schema. User prompt includes patient demographics, normalized clinical text, numerical guardrails Markdown table, RAG context, and (if present) visual findings from Phase 1.5. Temperature = 0.3.
2. **Devil's Advocate Pass**: Same schema, but the system prompt now instructs the LLM to *challenge* the draft and find evidence for differentials. Draft JSON is appended to the user prompt. Temperature = 0.5.
3. **JSON Parsing + Pydantic Validation**: The response is parsed and validated as `StructuredDiagnosis`. If a low-confidence score is present with fewer than 3 `missing_data_points`, the validator raises an error and the step retries with higher temperature.
4. **Clinical-Visual Mismatch**: If `visual_findings` contradict lab data, `clinical_visual_congruence=False` is forced into the output regardless of what the LLM returns; the discordance is reflected in the primary or differential diagnoses.

### Vision-Language Model Integration

`VisionPerceptionAgent.analyze_image()` pipeline:

1. `MedicalImageRouter.route_and_process()` detects modality by extension (`.dcm` → `DicomProcessor`; `.jpg/.png` → `validate_image_quality()` then Pillow; `.nii/.nii.gz` → NiBabel).
2. For DICOM: extracts pixel array, applies HU rescaling, normalizes to [0,1], and extracts modality/body part/orientation metadata.
3. For 3D volumes: `get_representative_slices()` samples middle slices for the VLM.
4. Base64-encodes the image(s) and constructs a `content_blocks` payload (text prompt + image_url blocks) for the NVIDIA API.
5. `_call_vlm()` sends the multimodal payload to `https://integrate.api.nvidia.com/v1/chat/completions` using `google/gemma-3-27b-it`.
6. Parses the response into a `VisualFinding` object (modality, preliminary report, key observations, confidence).
7. `validate_side_orientation()` cross-references claimed laterality against DICOM header — raises `CriticalOrientationMismatch` on divergence.

### PII/PHI Anonymization

`PrivacyService` is initialized once at startup (singleton pattern) and reused across all sessions:

- Configures Presidio with `en_core_web_sm` (not `en_core_web_lg`) to stay within Render's RAM limits while retaining sufficient NER accuracy.
- Adds three custom recognizers on top of Presidio's default suite:
  - `MEDICAL_ID` regex: catches MRN, Report No., Case No., Accession No.
  - `PERSON` prefix booster: catches clinical name patterns (Dr., Pt:, Patient Name:).
  - `CLINICAL_TERM` allow-list: prevents common drug names and biomarkers from being redacted as locations or names.
- `anonymize_text()`, `anonymize_metadata()`, and `anonymize_tabular_data()` handle different data structures.
- Safe DICOM keys (Modality, BodyPartExamined, PixelSpacing, WindowCenter, etc.) are explicitly whitelisted and preserved.

### Numerical Guardrails

`NumericalGuardrailsExtractor` runs before each LLM prompt:

- Pre-compiles one regex per biomarker (including all known aliases) at construction time.
- `process_historical_context()` merges lab values from the current clinical note AND from RAG chunks — giving the LLM a combined, timestamped lab history.
- `format_markdown_table()` renders extracted values as a structured table with reference range columns and `!! High !!` / `!! Low !!` flag columns.
- The table is injected as the `--- NUMERICAL GUARDRAILS (LABS) ---` section in both LLM passes.

### Clinical Chunking

`ChunkingService.chunk_document()`:

1. Scans the text for any of 20+ clinical section headers using a compiled case-insensitive multiline regex.
2. Splits on header positions — each section becomes an independent chunk candidate.
3. Sections exceeding `target_chunk_size` (default: 1500 chars) are further split with `overlap` (default: 200 chars) character-level sliding windows.
4. Each chunk is returned as a dict: `{"section": "<header>", "text": "<content>", "chunk_index": int}`.

### Session Isolation & Metadata Drift Prevention

`SessionManager` + `PatientSessionState`:

- A global dict maps `session_id → PatientSessionState` (each with its own `asyncio.Lock`).
- `verify_or_set_patient(new_patient_id)` atomically sets the expected patient for a session on the first call, then rejects any subsequent file with a different patient ID — preventing cross-patient contamination in batch uploads.
- `atomic_session_lock` async context manager (used by the intake route) acquires the session lock for the entire duration of a staged intake operation.
- On analysis completion (or error), the session patient lock is released and session state is cleaned up.

## Architecture

### Agent Layer

| Component | File | Responsibility |
|---|---|---|
| **MedicalParserAgent** | `src/agents/parser_agent.py` | Entry-point; dispatches raw bytes to format extractors; returns `MedicalDocumentSchema` |
| **VisionPerceptionAgent** | `src/agents/vision_perception_agent.py` | Routes images by modality; calls NVIDIA VLM; applies Side-Swap safety check |
| **PrivacyProtectionAgent** | `src/agents/privacy_agent.py` | Orchestrates Phase 1+1.5+2; runs Presidio on every text field and filename |
| **DataPrepAgent** | `src/agents/data_prep_agent.py` | Acronym expansion → unit normalization → clinical chunking |
| **MedicalRAGAgent** | `src/agents/medical_rag_agent.py` | Dual pgvector stores; builds + runs CRAG LangGraph; manages patient store lifecycle |
| **DiagnosticAgent** | `src/agents/diagnostic_agent.py` | Draft + Advocate 2-pass; Numerical Guardrails; Pydantic safety validators |
| **ExplainabilityAgent** | `src/agents/explainability_agent.py` | Evidence traceability; citation hallucination blocking; patient-friendly output |
| **MedicalPipeline** | `src/pipelines/medical_pipeline.py` | Master orchestrator; tenacity retry policy; visual findings merge across documents |

### Core Modules

| Component | File | Responsibility |
|---|---|---|
| **EmbeddingService** | `src/rag/embedding_service.py` | NVIDIA API embedder; batched requests |
| **PGVectorStore** | `src/rag/pgvector_store.py` | pgvector-backed store; dual-table (knowledge / patient); FAISS-compatible interface |
| **CRAG Graph** | `src/rag/crag_graph.py` | LangGraph StateGraph: embed → grade → rewrite → flag low confidence |
| **LLMService** | `src/services/llm_service.py` | Cerebras Cloud SDK; JSON-mode structured generation; temperature-controlled passes |
| **PrivacyService** | `src/services/privacy_service.py` | Presidio Analyzer + Anonymizer; 3 custom recognizers; medical allow-list |
| **ChunkingService** | `src/services/chunking_service.py` | Clinical header regex splitter + overlapping fallback |
| **TerminologyService** | `src/services/terminology_service.py` | 100+ medical abbreviation expansions; lab unit normalization |
| **NumericalGuardrailsExtractor** | `src/services/numerical_extractor.py` | Lab value regex extraction; unit normalization; flagged Markdown table |
| **Extractors** | `src/services/extractors.py` | Stateless format extractors: PyMuPDF, pydicom, pytesseract OCR, openpyxl |
| **UnifiedBatchIntakeService** | `src/services/zip_intake.py` | Stages files by type; ZIP extraction; SHA-256 deduplication; 2 GB bomb guard |
| **SessionContext** | `src/services/session_context.py` | Atomic session locks; patient ID verification; Metadata Drift prevention |
| **MedicalImageRouter** | `src/agents/vision/router.py` | Modality detection + preprocessing; NIfTI flattening; quality gate |
| **DicomProcessor** | `src/agents/vision/dicom_processor.py` | HU rescaling; orientation extraction; Side-Swap validation |

## API Reference

### Session & Pipeline Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/session/create` | Create an isolated session UUID |
| `DELETE` | `/session/{session_id}/cleanup` | Delete all staged files for a session (called on tab close) |
| `GET` | `/health` | Basic health check — returns `{"status": "ok", "environment": "..."}` |
| `POST` | `/intake` | Upload one or more files to a session; returns `IntakeManifest` |
| `POST` | `/intake-batch` | Upload a ZIP archive of mixed medical files; returns `IntakeManifest` |
| `POST` | `/analyze-medical-session/{session_id}` | Run the full 8-phase pipeline; returns `FinalDiagnosticReport` |

### RAG Management Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/rag/index-patient` | Embed + store a patient's prepared documents into the patient pgvector table |
| `POST` | `/rag/retrieve` | Direct retrieval query against a session's patient store + global knowledge base |
| `DELETE` | `/rag/cleanup/{session_id}` | Remove all patient vectors for a session from pgvector |

### Response Schema — `FinalDiagnosticReport`

```json
{
  "session_id": "uuid",
  "clinician_brief": "Technical summary for the physician...",
  "patient_explanation": "Jargon-free, empathetic explanation for the patient...",
  "evidence_table": [
    {
      "statement": "Clinical claim",
      "source_chunk_ids": ["chunk_abc123"],
      "source_type": "Patient_Record | Global_Literature | Inferred_Reasoning",
      "is_contradictory": false,
      "confidence_of_mapping": 0.91
    }
  ],
  "citations": ["AMA-formatted citation if global literature was used"],
  "disclaimer": "This is an AI-generated clinical synthesis...",
  "structured_diagnosis": {
    "primary_diagnosis": "string",
    "differential_diagnoses": ["string"],
    "supporting_evidence": ["string"],
    "visual_evidence": { "modality": "...", "key_observations": [] } ,
    "clinical_visual_congruence": true,
    "urgency_level": "Low | Medium | High | Critical",
    "missing_data_points": ["string"],
    "contraindications": ["string"],
    "confidence_score": 0.85
  }
}
```

## Requirements

- Python 3.11+
- Node.js 20+
- PostgreSQL 16 with `pgvector` extension (or Docker)
- Tesseract OCR binary — required for scanned image/PDF support
- NVIDIA API Key — for `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` embeddings and `google/gemma-3-27b-it` vision model
- Cerebras API Key — for `llama3.1-8b` LLM (diagnosis, CRAG grading, explainability)
- Supabase Project — optional; for managed PostgreSQL + pgvector and file storage
- Docker — optional, for containerized deployment

## Troubleshooting

**pgvector Extension Not Found**

Ensure `CREATE EXTENSION IF NOT EXISTS vector;` has been run in your `mediflow` database. If using Docker, the `pgvector/pgvector:pg16` image includes the extension — it is created automatically on first connection by `PGVectorStore._ensure_schema()`.

**Tesseract Not Found**

Install the Tesseract binary for your OS. On Debian/Ubuntu: `apt-get install tesseract-ocr tesseract-ocr-eng`. On Mac: `brew install tesseract`. The Docker image already includes it. Scanned JPG/PNG uploads will fail with a pytesseract error without it.

**spaCy Model Not Found**

Run `python -m spacy download en_core_web_sm` after `pip install -r requirements.txt`. The Docker image downloads it automatically during build. The `en_core_web_lg` model (auto-installed by presidio-analyzer) is uninstalled during the Docker build to save ~400 MB RAM.

**NVIDIA API Timeout**

Increase `rag_embedding_request_timeout_seconds` in `config.py` (default: 60s) or via the `EMBEDDING_TIMEOUT` env var. The NVIDIA embedding API is called per-batch (default: 32 chunks per call) — reduce `rag_embedding_nvidia_max_batch_size` if hitting rate limits on free-tier keys.

**CRAG Graph: No Relevant Chunks Found**

Check that `python -m src.rag.ingest_global` was run and that `data/knowledge_base/` contains at least some clinical reference material. The `mediflow_knowledge` table can be inspected with:
```sql
SELECT COUNT(*), namespace FROM mediflow_knowledge GROUP BY namespace;
```

**Cerebras API Key Missing**

Without `CEREBRAS_API_KEY`, `LLMService` will raise `ValueError: Cerebras API Key is missing.` on the first LLM call. CRAG grading, diagnostic prediction, and explainability all require it. Set the key in `.env` and restart.

**DICOM Orientation Mismatch Error**

A `CriticalOrientationMismatch` exception means the VLM reported a finding on a side that contradicts the DICOM header. This is a safety feature. Review the raw VLM output and DICOM metadata in the logs to determine if the image has a non-standard orientation encoding.

**Session Metadata Drift Warning**

If you see `metadata_drift_detected` in logs, a file in the batch had a different Patient ID from the session owner. The file is rejected. Ensure all files in a single upload batch belong to the same patient.

**OOM on Render Starter**

The Docker image uses `en_core_web_sm` and NVIDIA API embeddings (no local model weights) to stay within ~512 MB RAM. If memory still exceeds limits, reduce `rag_embedding_nvidia_max_batch_size` (default: 32).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
