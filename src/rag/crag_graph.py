"""
src/rag/crag_graph.py
----------------------
Corrective RAG (CRAG) StateGraph — replaces the single-shot retrieve() call
in MedicalRAGAgent with a self-correcting retrieval loop.

Flow
----
query
  │
  ▼
[embed_and_retrieve]  ──── pgvector global_store + patient_store
  │
  ▼
[grade_documents]  ←── LLM grades each chunk: relevant? yes/no
  │
  ├── pass  ──▶  END  (return high-quality context)
  │
  └── fail  ──▶  [rewrite_query]  (max MAX_RETRIES times)
                        │
                        ▼
                  [embed_and_retrieve]   ← loops back
                        │
                        ▼
                  [grade_documents]
                        │
                        └── still fail  ──▶  [flag_low_confidence]  ──▶  END

The graph returns a dict with keys:
    "results"         : List[Dict]  — formatted retrieval hits
    "low_confidence"  : bool        — True if retrieval quality was poor
    "rewrite_count"   : int         — how many query rewrites were performed
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np

from langgraph.graph import END, StateGraph

from src.rag.common import flatten_record
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2

# ── State schema ──────────────────────────────────────────────────────────────

class CRAGState(TypedDict):
    # Inputs
    query: str
    session_id: Optional[str]
    top_k_patient: int
    top_k_global: int
    top_k_total: int

    # Runtime
    rewrite_count: int
    raw_hits: List[Dict[str, Any]]        # (distance, metadata) pairs before grading
    graded_hits: List[Dict[str, Any]]     # hits that passed relevance grading
    rejected_hits: List[Dict[str, Any]]   # hits the grader marked irrelevant

    # Output
    results: List[Dict[str, Any]]
    low_confidence: bool


# ── Node helpers (pure functions, injected deps via closure) ──────────────────

def _format_hit(distance: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Identical to MedicalRAGAgent._format_retrieval_hit."""
    from src.rag.common import build_chunk_id
    normalized = flatten_record(metadata)
    section = str(normalized.get("section") or "GENERAL")
    source_file = str(normalized.get("source_file") or "unknown")
    text = str(normalized.get("text") or "")
    chunk_id = str(normalized.get("chunk_id") or build_chunk_id(source_file, section, text))
    origin = str(normalized.get("origin") or "unknown")
    document_id = str(normalized.get("document_id") or "")
    session_id = normalized.get("session_id")
    metadata_payload = normalized.get("metadata") or {}
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source_file": source_file,
        "origin": origin,
        "section": section,
        "text": text,
        "session_id": session_id,
        "distance": float(distance),
        "l2_distance": float(distance),
        "metadata": metadata_payload,
    }


def _apply_decay(
    raw_hits: List[tuple[float, Dict[str, Any]]],
) -> list[tuple[float, Dict[str, Any]]]:
    """Temporal decay weighting — same logic as the old MedicalRAGAgent.retrieve()."""
    from datetime import datetime
    weighted = []
    for distance, metadata in raw_hits:
        ts_str = metadata.get("metadata", {}).get("document_timestamp")
        penalty = 1.0
        if ts_str and ts_str != "Historical":
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
                days_old = (now - dt).days
                if days_old > 0:
                    penalty = 1.0 + min(0.5, (days_old / 30.0) * 0.05)
            except Exception:
                penalty = 1.2
        weighted.append((distance * penalty, metadata))
    weighted.sort(key=lambda x: x[0])
    return weighted


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_crag_graph(
    embedder,
    global_store,
    get_patient_store_fn,
    llm_service,
    cache_service=None,
    retrieval_cache_ttl_seconds: int = 300,
):
    """
    Constructs and compiles the CRAG StateGraph.

    Parameters
    ----------
    embedder:
        EmbeddingService instance — provides embed_text().
    global_store:
        PGVectorStore for the medical knowledge base.
    get_patient_store_fn:
        Callable[[str], Optional[PGVectorStore]] — returns the session patient store.
    llm_service:
        LLMService instance — used for document grading and query rewriting.

    Returns
    -------
    CompiledGraph ready to call via .invoke(state).
    """

    # ── Node 1: embed_and_retrieve ────────────────────────────────────────────

    def embed_and_retrieve(state: CRAGState) -> CRAGState:
        query = state["query"]
        session_id = state.get("session_id")
        top_k_patient = state.get("top_k_patient", 5)
        top_k_global = state.get("top_k_global", 5)

        logger.info(
            "crag_retrieve",
            query=query[:80],
            session_id=session_id,
            rewrite_count=state.get("rewrite_count", 0),
        )

        patient_store = get_patient_store_fn(session_id) if session_id else None
        patient_count = patient_store._count() if patient_store else 0
        global_count = global_store._count()

        cache_key = ""
        if cache_service is not None and retrieval_cache_ttl_seconds > 0:
            query_hash = hashlib.sha256(" ".join(query.strip().split()).encode("utf-8")).hexdigest()
            sid = session_id or "none"
            cache_key = (
                f"ret:v2:session:{sid}:query:{query_hash}:"
                f"kp:{top_k_patient}:kg:{top_k_global}:"
                f"pc:{patient_count}:gc:{global_count}"
            )
            cached_payload = cache_service.get_json(cache_key)
            if isinstance(cached_payload, list):
                cached_hits: list[tuple[float, Dict[str, Any]]] = []
                for row in cached_payload:
                    if not isinstance(row, dict):
                        continue
                    if "distance" not in row or "metadata" not in row:
                        continue
                    try:
                        distance = float(row["distance"])
                    except Exception:
                        continue
                    metadata = row.get("metadata")
                    if isinstance(metadata, dict):
                        cached_hits.append((distance, metadata))
                if cached_hits:
                    logger.info(
                        "crag_retrieve_cache_hit",
                        session_id=session_id,
                        rewrite_count=state.get("rewrite_count", 0),
                        hits=len(cached_hits),
                    )
                    return {**state, "raw_hits": cached_hits}

        q_vec = embedder.embed_text(query)
        raw: list[tuple[float, Dict]] = []

        if patient_store and patient_count > 0:
            patient_hits = patient_store.search(
                q_vec,
                k=top_k_patient,
                metadata_filter={"session_id": session_id},
            )
            raw.extend(patient_hits)

        if global_count > 0:
            global_hits = global_store.search(q_vec, k=top_k_global)
            raw.extend(global_hits)

        raw = _apply_decay(raw)
        if cache_key and cache_service is not None and raw:
            cache_payload = [
                {"distance": float(distance), "metadata": metadata}
                for distance, metadata in raw
            ]
            cache_service.set_json(cache_key, cache_payload, ttl_seconds=retrieval_cache_ttl_seconds)

        return {**state, "raw_hits": raw}

    # ── Node 2: grade_documents ───────────────────────────────────────────────

    async def grade_documents(state: CRAGState) -> CRAGState:
        query = state["query"]
        raw_hits = state.get("raw_hits", [])

        if not raw_hits:
            logger.info("crag_grade_empty", query=query[:80])
            return {**state, "graded_hits": [], "rejected_hits": []}

        # Build grading prompt
        chunks_text = "\n\n".join(
            f"[Chunk {i}]\n{hit.get('text', '') if isinstance(hit, dict) else flatten_record(hit[1]).get('text', '')}"
            for i, hit in enumerate(raw_hits)
        )

        system_prompt = (
            "You are a medical document relevance grader.\n"
            "Given a clinical query and a set of retrieved document chunks, "
            "return a JSON object with a single key 'grades' — an array of booleans, "
            "one per chunk (true = relevant, false = not relevant).\n"
            "A chunk is relevant if it contains information that could help answer the clinical query.\n"
            "Be strict: vague or tangentially related content should be marked false.\n"
            "Output ONLY valid JSON. Example: {\"grades\": [true, false, true]}"
        )
        user_prompt = (
            f"Clinical Query: {query}\n\n"
            f"Chunks to grade:\n{chunks_text}\n\n"
            f"Return a JSON object with key 'grades' containing exactly {len(raw_hits)} booleans."
        )

        grades: List[bool] = []
        try:
            raw_json = await llm_service.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            parsed = json.loads(raw_json)
            grades = [bool(g) for g in parsed.get("grades", [])]
            # Pad or truncate to match hit count
            while len(grades) < len(raw_hits):
                grades.append(True)
            grades = grades[: len(raw_hits)]
        except Exception as exc:
            logger.warning("crag_grade_failed", error=str(exc), msg="Defaulting all chunks to relevant.")
            grades = [True] * len(raw_hits)

        graded, rejected = [], []
        for hit, is_relevant in zip(raw_hits, grades):
            if is_relevant:
                graded.append(hit)
            else:
                rejected.append(hit)

        logger.info(
            "crag_graded",
            relevant=len(graded),
            rejected=len(rejected),
            rewrite_count=state.get("rewrite_count", 0),
        )
        return {**state, "graded_hits": graded, "rejected_hits": rejected}

    # ── Node 3: rewrite_query ─────────────────────────────────────────────────

    async def rewrite_query(state: CRAGState) -> CRAGState:
        original_query = state["query"]
        rewrite_count = state.get("rewrite_count", 0) + 1

        system_prompt = (
            "You are a medical query reformulation expert.\n"
            "Given an original clinical query that failed to retrieve relevant documents, "
            "rewrite it to be more specific and clinically precise.\n"
            "Use medical terminology, expand abbreviations, and add clinically relevant context.\n"
            "Return ONLY a JSON object with key 'rewritten_query'. "
            "Example: {\"rewritten_query\": \"...\"}"
        )
        user_prompt = (
            f"Original query: {original_query}\n\n"
            "Rewrite this query to retrieve more relevant medical knowledge base chunks."
        )

        new_query = original_query
        try:
            raw_json = await llm_service.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
            )
            parsed = json.loads(raw_json)
            new_query = parsed.get("rewritten_query", original_query).strip() or original_query
        except Exception as exc:
            logger.warning("crag_rewrite_failed", error=str(exc), msg="Keeping original query.")

        logger.info(
            "crag_query_rewritten",
            original=original_query[:80],
            rewritten=new_query[:80],
            rewrite_count=rewrite_count,
        )
        return {**state, "query": new_query, "rewrite_count": rewrite_count}

    # ── Node 4: flag_low_confidence ───────────────────────────────────────────

    def flag_low_confidence(state: CRAGState) -> CRAGState:
        """
        Max retries exhausted with poor results.
        Passes whatever was retrieved with a low_confidence=True warning flag
        so DiagnosticAgent knows the evidence base is weak.
        """
        top_k_total = state.get("top_k_total", 8)
        raw_hits = state.get("raw_hits", [])

        logger.warning(
            "crag_low_confidence",
            query=state["query"][:80],
            retrieved=len(raw_hits),
        )

        # Fall back to best available hits even if grader rejected them
        results = _build_final_results(raw_hits, top_k_total)
        return {**state, "results": results, "low_confidence": True}

    # ── Node 5: finalize ──────────────────────────────────────────────────────

    def finalize(state: CRAGState) -> CRAGState:
        """Format graded hits into the final results list."""
        top_k_total = state.get("top_k_total", 8)
        graded_hits = state.get("graded_hits", [])
        results = _build_final_results(graded_hits, top_k_total)
        return {**state, "results": results, "low_confidence": False}

    def _build_final_results(
        hits: List[Any], top_k_total: int
    ) -> List[Dict[str, Any]]:
        seen = set()
        final = []
        for item in hits:
            # hits can be (distance, metadata) tuples or already-formatted dicts
            if isinstance(item, tuple):
                distance, metadata = item
                formatted = _format_hit(distance, metadata)
            else:
                formatted = item

            if formatted["chunk_id"] in seen:
                continue
            seen.add(formatted["chunk_id"])
            final.append(formatted)
            if len(final) >= top_k_total:
                break
        return final

    # ── Conditional edge logic ────────────────────────────────────────────────

    def _route_after_grading(state: CRAGState) -> str:
        graded_hits = state.get("graded_hits", [])
        rewrite_count = state.get("rewrite_count", 0)
        top_k_total = state.get("top_k_total", 8)

        # Enough relevant chunks — done
        if len(graded_hits) >= max(1, top_k_total // 2):
            return "finalize"

        # Still have retries left
        if rewrite_count < MAX_RETRIES:
            return "rewrite_query"

        # Exhausted retries
        return "flag_low_confidence"

    # ── Assemble graph ────────────────────────────────────────────────────────

    graph = StateGraph(CRAGState)

    graph.add_node("embed_and_retrieve", embed_and_retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("flag_low_confidence", flag_low_confidence)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("embed_and_retrieve")

    graph.add_edge("embed_and_retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {
            "finalize": "finalize",
            "rewrite_query": "rewrite_query",
            "flag_low_confidence": "flag_low_confidence",
        },
    )

    graph.add_edge("rewrite_query", "embed_and_retrieve")
    graph.add_edge("finalize", END)
    graph.add_edge("flag_low_confidence", END)

    return graph.compile()
