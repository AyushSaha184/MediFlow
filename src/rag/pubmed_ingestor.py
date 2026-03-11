"""
src/rag/pubmed_ingestor.py
--------------------------
Fetches PubMed metadata/abstracts for KB ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from Bio import Entrez, Medline
except Exception:  # pragma: no cover - optional dependency in limited envs
    Entrez = None  # type: ignore[assignment]
    Medline = None  # type: ignore[assignment]


@dataclass
class PubMedIngestConfig:
    email: str
    api_key: str
    max_results: int = 50


def _require_biopython() -> None:
    if Entrez is None or Medline is None:
        raise RuntimeError("biopython is required for PubMed ingestion. Install `biopython`.")


def build_pubmed_query(topic: str, mesh_term: str | None = None, lookback_days: int | None = None) -> str:
    """
    Build a high-quality clinical evidence query.
    """
    type_filter = (
        '(guideline[Publication Type] OR "practice guideline"[Publication Type] '
        'OR "systematic review"[Publication Type] OR "meta-analysis"[Publication Type])'
    )
    language_filter = "english[Language]"
    human_filter = "humans[MeSH Terms]"
    core = mesh_term.strip() if mesh_term else topic.strip()
    if mesh_term and "[MeSH]" not in core:
        core = f"{core}[MeSH Terms]"
    date_filter = ""
    if lookback_days and lookback_days > 0:
        date_filter = f'AND ("{lookback_days} days"[PDat])'
    return f"({core}) AND {type_filter} AND {language_filter} AND {human_filter} {date_filter}".strip()


def fetch_pubmed_records(query: str, config: PubMedIngestConfig) -> List[Dict[str, Any]]:
    """
    Fetch MEDLINE records from PubMed for a query.
    """
    _require_biopython()
    Entrez.email = config.email
    Entrez.api_key = config.api_key or None

    logger.info("pubmed_search_started", query=query, max_results=config.max_results)

    with Entrez.esearch(db="pubmed", term=query, retmax=config.max_results, usehistory="n") as handle:
        search_record = Entrez.read(handle)
    pmids = search_record.get("IdList", [])
    if not pmids:
        return []

    with Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="medline",
        retmode="text",
    ) as handle:
        parsed = list(Medline.parse(handle))

    records: List[Dict[str, Any]] = []
    for item in parsed:
        pmid = str(item.get("PMID", "")).strip()
        title = str(item.get("TI", "")).strip()
        abstract = " ".join(item.get("AB", "").split())
        if not pmid or not title or not abstract:
            continue
        journal = str(item.get("JT", "")).strip()
        doi = ""
        for aid in item.get("AID", []) or []:
            text = str(aid)
            if "[doi]" in text.lower():
                doi = text.split(" ", 1)[0].strip()
                break
        mesh_terms = [str(m).strip() for m in (item.get("MH", []) or []) if str(m).strip()]
        records.append(
            {
                "document_id": f"pubmed::{pmid}",
                "source_file": f"pubmed_{pmid}",
                "title": title,
                "text": abstract,
                "issuer": journal or "PubMed",
                "source_type": "systematic_review",
                "source_tier": "evidence",
                "evidence_level": "A",
                "specialty": "general",
                "jurisdiction": "global",
                "published_at": str(item.get("DP", "")).strip(),
                "pmid": pmid,
                "doi": doi,
                "mesh_terms": mesh_terms,
            }
        )

    logger.info("pubmed_search_complete", query=query, records=len(records))
    return records
