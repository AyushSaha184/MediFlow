"""
src/services/hitl_review_service.py
-----------------------------------
Session-scoped clinician review persistence for human-in-the-loop gating.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.diagnostic_models import HITLReviewStatus, StructuredDiagnosis, UrgencyLevel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HITLReviewService:
    def __init__(self, patient_data_root: str = "data/User") -> None:
        self.patient_data_root = Path(patient_data_root)

    def _session_dir(self, session_id: str) -> Path:
        return self.patient_data_root / session_id

    def _review_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "hitl_review.json"

    def _read_review(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self._review_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("hitl_review_read_failed", session_id=session_id, error=str(exc))
            return None

    def _write_review(self, session_id: str, payload: Dict[str, Any]) -> None:
        path = self._review_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def evaluate_gate(
        self,
        diagnosis: StructuredDiagnosis,
        rag_context: List[Dict[str, Any]],
    ) -> List[str]:
        reasons: List[str] = []
        if diagnosis.confidence_score < 0.7:
            reasons.append(f"Low diagnosis confidence ({diagnosis.confidence_score:.2f} < 0.70).")
        if diagnosis.clinical_visual_congruence is False:
            reasons.append("Clinical-visual mismatch flagged by diagnostic agent.")
        if diagnosis.urgency_level in {UrgencyLevel.HIGH, UrgencyLevel.CRITICAL}:
            reasons.append(f"High-risk urgency level detected ({diagnosis.urgency_level.value}).")

        low_confidence_rag = any(
            bool((hit.get("metadata") or {}).get("low_confidence"))
            for hit in rag_context
            if isinstance(hit, dict)
        )
        if low_confidence_rag:
            reasons.append("RAG retrieval marked as low confidence.")

        return reasons

    def create_pending_review(
        self,
        session_id: str,
        diagnosis: StructuredDiagnosis,
        rag_context: List[Dict[str, Any]],
        reasons: List[str],
    ) -> HITLReviewStatus:
        existing = self._read_review(session_id)
        review_id = str(existing.get("review_id")) if existing and existing.get("review_id") else str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "status": "pending_clinician_review",
            "review_id": review_id,
            "reasons": reasons,
            "reviewer_id": None,
            "reviewer_notes": None,
            "diagnosis": diagnosis.model_dump(),
            "rag_context": rag_context,
        }
        self._write_review(session_id, payload)
        logger.info("hitl_review_created", session_id=session_id, review_id=review_id, reasons=len(reasons))
        return HITLReviewStatus.model_validate(payload)

    def get_status(self, session_id: str) -> HITLReviewStatus:
        payload = self._read_review(session_id)
        if not payload:
            return HITLReviewStatus(session_id=session_id, status="none")
        return HITLReviewStatus(
            session_id=session_id,
            status=payload.get("status", "none"),
            review_id=payload.get("review_id"),
            reasons=payload.get("reasons", []),
            reviewer_id=payload.get("reviewer_id"),
            reviewer_notes=payload.get("reviewer_notes"),
        )

    def get_payload(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._read_review(session_id)

    def get_approved_payload(self, session_id: str) -> Optional[Dict[str, Any]]:
        payload = self._read_review(session_id)
        if not payload:
            return None
        if payload.get("status") != "approved":
            return None
        return payload

    def approve(self, session_id: str, reviewer_id: str, notes: Optional[str]) -> HITLReviewStatus:
        payload = self._read_review(session_id)
        if not payload:
            raise ValueError("No HITL review found for this session.")
        if payload.get("status") != "pending_clinician_review":
            raise ValueError(f"Review is not pending (current status: {payload.get('status')}).")
        payload["status"] = "approved"
        payload["reviewer_id"] = reviewer_id
        payload["reviewer_notes"] = notes
        self._write_review(session_id, payload)
        logger.info("hitl_review_approved", session_id=session_id, review_id=payload.get("review_id"), reviewer_id=reviewer_id)
        return self.get_status(session_id)

    def reject(self, session_id: str, reviewer_id: str, notes: Optional[str]) -> HITLReviewStatus:
        payload = self._read_review(session_id)
        if not payload:
            raise ValueError("No HITL review found for this session.")
        if payload.get("status") != "pending_clinician_review":
            raise ValueError(f"Review is not pending (current status: {payload.get('status')}).")
        payload["status"] = "rejected"
        payload["reviewer_id"] = reviewer_id
        payload["reviewer_notes"] = notes
        self._write_review(session_id, payload)
        logger.info("hitl_review_rejected", session_id=session_id, review_id=payload.get("review_id"), reviewer_id=reviewer_id)
        return self.get_status(session_id)
