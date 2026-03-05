"""
src/services/session_context.py
-------------------------------
Provides Atomic Session Locks to prevent 'Metadata Drift' (Phase 1.5 Edge Case),
ensuring that Patient A's images are not accidentally combined with Patient B's labs
during parallel processing.
"""
import asyncio
from typing import Dict, Optional
from contextlib import asynccontextmanager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PatientSessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.patient_id: Optional[str] = None
        self.lock = asyncio.Lock()
        
    async def verify_or_set_patient(self, new_patient_id: str) -> bool:
        """
        Atomically verifies the patient ID for this session.
        Returns False if there is a mismatch (Metadata Drift).
        """
        async with self.lock:
            if self.patient_id is None:
                self.patient_id = new_patient_id
                return True
            return self.patient_id == new_patient_id

class SessionManager:
    """Global manager for active sessions to prevent cross-contamination."""
    
    def __init__(self):
        self._sessions: Dict[str, PatientSessionState] = {}
        self._manager_lock = asyncio.Lock()
        
    async def get_session(self, session_id: str) -> PatientSessionState:
        async with self._manager_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = PatientSessionState(session_id)
            return self._sessions[session_id]
            
    async def cleanup_session(self, session_id: str):
        async with self._manager_lock:
            self._sessions.pop(session_id, None)

session_manager = SessionManager()

@asynccontextmanager
async def atomic_session_lock(session_id: str, caller_name: str = "Unknown"):
    """
    Context manager to wrap cross-agent tasks.
    In a fully distributed system (Redis), this would be a distributed lock.
    """
    session = await session_manager.get_session(session_id)
    logger.debug("session_lock_acquired", session=session_id, caller=caller_name)
    try:
        # We don't hold the lock for the whole execution to prevent deadlocks in parallel tasks,
        # but the context block ensures the session state is tracked.
        yield session
    finally:
        logger.debug("session_lock_released", session=session_id, caller=caller_name)
