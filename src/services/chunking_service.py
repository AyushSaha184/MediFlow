"""
src/services/chunking_service.py
--------------------------------
A smart text chunker specialized for clinical documents.
Splits text based on common clinical headers (e.g. "HISTORY OF PRESENT ILLNESS:")
and falls back to overlapping character counts for very long sections.
"""

import re
from typing import Any, Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Common clinical headers that signify a strong context shift.
_CLINICAL_HEADERS = [
    r"CHIEF COMPLAINT:",
    r"HPI:",
    r"HISTORY OF PRESENT ILLNESS:",
    r"HISTORY OF ILLNESS:",
    r"CLINICAL HISTORY:",
    r"PATIENT HISTORY:",
    r"REVIEW OF SYSTEMS:",
    r"PHYSICAL EXAMINATION:",
    r"LABORATORY DATA:",
    r"IMAGING:",
    r"ASSESSMENT AND PLAN:",
    r"ASSESSMENT:",
    r"PLAN:",
    r"DISCHARGE DIAGNOSES:",
    r"HOSPITAL COURSE:",
    r"PAST MEDICAL HISTORY:",
    r"PAST SURGICAL HISTORY:",
    r"SOCIAL HISTORY:",
    r"FAMILY HISTORY:",
    r"ALLERGIES:",
    r"MEDICATIONS:",
    r"DISPOSITION:",
    r"FOLLOW UP:",
    # Variants without colons or slightly different wording can be added here
]

# Compile a mega-regex that matches ANY of the headers, anchoring them loosely to the start of a line
# (?im) = case-insensitive, multiline mode (so ^ matches start of line, not just start of string)
_HEADER_REGEX = re.compile(rf"^(?:{'|'.join(_CLINICAL_HEADERS)})", flags=re.IGNORECASE | re.MULTILINE)


class ChunkingService:
    """
    Service responsible for splitting clinical text into overlapping chunks
    suitable for RAG/Embedding pipelines.
    """

    def __init__(self, target_chunk_size: int = 1500, overlap: int = 200):
        """
        Initialize the chunker.
        
        Args:
            target_chunk_size: Maximum desired character count per chunk.
            overlap: Number of characters to overlap when splitting giant sections.
        """
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def _split_by_headers(self, text: str) -> List[Dict[str, str]]:
        """
        Splits the raw text at major clinical headers.
        Returns a list of dicts: {"header": "...", "content": "..."}
        """
        if not text:
            return []

        # Find all header matches
        matches = list(_HEADER_REGEX.finditer(text))
        
        sections = []
        if not matches:
            # No clinical headers found, treat the whole thing as one section
            sections.append({
                "header": "GENERAL",
                "content": text.strip()
            })
            return sections

        # If there's text *before* the first header, capture it
        first_match_start = matches[0].start()
        if first_match_start > 0:
            pre_text = text[:first_match_start].strip()
            if pre_text:
                sections.append({
                    "header": "GENERAL",
                    "content": pre_text
                })

        # Capture text between each header
        for i, match in enumerate(matches):
            header_text = match.group(0).strip()
            start_idx = match.end()
            
            # The section ends where the next header begins, or the end of the text
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            content = text[start_idx:end_idx].strip()
            
            if content:
                sections.append({
                    "header": header_text.upper().replace(":", ""), # Clean standard uppercase header
                    "content": content
                })

        return sections

    def _apply_overlap_chunking(self, content: str, header: str) -> List[Dict[str, Any]]:
        """
        Takes a single massive section and breaks it down into standard overlapping chunks.
        """
        # If it fits within the target size, no overlap chunking needed
        if len(content) <= self.target_chunk_size:
            return [{
                "text": f"[{header}]\n{content}",
                "section": header,
                "metadata": {"chunk_type": "full_section"}
            }]

        chunks = []
        start = 0
        text_len = len(content)
        chunk_index = 0

        while start < text_len:
            end = start + self.target_chunk_size
            
            # If this isn't the final chunk, try to find a clean break (paragraph, newline, or period)
            if end < text_len:
                # Look backwards within the last 200 chars to find a natural break
                window_size = 200
                lookback_window = content[max(start, end - window_size):end]
                
                last_double_newline = lookback_window.rfind("\n\n")
                last_newline = lookback_window.rfind("\n")
                last_period = lookback_window.rfind(". ")
                
                # Priority: Paragraph -> Line -> Sentence
                if last_double_newline != -1:
                    end = max(start, end - window_size) + last_double_newline + 2
                elif last_newline != -1:
                    end = max(start, end - window_size) + last_newline + 1
                elif last_period != -1:
                    end = max(start, end - window_size) + last_period + 2

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append({
                    # Prepend the clinical header so the embedding model knows the context of this random slice
                    "text": f"[{header}]\n{chunk_content}",
                    "section": header,
                    "metadata": {
                        "chunk_type": "split_section",
                        "chunk_index": chunk_index
                    }
                })
                
            if end >= text_len:
                break
            
            # Advance start by target size MINUS overlap.
            # However, if we found a natural break (end < start + target_chunk_size),
            # we need to ensure we don't accidentally advance `start` backwards!
            next_start = end - self.overlap
            start = max(start + 1, next_start)
            chunk_index += 1

            # Prevent infinite loops if overlap > target size (edge case guard)
            if start <= end - self.target_chunk_size + self.overlap and self.overlap >= self.target_chunk_size:
                 start = end # Hard fallback

        return chunks

    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """
        The main public method. Takes a normalized document string and returns a list of chunk dicts.
        """
        if not text:
            return []

        # 1. Split logically by clinical headers
        sections = self._split_by_headers(text)
        
        # 2. Split physically if any section is too large, and deduplicate
        final_chunks = []
        seen_texts = set()
        
        for sec in sections:
            split_chunks = self._apply_overlap_chunking(sec["content"], sec["header"])
            for sc in split_chunks:
                text_val = sc["text"]
                
                # In testing, overlapping chunks of repeating chars caused hash strikes.
                # Only consider it a duplicate if the EXACT string exists AND it wasn't
                # generated as part of a sequential overlap split. 
                # A hash of (header, text, chunk_index) allows identical text ONLY if they 
                # are specifically sequential overlaps (which indicates repeating raw text).
                chunk_index = sc.get("metadata", {}).get("chunk_index", -1)
                dedupe_key = f"{sec['header']}_{chunk_index}_{text_val}"
                
                # 30 chars is too restrictive for short headers like CHIEF COMPLAINT: Chest pain.
                # Just drop empty strings and exact duplicate chunks (like repeated instructions)
                if len(text_val.strip()) > 0 and dedupe_key not in seen_texts:
                    seen_texts.add(dedupe_key)
                    final_chunks.append(sc)
            
        logger.debug(
            "document_chunked",
            sections_found=len(sections),
            chunks_generated=len(final_chunks)
        )
            
        return final_chunks
