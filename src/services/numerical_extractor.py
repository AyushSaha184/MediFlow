"""
src/services/numerical_extractor.py
-----------------------------------
Pre-processor that extracts numerical lab values from text or tabular data,
normalizes units (e.g., mg/dL to SI where appropriate), checks against 
normal reference ranges, and formats them into a visual Markdown table.

This creates a "hard attention" mechanism for the LLM to avoid hallucinating
numbers and less/greater-than signs.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Reference ranges and normalization factors
# Format: { "biomarker_regex": { "unit": "standard_unit", "min": val, "max": val, "aliases": [...] } }
REFERENCE_RANGES = {
    "glucose": {"unit": "mg/dL", "min": 70, "max": 99, "aliases": ["glc", "blood sugar", "fasting glucose"]},
    "creatinine": {"unit": "mg/dL", "min": 0.5, "max": 1.2, "aliases": ["creat", "cr"]},
    "hemoglobin": {"unit": "g/dL", "min": 12.0, "max": 17.5, "aliases": ["hgb", "hb"]},
    "wbc": {"unit": "K/uL", "min": 4.5, "max": 11.0, "aliases": ["white blood cell", "leukocyte count"]},
    "potassium": {"unit": "mmol/L", "min": 3.5, "max": 5.1, "aliases": ["k", "k+"]},
    "sodium": {"unit": "mmol/L", "min": 135, "max": 145, "aliases": ["na", "na+"]}
}

class NumericalGuardrailsExtractor:
    """
    Extracts and formats lab values from raw text or structured dictionaries,
    applying '!! [Value] (High/Low) !!' flags to draw LLM attention.
    """
    def __init__(self):
        # Build optimized regex for finding any aliases followed by a number
        self._compiled_regexes = {}
        for key, ref in REFERENCE_RANGES.items():
            words = [key] + ref.get("aliases", [])
            # e.g., allow up to 15 non-digit chars between the biomarker and the number
            pattern = r"(?i)\b(" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in words) + r")\b" \
                      r"\D{0,15}?(\d+\.\d+|\d+)\s*" \
                      r"(mg/dl|g/dl|mmol/l|k/ul|u/l)?"
            self._compiled_regexes[key] = re.compile(pattern)

    def extract_from_text(self, text: str, timestamp: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Scans free text for common biomarkers and their values.
        Returns a list of structured lab dictionaries.
        """
        extracted = []
        if not text:
            return extracted
            
        for key, regex in self._compiled_regexes.items():
            for match in regex.finditer(text):
                matched_name = match.group(1).strip()
                val_str = match.group(2)
                unit_str = match.group(3) or ""
                
                try:
                    val = float(val_str)
                    
                    # Normalize basic units (e.g. Glucose mmol/L -> mg/dL)
                    if key == "glucose" and unit_str.lower() == "mmol/l":
                        val = val * 18.0182  # to mg/dL
                    
                    extracted.append({
                        "biomarker": key.capitalize(),
                        "value": round(val, 2),
                        "unit": REFERENCE_RANGES[key]["unit"],
                        "original_match": match.group(0),
                        "min": REFERENCE_RANGES[key]["min"],
                        "max": REFERENCE_RANGES[key]["max"],
                        "timestamp": timestamp or "Unknown"
                    })
                except Exception as e:
                    logger.debug("lab_extraction_error", error=str(e), match=match.group(0))
        
        # Removed deduplication to maintain multiple values per biomarker (Temporal Trends)
        return extracted

    def format_markdown_table(self, labs: List[Dict[str, str]]) -> str:
        """
        Converts extracted lab dictionaries into a Markdown table with visual Flags.
        """
        if not labs:
            return "No numerical lab data extracted."

        lines = [
            "| Date/Time | Biomarker | Value | Unit | Reference Range | Flag |",
            "|-----------|-----------|-------|------|-----------------|------|"
        ]

        # Sort labs by biomarker, then by timestamp
        labs_sorted = sorted(labs, key=lambda x: (x["biomarker"], x.get("timestamp", "")))

        for lab in labs_sorted:
            val = lab["value"]
            min_val = lab["min"]
            max_val = lab["max"]
            ts = lab.get("timestamp", "Unknown")
            
            flag = "Normal"
            val_str = str(val)
            
            if val < min_val:
                flag = "Low"
                val_str = f"!! {val} (Low) !!"
            elif val > max_val:
                flag = "High"
                val_str = f"!! {val} (High) !!"

            line = f"| {ts} | {lab['biomarker']} | {val_str} | {lab['unit']} | {min_val} - {max_val} | {flag} |"
            lines.append(line)
            
        return "\n".join(lines)

    def process_document(self, text: str, timestamp: Optional[str] = None) -> str:
        """
        Takes raw text, extracts numbers with optional timestamp, and outputs markdown table.
        """
        extracted = self.extract_from_text(text, timestamp)
        return self.format_markdown_table(extracted)

    def process_historical_context(self, current_text: str, current_ts: Optional[str], rag_chunks: List[Dict[str, Any]]) -> str:
        """
        Extracts lab trends from current document and historical RAG chunks combined.
        """
        all_labs = []
        all_labs.extend(self.extract_from_text(current_text, current_ts))
        
        for chunk in rag_chunks:
            chunk_text = chunk.get("text", "")
            chunk_ts = chunk.get("metadata", {}).get("document_timestamp", "Historical")
            all_labs.extend(self.extract_from_text(chunk_text, chunk_ts))
            
        # Optional: Further filter to ensure we only show a reasonable number, but 
        # listing all historical points allows the LLM to see the explicit trend line.
        return self.format_markdown_table(all_labs)
