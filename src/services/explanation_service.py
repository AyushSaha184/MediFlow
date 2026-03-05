"""
src/services/explanation_service.py
-----------------------------------
Phase 6 Service: Provides logic to transform complex medical outputs into 
patient-friendly forms and visual helpers.
- Narrative Sparklines (few-shot text rendering of temporal deltas).
- Reverse Terminology Mapping (expanding medical acronyms dynamically).
"""

from typing import List, Dict, Any, Optional
import re
from src.services.terminology_service import TerminologyService
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ExplanationService:
    def __init__(self, terminology_service: Optional[TerminologyService] = None):
        self.terminology = terminology_service or TerminologyService()
        # Build a reverse lookup from the terminology service
        self.reverse_map = self._build_reverse_map()

    def _build_reverse_map(self) -> Dict[str, str]:
        # Basic mapping: acronym -> full term & simplified explanation
        # In production this would hit the Snomed/LOINC DB
        return {
            "mi": "Myocardial Infarction (Heart Attack)",
            "hrt": "Hormone Replacement Therapy",
            "htn": "Hypertension (High Blood Pressure)",
            "cad": "Coronary Artery Disease",
            "copd": "Chronic Obstructive Pulmonary Disease",
            "aki": "Acute Kidney Injury",
            # Add more as needed
        }

    def reverse_terminology_lookup(self, text: str) -> str:
        """
        Scans patient-facing text for medical jargon/acronyms and injects 
        parenthetical explanations.
        """
        # A simple regex approach for scaffolding.
        # Find isolated acronyms and inject their explanation.
        words = text.split()
        explained_words = []
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z0-9]', '', word.lower())
            if clean_word in self.reverse_map and "(" not in word:
                # E.g., "MI" -> "MI (Myocardial Infarction (Heart Attack))"
                explanation = self.reverse_map[clean_word]
                # Try to maintain original casing
                explained_words.append(f"{word} ({explanation})")
            else:
                explained_words.append(word)
                
        return " ".join(explained_words)

    def generate_narrative_sparklines(self, lab_historical_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Converts temporal lab lists into narrative "Sparklines".
        Format: "Hemoglobin (↓): 12.1 -> 10.5 -> 9.2"
        """
        sparklines = []
        
        for lab_name, measurements in lab_historical_data.items():
            if len(measurements) < 2:
                continue # Needs at least 2 points for a trend
                
            # Sort by date
            measurements.sort(key=lambda x: x.get("timestamp", ""))
            
            first_val = float(measurements[0].get("value", 0))
            last_val = float(measurements[-1].get("value", 0))
            
            trend_icon = "→"
            if last_val > first_val * 1.05:
                trend_icon = "↑"
            elif last_val < first_val * 0.95:
                trend_icon = "↓"
                
            vals_str = " → ".join([str(m.get("value", "")) for m in measurements])
            unit = measurements[0].get("unit", "")
            
            sparklines.append(f"- **{lab_name.title()} ({trend_icon})**: {vals_str} {unit}")
            
        if not sparklines:
            return "No significant historical trends detected."
            
        return "\n".join(sparklines)

    def inject_hedge_words(self, text: str) -> str:
        """
        Calibrates patient-facing language to lower diagnostic rigidity.
        Replaces definitive words with suggestive words.
        """
        replacements = {
            r"\bproves\b": "suggests",
            r"\bconfirms\b": "is consistent with",
            r"\bis definitively\b": "appears to be",
            r"\bwe know\b": "the evidence indicates",
            r"\bmeans that\b": "may indicate that"
        }
        
        adjusted_text = text
        for pattern, replacement in replacements.items():
            adjusted_text = re.sub(pattern, replacement, adjusted_text, flags=re.IGNORECASE)
            
        return adjusted_text
