"""
src/services/terminology_service.py
-----------------------------------
A lightweight utility service to clean up medical text before it enters
the LLM or embedding space. Expands common medical acronyms and standardises
clinical laboratory units.
"""

import re
from typing import Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)

# A foundational set of common medical abbreviations that improve
# embedding semantics when expanded.
_MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    "HTN": "Hypertension",
    "MI": "Myocardial Infarction",
    "CHF": "Congestive Heart Failure",
    "DM": "Diabetes Mellitus",
    "T2DM": "Type 2 Diabetes Mellitus",
    "DOB": "Date of Birth",
    "BP": "Blood Pressure",
    "HR": "Heart Rate",
    "RR": "Respiratory Rate",
    "SOB": "Shortness of Breath",
    "N/V": "Nausea and Vomiting",
    "N/V/D": "Nausea, Vomiting, and Diarrhea",
    "WNL": "Within Normal Limits",
    "Dx": "Diagnosis",
    "Hx": "History",
    "Rx": "Prescription",
    "Tx": "Treatment",
    "Pt": "Patient",
    "BID": "Twice a day",
    "TID": "Three times a day",
    "QID": "Four times a day",
    "PRN": "As needed",
    "PO": "By mouth",
    "IV": "Intravenous",
    "IM": "Intramuscular",
    "STAT": "Immediately",
    "ER": "Emergency Room",
    "ED": "Emergency Department",
    "ICU": "Intensive Care Unit",
    "PACU": "Post-Anesthesia Care Unit",
    "COPD": "Chronic Obstructive Pulmonary Disease",
    "CABG": "Coronary Artery Bypass Graft",
    "ECG": "Electrocardiogram",
    "EKG": "Electrocardiogram",
    "EEG": "Electroencephalogram",
    "MRI": "Magnetic Resonance Imaging",
    "CT": "Computed Tomography",
    "CXR": "Chest X-Ray",
    "CBC": "Complete Blood Count",
    "BMP": "Basic Metabolic Panel",
    "CMP": "Comprehensive Metabolic Panel",
    "LFT": "Liver Function Test",
    "UA": "Urinalysis",
    "UTI": "Urinary Tract Infection",
    "URI": "Upper Respiratory Infection",
    "GERD": "Gastroesophageal Reflux Disease",
    "IBS": "Irritable Bowel Syndrome",
    "IBD": "Inflammatory Bowel Disease",
    "CVA": "Cerebrovascular Accident",
    "TIA": "Transient Ischemic Attack",
    "DVT": "Deep Vein Thrombosis",
    "PE": "Pulmonary Embolism",
    "A-fib": "Atrial Fibrillation",
    "AFib": "Atrial Fibrillation",
    "V-fib": "Ventricular Fibrillation",
    "VFib": "Ventricular Fibrillation",
    "CPR": "Cardiopulmonary Resuscitation",
    "DNR": "Do Not Resuscitate",
    "DNI": "Do Not Intubate",
    "GCS": "Glasgow Coma Scale",
    "LOC": "Loss of Consciousness",
    "AMA": "Against Medical Advice",
    "NPO": "Nothing By Mouth",
    "C. diff": "Clostridioides difficile",
    "MRSA": "Methicillin-resistant Staphylococcus aureus",
    "VRE": "Vancomycin-resistant Enterococcus",
    "ESBL": "Extended-Spectrum Beta-Lactamase",
}

# Standardizing units ensures that numbers like "14 mg /  dl" and "14mg/dL" map
# to the same token sequences in the embedding model.
_UNIT_MAPPINGS: Dict[str, str] = {
    r"(?<![a-zA-Z])\s*mg(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)dl(?:-1|−1)?\b": " mg/dL",
    r"(?<![a-zA-Z])\s*mmhg\b": " mmHg",
    r"(?<![a-zA-Z])\s*bpm\b": " bpm",
    r"(?<![a-zA-Z])\s*ml\b": " mL",
    r"(?<![a-zA-Z])\s*meq(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)l(?:-1|−1)?\b": " mEq/L",
    r"(?<![a-zA-Z])\s*u(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)l(?:-1|−1)?\b": " U/L",
    r"(?<![a-zA-Z])\s*g(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)dl(?:-1|−1)?\b": " g/dL",
    r"(?<![a-zA-Z])\s*(?:mcg|µg|μg|ug)(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)dl(?:-1|−1)?\b": " mcg/dL",
    r"(?<![a-zA-Z])\s*(?:mcg|µg|μg|ug)(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)l(?:-1|−1)?\b": " mcg/L",
    r"(?<![a-zA-Z])\s*ng(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)ml(?:-1|−1)?\b": " ng/mL",
    r"(?<![a-zA-Z])\s*pg(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)ml(?:-1|−1)?\b": " pg/mL",
    r"(?<![a-zA-Z])\s*mmol(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)l(?:-1|−1)?\b": " mmol/L",
    r"(?<![a-zA-Z])\s*umol(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)l(?:-1|−1)?\b": " umol/L",
    r"(?<![a-zA-Z])\s*fl\b": " fL",
    r"(?<![a-zA-Z])\s*pg\b": " pg",
    r"(?<![a-zA-Z])\s*k(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)(?:ul|µl|μl)(?:-1|−1)?\b": " K/uL",
    r"(?<![a-zA-Z])\s*10\^3(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)(?:ul|µl|μl)(?:-1|−1)?\b": " 10^3/uL",
    r"(?<![a-zA-Z])\s*10\^6(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)(?:ul|µl|μl)(?:-1|−1)?\b": " 10^6/uL",
    r"(?<![a-zA-Z])\s*l(?:\s*/\s*|\s*per\s*|\s*·\s*|\s*)min(?:-1|−1)?\b": " L/min",
    r"(?<![a-zA-Z])\s*kg\b": " kg",
    r"(?<![a-zA-Z])\s*lbs\b": " lbs",
    r"(?<![a-zA-Z])\s*cm\b": " cm",
}


class TerminologyService:
    """
    Service responsible for normalizing clinical text.
    Replaces common abbreviations and standardizes units.
    """

    def __init__(self):
        # Precompile abbreviation regexes (word boundary restricted)
        # Sort by length descending to match longest abbreviations first
        # (e.g. want to match 'N/V/D' before 'N/V')
        sorted_abbrs = sorted(_MEDICAL_ABBREVIATIONS.keys(), key=len, reverse=True)
        self._abbr_patterns = []
        for abbr in sorted_abbrs:
            # Escape literal strings like N/V
            escaped_abbr = re.escape(abbr)
            # Match word boundary before and after. We use (?i) to be case-insensitive,
            # but ONLY if the text is uppercase or mixed case in a way that suggests an acronym?
            # Actually, just case-sensitive is safer to avoid expanding "it" -> "Intrathecal" etc.
            # But the dict has some mixed case.
            # Let's enforce case-sensitive for now to prevent false positives (like 'or' -> Operating Room)
            # Some exceptions like 'C. diff' we can relax.
            # Let's just use EXACT dictionary case matching with word boundaries.
            pattern = re.compile(rf"\b{escaped_abbr}\b")
            self._abbr_patterns.append((pattern, _MEDICAL_ABBREVIATIONS[abbr]))

        # Precompile unit regexes (case insensitive for units)
        self._unit_patterns = []
        for pattern_str, replacement in _UNIT_MAPPINGS.items():
            pattern = re.compile(pattern_str, flags=re.IGNORECASE)
            self._unit_patterns.append((pattern, replacement))

    def expand_abbreviations(self, text: str) -> str:
        """
        Replaces known abbreviations with their full medical terms.
        """
        if not text:
            return ""

        expanded = text
        for pattern, full_term in self._abbr_patterns:
            expanded = pattern.sub(full_term, expanded)
        return expanded

    def standardize_units(self, text: str) -> str:
        """
        Cleans up spacing and capitalization around clinical units.
        """
        if not text:
            return ""

        standardized = text
        for pattern, clean_unit in self._unit_patterns:
            standardized = pattern.sub(clean_unit, standardized)
        return standardized

    def normalize(self, text: str) -> str:
        """
        Run the full normalization pipeline on raw text.
        """
        if not text:
            return ""
            
        t = text
        t = self.expand_abbreviations(t)
        t = self.standardize_units(t)
        
        # General cleanup: remove multiple spaces
        t = re.sub(r' {2,}', ' ', t)
        
        # Numeric standardisation: Remove thousands separator commas (1,200 -> 1200)
        t = re.sub(r'(?<=\d),(?=\d{3}\b)', '', t)
        
        return t.strip()
