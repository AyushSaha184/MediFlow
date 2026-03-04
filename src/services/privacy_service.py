"""
src/services/privacy_service.py
-------------------------------
A service wrapping Microsoft Presidio for detecting and anonymizing
Personally Identifiable Information (PII) and Protected Health
Information (PHI).

Loads the spaCy models and Presidio Engines globally to avoid
excessive overhead across multiple calls.
"""

from typing import Any, Dict, List, Optional
import os

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.utils.logger import get_logger

logger = get_logger(__name__)

SAFE_DICOM_KEYS = {
    "StudyDate", "StudyTime", "Modality", "BodyPartExamined", "PixelSpacing",
    "Rows", "Columns", "WindowCenter", "WindowWidth", "SeriesDescription",
}


class PrivacyService:
    """
    Stateless utility class wrapping Presidio Analyzer and Anonymizer.
    Initialization builds the NLP engine, making it a relatively heavy object.
    Should be instantiated once and reused (Singleton pattern).
    """

    def __init__(self) -> None:
        logger.info("privacy_service_init_start", engine="presidio")
        
        # Suppress spaCy huggingface warnings if present
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Initialize Presidio Analyzer (uses spaCy under the hood for NER)
        self.analyzer = AnalyzerEngine()
        
        # ── 1. Custom Recognizer: Medical Identifiers (MRN, Patient ID) ──
        # Matches patterns like "Patient ID: 4839201", "MRN: 12345", "Case No: 999"
        medical_id_pattern = Pattern(
            name="medical_id_pattern",
            regex=r"(?i)\b(?:patient[\s_-]?id|mrn|report[\s_-]?(?:no|number)|case[\s_-]?(?:no|number)|accession[\s_-]?(?:no|number))\s*[:#-]?\s*[A-Z0-9-]+\b",
            score=0.85
        )
        medical_id_recognizer = PatternRecognizer(
            supported_entity="MEDICAL_ID",
            patterns=[medical_id_pattern]
        )
        self.analyzer.registry.add_recognizer(medical_id_recognizer)

        # ── 2. Custom Recognizer: Misspelled/Prefixed Names ──
        # Matches patterns like "Pt: J Doe", "Patient Name: Smith, John", "Dr. Patel", "Mrs. Brown"
        # Presidio's built-in PERSON recognizer is good, but explicit prefixes boost confidence.
        name_prefix_pattern = Pattern(
            name="name_prefix_pattern",
            regex=r"(?i)\b(?:pt|patient|patient name|patient:\s*name|dr|mr|mrs|ms)[.:]?\s*[:#-]?\s*[A-Z][a-z]*(?:\s*(?:[A-Z][a-z]*|\.))*\b",
            score=0.85
        )
        name_prefix_recognizer = PatternRecognizer(
            supported_entity="PERSON",  # Map to existing PERSON entity
            patterns=[name_prefix_pattern]
        )
        self.analyzer.registry.add_recognizer(name_prefix_recognizer)
        
        # Initialize Presidio Anonymizer (handles string replacement based on analyzer results)
        self.anonymizer = AnonymizerEngine()
        
        # Define the set of entities we actively want to scrub
        # Includes our custom MEDICAL_ID
        self.entities = [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "US_SSN",
            "MEDICAL_ID",
            "CREDIT_CARD",
            "CRYPTO",
            "IBAN_CODE",
            "IP_ADDRESS",
            "US_DRIVER_LICENSE",
            "US_PASSPORT",
            "US_BANK_NUMBER",
        ]
        
        logger.info("privacy_service_init_complete")

    def anonymize_text(self, text: str) -> str:
        """
        Scan a string for PII and replace findings with <ENTITY_TYPE>.
        Returns the original string if empty or no findings.
        """
        if not text or not text.strip():
            return text

        try:
            # 1. Analyze
            results = self.analyzer.analyze(
                text=text,
                entities=self.entities,
                language='en'
            )
            
            if not results:
             return text
                
            # 2. Anonymize using <ENTITY_TYPE> masks
            # For each detected entity, simply replace it with its type name wrapped in brackets
            operators = {
                entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
                for entity in self.entities
            }
            
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators
            )
            
            return anonymized_result.text
            
        except Exception as exc:
            logger.error("privacy_text_anonymize_error", error=str(exc))
            # In clinical environments, if privacy fails, failing open (returning raw text) 
            # is extremely dangerous. We must raise an error so the pipeline halts for this doc.
            raise RuntimeError(f"Failed to anonymize text: {exc}") from exc

    def anonymize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively traverse a dictionary and anonymize all string values.
        Used to clean the `metadata` dictionary returned by DICOM/PDF extractors.
        Safe DICOM keys (like Modality) are strictly skipped to prevent data corruption.
        """
        clean_meta = {}
        for key, value in metadata.items():
            # Skip safe keys (DICOM structural fields)
            if key in SAFE_DICOM_KEYS:
                clean_meta[key] = value
                continue

            if isinstance(value, str):
                clean_meta[key] = self.anonymize_text(value)
            elif isinstance(value, dict):
                clean_meta[key] = self.anonymize_metadata(value)
            elif isinstance(value, list):
                # Anonymize each list item if it's a string or dict
                clean_list = []
                for item in value:
                    if isinstance(item, str):
                        clean_list.append(self.anonymize_text(item))
                    elif isinstance(item, dict):
                        clean_list.append(self.anonymize_metadata(item))
                    else:
                        clean_list.append(item)
                clean_meta[key] = clean_list
            else:
                # Keep ints, floats, booleans, None exactly as-is
                clean_meta[key] = value
        return clean_meta

    def anonymize_tabular_data(self, tabular_data: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """
        Anonymizes a list of dictionaries (tabular rows).
        Used primarily for XLSX extractions.
        """
        if tabular_data is None:
            return None
            
        clean_rows = []
        for row in tabular_data:
            clean_rows.append(self.anonymize_metadata(row))
            
        return clean_rows
