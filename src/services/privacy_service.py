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

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.utils.logger import get_logger

logger = get_logger(__name__)


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
        
        # Initialize Presidio Anonymizer (handles string replacement based on analyzer results)
        self.anonymizer = AnonymizerEngine()
        
        # Define the set of entities we actively want to scrub
        # Excludes medical entities (which Presidio doesn't natively target anyway,
        # but prevents accidental over-scrubbing)
        self.entities = [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "US_SSN",
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
        """
        clean_meta = {}
        for key, value in metadata.items():
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
