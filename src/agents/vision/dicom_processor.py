"""
src/agents/vision/dicom_processor.py
------------------------------------
Handles metadata extraction, windowing, and orientation validation for DICOM files.
"""

import pydicom
import numpy as np
from typing import Dict, Any, Tuple, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CriticalOrientationMismatch(Exception):
    """Raised when the VLM's claimed side (L/R) contradicts the DICOM header."""
    pass

class DicomProcessor:
    def __init__(self):
        # Add any NLP services or privacy services if needed to strip headers
        pass

    def extract_and_prepare(self, file_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Reads a DICOM file, extracts critical Phase 5 metadata, and returns
        the pixel volume (or slice).
        """
        try:
            ds = pydicom.dcmread(file_path)
            
            # 1. Extract Phase 5 Metadata (Orientation, Modality)
            metadata = {
                "modality": ds.Modality if hasattr(ds, 'Modality') else "Unknown",
                "body_part": getattr(ds, "BodyPartExamined", "Unknown"),
                "timestamp": ds.ContentDate if hasattr(ds, 'ContentDate') else None,
                "orientation_patient": getattr(ds, "ImageOrientationPatient", None)
            }
            
            # 2. Extract Pixel Array
            pixel_data = ds.pixel_array
            
            # 3. Handle Volumetric Normalization (CT/MRI)
            pixel_data = pixel_data.astype(np.float32)
            if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
                intercept = float(ds.RescaleIntercept)
                slope = float(ds.RescaleSlope)
                pixel_data = pixel_data * slope + intercept
                
            # Basic normalization (min-max) across the whole volume/slice
            if pixel_data.max() > pixel_data.min():
                pixel_data = (pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min())
            
            return metadata, pixel_data
            
        except Exception as e:
            logger.error("dicom_processing_failed", file_path=file_path, error=str(e))
            raise RuntimeError(f"Failed to process DICOM: {e}") from e

    def validate_side_orientation(self, vlm_claim: str, dicom_metadata: Dict[str, Any]) -> bool:
        """
        Cross-references the VLM's output string against DICOM orientation tags.
        (Simplified implementation for scaffolding)
        """
        orientation = dicom_metadata.get("orientation_patient")
        if not orientation:
            return True # Cannot validate
            
        # If VLM claims 'left' but orientation array heavily implies right-sided only scan
        # Real logic involves math on the direction vectors (x, y, z cosines), 
        # but for safety, we flag discrepancies.
        vlm_lower = vlm_claim.lower()
        if "left lung" in vlm_lower and "right" in str(dicom_metadata.get("body_part")).lower():
            raise CriticalOrientationMismatch(
                "VLM reported finding on LEFT side, but DICOM metadata restricts scan to RIGHT side."
            )
            
        return True

    def get_representative_slices(self, pixel_data: np.ndarray, num_slices: int = 1) -> List[np.ndarray]:
        """
        Samples the 3D volume to get representative slices.
        If pixel_data is 2D, returns it as a list of 1.
        """
        if pixel_data.ndim == 2:
            return [pixel_data]
            
        if pixel_data.ndim == 3:
            total_slices = pixel_data.shape[0]
            if num_slices >= total_slices:
                indices = range(total_slices)
            else:
                # Evenly sample across the volume
                indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
            
            return [pixel_data[idx] for idx in indices]
            
        return []
