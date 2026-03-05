"""
src/agents/vision/router.py
---------------------------
Routes incoming images by modality (DICOM, NIfTI, Standard).
Applies appropriate preprocessing (Quality Gates, Volumetric Slice Selection)
before yielding the payload to the Vision-Language Model.
"""

import numpy as np
from PIL import Image
from typing import Dict, Any
from src.utils.logger import get_logger
from src.agents.vision.quality_gate import validate_image_quality
from src.agents.vision.dicom_processor import DicomProcessor

logger = get_logger(__name__)

class MedicalImageRouter:
    """Routes and pre-processes medical images for the VLM."""
    
    def __init__(self):
        self.dicom_processor = DicomProcessor()
    
    async def route_and_process(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """
        Determines modality by extension and routes to appropriate handler.
        """
        file_ext = file_path.split('.')[-1].lower()
        
        try:
            if file_ext == "dcm":
                return await self._handle_dicom(file_path)
            elif file_ext in ["jpg", "png", "jpeg"]:
                return await self._handle_standard_image(file_path)
            elif file_ext in ["nii", "nii.gz"]:
                return await self._handle_nifti(file_path)
            else:
                raise ValueError(f"Unsupported modality extension: {file_ext}")
        except Exception as e:
            logger.error("image_routing_failed", error=str(e), file_path=file_path)
            # In a real pipeline, we'd log the session_id to state
            return {"error": "PROCESSING_FAILED", "detail": str(e)}

    async def _handle_dicom(self, path: str) -> Dict[str, Any]:
        """
        Uses dicom_processor to extract metadata and orientation-safe pixels.
        """
        metadata, pixel_array = self.dicom_processor.extract_and_prepare(path)
        
        # Volumetric Slice Selection would go here for 3D stacks
        # Returning a simplified payload for the VLM
        return {
            "type": "DICOM",
            "metadata": metadata,
            "pixels": pixel_array
        }

    async def _handle_standard_image(self, path: str) -> Dict[str, Any]:
        """
        Handles JPG/PNG with a strict Quality / Blur gate.
        """
        img = Image.open(path)
        
        # Quality Gate Check (Phase 1.5 Edge Case)
        is_valid, reason = validate_image_quality(img)
        if not is_valid:
            # Rejects with standard error code mapped in Phase 1
            return {"error": "LOW_IMAGE_QUALITY", "detail": reason}
            
        return {"type": "STANDARD", "metadata": {"modality": "Image"}, "image": img}

    async def _handle_nifti(self, path: str) -> Dict[str, Any]:
        """
        Stub for Neuroimaging. Flattens or slices the NIfTI array.
        """
        import nibabel as nib
        
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        
        # Normalize the whole volume
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
            
        # Return full volume for the agent to sample
        return {
            "type": "NIFTI", 
            "metadata": {"modality": "MRI/CT"}, 
            "pixels": data
        }
