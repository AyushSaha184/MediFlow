"""
src/agents/vision_perception_agent.py
-------------------------------------
Phase 1.5 Orchestrator: Receives routed image arrays, builds longitudinal context
(ComparisonBuffer), and calls the VLM (NVIDIA Gemma-3-27b-it API).
Enforces the Side-Swap safety check on the output.
"""

import json
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
import httpx
from PIL import Image
import numpy as np

from src.core.base_agent import BaseAgent
from src.core.config import settings
from src.agents.vision.router import MedicalImageRouter
from src.agents.vision.dicom_processor import CriticalOrientationMismatch
from src.models.medical_document import MedicalDocumentSchema

class VisionPerceptionAgent(BaseAgent):
    def __init__(self):
        super().__init__("VisionPerceptionAgent")
        self.router = MedicalImageRouter()
        self.nvidia_api_key = settings.nvidia_api_key
        self.model_name = settings.vision_model_name
        self.invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.CONFIDENCE_THRESHOLD = 0.6

    def _encode_image(self, img_source: Any) -> str:
        """Helper to convert PIL Image or Numpy array into base64 string for the API."""
        if isinstance(img_source, Image.Image):
            pil_img = img_source
        elif isinstance(img_source, np.ndarray):
            # If 2D (grayscale) or 3D, ensure uint8
            if img_source.dtype != np.uint8:
                # normalize if needed, assumed done in router
                pass
            pil_img = Image.fromarray(img_source)
        else:
            raise ValueError("Unsupported image source type for encoding.")
            
        # Compress and encode
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def _call_vlm(self, base64_images: List[str], prompt: str) -> str:
        """Hits the NVIDIA endpoint supporting one or more images."""
        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Accept": "application/json"
        }
        
        content_blocks = [{"type": "text", "text": prompt}]
        for b64 in base64_images:
            content_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content_blocks
                }
            ],
            "max_tokens": 768,
            "temperature": 0.20,
            "top_p": 0.70,
            "stream": False
        }
        
        # Synchronous httpx call wrapped in async (for scaffold)
        # Using async httpx client
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.invoke_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def analyze_image(self, file_path: str, session_id: str, historical_findings: Optional[str] = None, implant_history: str = "") -> Dict[str, Any]:
        """
        End-to-end pipeline: Route -> Encode -> Prompt -> Check Safety -> Return
        """
        # 1. Routing & Preprocessing
        self.logger.info("vx_agent_routing", file=file_path)
        routed_data = await self.router.route_and_process(file_path, session_id)
        
        if "error" in routed_data:
            return {"error": routed_data["error"], "detail": routed_data["detail"]}
            
        metadata = routed_data.get("metadata", {})
        
        # Extract pixels/image
        img_source = routed_data.get("pixels") if "pixels" in routed_data else routed_data.get("image")
        
        # For initial pass, take the middle slice if it's 3D
        if isinstance(img_source, np.ndarray) and img_source.ndim == 3:
            initial_slice = self.router.dicom_processor.get_representative_slices(img_source, num_slices=1)[0]
        else:
            initial_slice = img_source
            
        base64_image = self._encode_image(initial_slice)
        
        # 2. Construct Prompt Constraints (Artifact Hallucination + ComparisonBuffer)
        prompt = (
            "You are the MediFlow Visual Perception Agent, an expert in radiology.\n"
            "Analyze the provided medical image and return a concise visual finding report.\n\n"
        )
        
        if implant_history:
            prompt += f"--- PATIENT IMPLANT HISTORY ---\n{implant_history}\n(CRITICAL: Ignore artifacts caused by these metallic objects)\n\n"
            
        if historical_findings:
            prompt += f"--- COMPARISON BUFFER: PREVIOUS FINDINGS ---\n{historical_findings}\n(CRITICAL: Perform Delta Analysis. Is the condition stable, growing, or resolving?)\n\n"
            
        prompt += "OUTPUT REQUIREMENT: Return a JSON dictionary with 'ai_generated_preliminary_report' (string), 'key_observations' (list of strings), and 'confidence_score' (float 0.0-1.0)."
        
        # 3. Call VLM
        self.logger.info("vx_agent_calling_vlm", model=self.model_name)
        vlm_response = await self._call_vlm([base64_image], prompt)
        
        # 4. Parse & Validate
        try:
            # Strip markdown block if present
            clean_json = vlm_response.replace("```json", "").replace("```", "").strip()
            finding_data = json.loads(clean_json)
        except Exception as e:
            self.logger.error("vx_agent_parse_fail", response=vlm_response)
            finding_data = {
                "ai_generated_preliminary_report": vlm_response,
                "key_observations": ["Failed to structure output"],
                "confidence_score": 0.5
            }
            
        # 5. Volumetric Skip (Full-Volume Sweep fallback)
        if finding_data.get("confidence_score", 1.0) < self.CONFIDENCE_THRESHOLD and routed_data["type"] in ["DICOM", "NIFTI"]:
            self.logger.info("vx_agent_triggering_volumetric_skip", confidence=finding_data.get("confidence_score"))
            
            pixel_volume = routed_data.get("pixels")
            if isinstance(pixel_volume, np.ndarray) and pixel_volume.ndim == 3:
                # Sample 5 slices for the sweep
                sweep_slices = self.router.dicom_processor.get_representative_slices(pixel_volume, num_slices=5)
                sweep_b64 = [self._encode_image(s) for s in sweep_slices]
                
                sweep_prompt = (
                    "EXHAUSTIVE VOLUMETRIC SWEEP REQUIRED.\n"
                    "The previous single-slice analysis had low confidence. Review these 5 sampled slices across the volume.\n"
                    "Search for subtle lesions, small nodules, or vascular anomalies that might have been missed.\n"
                    "Provide a more comprehensive final report.\n\n"
                ) + prompt # reuse original requirements/context
                
                sweep_response = await self._call_vlm(sweep_b64, sweep_prompt)
                
                try:
                    sweep_json = sweep_response.replace("```json", "").replace("```", "").strip()
                    sweep_data = json.loads(sweep_json)
                    # If sweep is more confident, use it
                    if sweep_data.get("confidence_score", 0) >= finding_data.get("confidence_score", 0):
                        finding_data = sweep_data
                        self.logger.info("vx_agent_sweep_accepted", new_confidence=finding_data.get("confidence_score"))
                except Exception:
                    self.logger.warning("vx_agent_sweep_parse_fail")
            
        # 6. The "Side-Swap" Safety Check
        # Uses dicom_processor to enforce L/R orientation alignment
        try:
            finding_text = str(finding_data)
            self.router.dicom_processor.validate_side_orientation(finding_text, metadata)
        except CriticalOrientationMismatch as e:
            self.logger.error("vx_agent_side_swap_alert", error=str(e))
            return {"error": "CRITICAL_ORIENTATION_MISMATCH", "detail": str(e)}
            
        return {
            "modality": metadata.get("modality", "Unknown"),
            "ai_generated_preliminary_report": finding_data.get("ai_generated_preliminary_report", ""),
            "key_observations": finding_data.get("key_observations", []),
            "confidence_score": finding_data.get("confidence_score", 0.5),
            "is_volumetric_sweep": finding_data.get("confidence_score", 0) >= self.CONFIDENCE_THRESHOLD and routed_data["type"] in ["DICOM", "NIFTI"]
        }
