"""
src/agents/vision/quality_gate.py
---------------------------------
Enforces strict quality controls on patient-uploaded standard images
(e.g., JPEG, PNG) before they reach the Vision-Language Model.
"""

from PIL import Image, ImageStat
from typing import Tuple

MIN_RESOLUTION = (512, 512)
MIN_ENTROPY = 4.0  # Simple proxy for "contains actual complex information vs blank/blurry"

def validate_image_quality(img: Image.Image) -> Tuple[bool, str]:
    """
    Checks if an image meets the minimum quality requirements for diagnostic AI.
    
    Returns:
        (is_valid: bool, reason: str)
    """
    width, height = img.size
    
    # 1. Resolution Check
    if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
        return False, f"Resolution too low ({width}x{height}). Minimum required is {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}."
        
    # 2. Entropy Check (Rough proxy for detail/blur)
    # A completely blurry or blank image will have very low entropy.
    try:
        # Convert to grayscale for entropy calculation
        gray_img = img.convert('L')
        stat = ImageStat.Stat(gray_img)
        # stat.extrema gives min/max, we can calculate actual Shannon entropy but let's 
        # use standard deviation as a proxy for contrast/detail
        std_dev = stat.stddev[0]
        
        # If standard deviation of pixel intensities is very low > image is washed out or blurry
        if std_dev < 15.0:
            return False, f"Image lacks sufficient contrast/detail (StdDev: {std_dev:.2f}). Please provide a clearer scan."
    except Exception as e:
        # If we can't calculate stats, assume it's corrupted
        return False, f"Failed to analyze image quality: {e}"
        
    return True, "Quality checks passed."
