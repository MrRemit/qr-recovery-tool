#!/usr/bin/env python3
"""
QR Image Processing Module
Extracts QR data from images and analyzes damage
"""

import cv2
import numpy as np
from pyzbar import pyzbar
from typing import Tuple, Dict, Optional

def extract_from_image(image_path: str) -> Tuple[Optional[bytes], Dict]:
    """
    Extract QR code data from image file

    Returns:
        (data, info) where data is bytes and info contains metadata
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None, {}

        # Try direct decode
        decoded_objects = pyzbar.decode(img)

        if decoded_objects:
            obj = decoded_objects[0]
            return obj.data, {
                'type': obj.type,
                'quality': 'good',
                'rect': obj.rect
            }

        # If failed, try preprocessing
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        decoded_objects = pyzbar.decode(thresh)
        if decoded_objects:
            obj = decoded_objects[0]
            return obj.data, {
                'type': obj.type,
                'quality': 'fair',
                'preprocessing': 'threshold',
                'rect': obj.rect
            }

        # 3. Try denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        decoded_objects = pyzbar.decode(denoised)

        if decoded_objects:
            obj = decoded_objects[0]
            return obj.data, {
                'type': obj.type,
                'quality': 'poor',
                'preprocessing': 'denoise',
                'rect': obj.rect
            }

        # Failed to decode
        return None, {'quality': 'failed', 'preprocessing_attempted': True}

    except Exception as e:
        return None, {'error': str(e)}

def analyze_qr_damage(data: bytes, version: int = 3, ec_level: str = 'L') -> Dict:
    """
    Analyze QR code data to estimate damage and suggest recovery strategy

    Returns:
        Dictionary with damage analysis and recommended strategy
    """
    from reedsolo import RSCodec, ReedSolomonError

    # QR Version parameters
    ecc_capacities = {
        'L': {'v1': 7, 'v2': 10, 'v3': 15, 'v4': 20},
        'M': {'v1': 10, 'v2': 16, 'v3': 26, 'v4': 28},
        'Q': {'v1': 13, 'v2': 22, 'v3': 36, 'v4': 44},
        'H': {'v1': 17, 'v2': 28, 'v3': 44, 'v4': 56}
    }

    ecc_codewords = ecc_capacities.get(ec_level, {}).get(f'v{version}', 15)

    # Try Reed-Solomon decode to assess damage
    try:
        rs = RSCodec(ecc_codewords)
        corrected = rs.decode(data)

        # Success - minimal or no damage
        damage_percent = 0.0
        strategy = "quick_ecc"
        estimated_time = "< 1 second"
        critical_positions = []

    except ReedSolomonError as e:
        # Failed - estimate damage from error message
        error_str = str(e)

        # Heuristic: count error positions
        if "Too many errors" in error_str:
            damage_percent = 30.0  # Exceeds ECC capacity
            strategy = "exhaustive_bruteforce"
            estimated_time = "5-60 minutes"
        else:
            damage_percent = 15.0  # Within ECC with help
            strategy = "targeted_bruteforce"
            estimated_time = "30 seconds - 5 minutes"

        # Identify likely damaged regions
        data_array = np.frombuffer(data, dtype=np.uint8)

        # Find positions with unusual values
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)

        outliers = np.where(np.abs(data_array - mean_val) > 2 * std_val)[0]

        # Critical positions for WIF recovery (end of data block)
        wif_critical = list(range(max(0, len(data) - 15), len(data)))

        critical_positions = list(set(outliers.tolist() + wif_critical))[:20]

    return {
        'damage_percent': damage_percent,
        'strategy': strategy,
        'estimated_time': estimated_time,
        'critical_positions': critical_positions,
        'ecc_capacity': ecc_codewords,
        'data_length': len(data)
    }

def enhance_qr_image(image_path: str, output_path: str = None) -> str:
    """
    Enhance QR code image for better scanning

    Returns path to enhanced image
    """
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Denoise
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    # Adaptive threshold
    enhanced = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Save
    if output_path is None:
        output_path = image_path.replace('.', '_enhanced.')

    cv2.imwrite(output_path, enhanced)
    return output_path
