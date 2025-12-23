"""
SUPER NEGATIVE PROCESSING SYSTEM - Image Processor

Standalone image processing functions for ProcessPoolExecutor.
These must be picklable (no class state, importable at module level).
All processing is done at full resolution for professional accuracy.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional


def process_image_full(path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full processing pipeline with auto-detection - runs in worker process.
    Always processes at full resolution for professional accuracy.

    This is the main function for initial image processing:
    - Frame detection (crop)
    - Auto-rotation detection (0/90/180/270)
    - Base color sampling
    - Negative inversion

    Args:
        path: Path to the image file
        settings: Dict with processing settings:
            - bg_threshold: Background threshold (default 15)
            - sensitivity: Detection sensitivity (default 50)
            - auto_rotate: Whether to detect rotation (default True)
            - skip_if_cached: Return early if settings already cached (default False)

    Returns:
        Dict with:
            - thumbnail: Processed thumbnail as numpy array
            - angle: Detected frame angle (fine rotation)
            - rotation: Detected 90° rotation (0, 90, 180, 270)
            - rotation_confidence: Confidence level of rotation detection
            - rotation_method: Method used for rotation detection
            - base_color: Sampled base color as list
            - bg_threshold: Threshold used
            - sensitivity: Sensitivity used
            - inverted: Full resolution inverted image (for preset application)
    """
    # Import here to avoid issues with multiprocessing
    from processing import (
        load_image,
        isolate_negative_from_background,
        detect_frame_angle,
        sample_base_color,
        extract_and_straighten,
        invert_negative,
    )
    from auto_rotate import detect_auto_rotation

    # Load at full resolution (uses RAW cache if available)
    img = load_image(path, use_cache=True)

    # Get settings with defaults
    bg_threshold = settings.get('bg_threshold', 15)
    sensitivity = settings.get('sensitivity', 50)
    do_auto_rotate = settings.get('auto_rotate', True)

    # 1. Isolate negative from background
    mask = isolate_negative_from_background(img, bg_threshold)

    # 2. Detect frame angle (fine rotation)
    angle = detect_frame_angle(img, mask, sensitivity)

    # 3. Sample base color from film border
    base_color = sample_base_color(img, mask)

    # 4. Convert sensitivity to margin (same formula as process_negative_image)
    # sensitivity 0-100 maps to margin 0.01-0.10
    margin = 0.01 + (sensitivity / 100) * 0.09

    # 5. Extract and straighten at full resolution (with 0° rotation initially)
    extracted = extract_and_straighten(img, mask, angle, margin, None)

    # 6. Invert the negative
    inverted = invert_negative(extracted, base_color)

    # 7. Auto-detect rotation (0/90/180/270) on the inverted image
    rotation = 0
    rotation_confidence = 'NONE'
    rotation_method = 'none'

    if do_auto_rotate:
        try:
            rot_result = detect_auto_rotation(inverted)
            rotation = rot_result.rotation
            rotation_confidence = rot_result.confidence.name
            rotation_method = rot_result.method
        except Exception as e:
            print(f"[AutoRotate] Error detecting rotation for {path}: {e}")

    # 8. Apply rotation to inverted image if needed
    if rotation != 0:
        # Rotate the inverted image (must match _apply_rotation in main.py)
        if rotation == 90:
            inverted = cv2.rotate(inverted, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            inverted = cv2.rotate(inverted, cv2.ROTATE_180)
        elif rotation == 270:
            inverted = cv2.rotate(inverted, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 9. Generate thumbnail from rotated full-res result
    h, w = inverted.shape[:2]
    scale = min(100 / w, 74 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    thumbnail = cv2.resize(inverted, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return {
        'thumbnail': thumbnail,
        'inverted': inverted,  # Full-res for preset application
        'angle': float(angle),
        'rotation': rotation,
        'rotation_confidence': rotation_confidence,
        'rotation_method': rotation_method,
        'base_color': base_color.tolist() if hasattr(base_color, 'tolist') else list(base_color),
        'bg_threshold': bg_threshold,
        'sensitivity': sensitivity,
    }


def detect_rotation_parallel(path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-rotation detection - runs in worker process.

    Args:
        path: Path to the image file
        settings: Dict (currently unused, reserved for future options)

    Returns:
        Dict with:
            - rotation: Detected rotation (0, 90, 180, 270)
            - confidence: Confidence level name
            - method: Detection method used
    """
    from processing import load_image, process_negative_image
    from auto_rotate import detect_auto_rotation

    # Load and process image to get inverted version
    img = load_image(path, use_cache=True)

    # Process to get inverted image (rotation detection works on positive image)
    inverted = process_negative_image(img)

    # Detect optimal rotation
    result = detect_auto_rotation(inverted)

    return {
        'rotation': result.rotation,
        'confidence': result.confidence.name,
        'method': result.method,
    }


def apply_preset_to_image(path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply preset adjustments to an image - runs in worker process.

    Args:
        path: Path to the image file
        settings: Dict with:
            - preset: Preset dict with 'adjustments' and 'curves'
            - preset_key: Key identifying the preset

    Returns:
        Dict with:
            - thumbnail: Adjusted thumbnail as numpy array
            - preset_key: The preset key that was applied
    """
    from processing import load_image, process_negative_image

    preset = settings.get('preset', {})
    preset_key = settings.get('preset_key', 'unknown')

    # Load and process to get inverted image
    img = load_image(path, use_cache=True)
    inverted = process_negative_image(img)

    # Apply adjustments
    adjusted = _apply_adjustments(inverted, preset.get('adjustments', {}))

    # Apply curves
    adjusted = _apply_curves(adjusted, preset.get('curves', {}))

    # Generate thumbnail
    h, w = adjusted.shape[:2]
    scale = min(100 / w, 74 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    thumbnail = cv2.resize(adjusted, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return {
        'thumbnail': thumbnail,
        'preset_key': preset_key,
    }


def _apply_adjustments(img: np.ndarray, adjustments: Dict[str, float]) -> np.ndarray:
    """
    Apply adjustment values to an image.
    Simplified version of the adjustments from widgets/adjustments.py.
    """
    result = img.copy()

    # Exposure (multiplicative, 2^value)
    exposure = adjustments.get('exposure', 0.0)
    if exposure != 0.0:
        result = result * (2.0 ** exposure)

    # White balance (channel multipliers)
    wb_r = adjustments.get('wb_r', 0.0)
    wb_g = adjustments.get('wb_g', 0.0)
    wb_b = adjustments.get('wb_b', 0.0)
    if wb_r != 0.0 or wb_g != 0.0 or wb_b != 0.0:
        # BGR order
        result[:, :, 0] = result[:, :, 0] * (1.0 + wb_b * 0.5)
        result[:, :, 1] = result[:, :, 1] * (1.0 + wb_g * 0.5)
        result[:, :, 2] = result[:, :, 2] * (1.0 + wb_r * 0.5)

    # Temperature (blue-orange shift)
    temperature = adjustments.get('temperature', 0.0)
    if temperature != 0.0:
        temp_shift = temperature * 0.1
        result[:, :, 0] = result[:, :, 0] - temp_shift  # Blue
        result[:, :, 2] = result[:, :, 2] + temp_shift  # Red

    # Contrast (S-curve around midpoint)
    contrast = adjustments.get('contrast', 0.0)
    if contrast != 0.0:
        factor = 1.0 + contrast * 0.5
        result = (result - 0.5) * factor + 0.5

    # Gamma
    gamma = adjustments.get('gamma', 0.0)
    if gamma != 0.0:
        gamma_val = 1.0 / (1.0 + gamma * 0.5) if gamma > 0 else 1.0 - gamma * 0.5
        result = np.clip(result, 0, 1)
        result = np.power(result, gamma_val)

    # Blacks (lift shadows)
    blacks = adjustments.get('blacks', 0.0)
    if blacks != 0.0:
        result = result + blacks * 0.1

    # Whites (compress highlights)
    whites = adjustments.get('whites', 0.0)
    if whites != 0.0:
        result = result + (1.0 - result) * whites * 0.1

    # Saturation
    saturation = adjustments.get('saturation', 0.0)
    if saturation != 0.0:
        gray = np.mean(result, axis=2, keepdims=True)
        sat_factor = 1.0 + saturation
        result = gray + (result - gray) * sat_factor

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    return result


def _apply_curves(img: np.ndarray, curves: Dict[str, list]) -> np.ndarray:
    """
    Apply curves adjustments using PCHIP interpolation.
    """
    from scipy.interpolate import PchipInterpolator

    result = img.copy()

    # Create LUT from curve points
    def curve_to_lut(points):
        if not points or len(points) < 2:
            return np.linspace(0, 1, 256)

        x_points = np.array([p[0] for p in points])
        y_points = np.array([p[1] for p in points])

        # PCHIP interpolation (monotonic, no overshoot)
        try:
            interp = PchipInterpolator(x_points, y_points)
            x_full = np.linspace(0, 1, 256)
            lut = np.clip(interp(x_full), 0, 1)
        except Exception:
            lut = np.linspace(0, 1, 256)

        return lut.astype(np.float32)

    # Apply RGB master curve first
    rgb_points = curves.get('rgb', [(0, 0), (1, 1)])
    if rgb_points and len(rgb_points) >= 2:
        lut = curve_to_lut(rgb_points)
        # Apply LUT to all channels
        result_uint8 = (result * 255).astype(np.uint8)
        lut_uint8 = (lut * 255).astype(np.uint8)
        for c in range(3):
            result[:, :, c] = lut[np.clip((result[:, :, c] * 255).astype(np.int32), 0, 255)]

    # Apply per-channel curves (BGR order)
    for channel_name, channel_idx in [('b', 0), ('g', 1), ('r', 2)]:
        channel_points = curves.get(channel_name, [(0, 0), (1, 1)])
        if channel_points and len(channel_points) >= 2:
            lut = curve_to_lut(channel_points)
            indices = np.clip((result[:, :, channel_idx] * 255).astype(np.int32), 0, 255)
            result[:, :, channel_idx] = lut[indices]

    return np.clip(result, 0.0, 1.0)
