#!/usr/bin/env python3
"""
Auto-rotation detection based on image content.

Uses multiple heuristics to detect which way is "up" in an image:
1. Face detection - faces should be upright
2. Gradient analysis - sky/bright areas typically at top
3. Edge distribution - natural images have specific patterns

All methods use only OpenCV (no additional dependencies).
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class RotationConfidence(Enum):
    """Confidence level for rotation detection."""
    HIGH = "high"      # Face detected, very reliable
    MEDIUM = "medium"  # Multiple heuristics agree
    LOW = "low"        # Single weak signal
    NONE = "none"      # No detection possible


@dataclass
class RotationResult:
    """Result of auto-rotation detection."""
    rotation: int  # 0, 90, 180, or 270 degrees
    confidence: RotationConfidence
    method: str  # Which method determined the rotation
    details: dict  # Additional info for debugging


def rotate_image_90(img: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate image by 0, 90, 180, or 270 degrees."""
    rotation = rotation % 360
    if rotation == 0:
        return img
    elif rotation == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# =============================================================================
# FACE DETECTION
# =============================================================================

def _load_face_cascade() -> cv2.CascadeClassifier:
    """Load the Haar cascade for face detection."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade from {cascade_path}")
    return cascade


def _load_eye_cascade() -> cv2.CascadeClassifier:
    """Load the Haar cascade for eye detection."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade from {cascade_path}")
    return cascade


# HOG Person Detector - built into OpenCV, very reliable
_hog_detector = None

def _get_hog_detector():
    """Get the HOG person detector (lazy initialization)."""
    global _hog_detector
    if _hog_detector is None:
        _hog_detector = cv2.HOGDescriptor()
        _hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog_detector


def _load_upperbody_cascade() -> cv2.CascadeClassifier:
    """Load the Haar cascade for upper body detection."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade from {cascade_path}")
    return cascade


def _detect_faces_in_orientation(
    gray: np.ndarray,
    cascade: cv2.CascadeClassifier,
    min_face_size: Tuple[int, int] = (60, 60)
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a grayscale image.

    Returns list of (x, y, w, h) rectangles for detected faces.
    """
    h, w = gray.shape[:2]

    # Resize for faster detection if image is large
    scale = 1.0
    if max(gray.shape) > 1000:
        scale = 1000 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
        min_face_size = (int(min_face_size[0] * scale), int(min_face_size[1] * scale))

    # Use stricter parameters to reduce false positives
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,  # Increased from 5 - requires more consistent detection
        minSize=min_face_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return []

    # Scale back to original coordinates
    if scale != 1.0:
        faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale))
                 for x, y, w, h in faces]
    else:
        faces = [tuple(f) for f in faces]

    # Filter out unlikely faces:
    # - Too small (less than 5% of image dimension)
    # - Aspect ratio too wrong (faces are roughly square)
    min_dim = min(h, w) * 0.05
    valid_faces = []
    for x, y, fw, fh in faces:
        # Check minimum size
        if fw < min_dim or fh < min_dim:
            continue
        # Check aspect ratio (faces are roughly square, allow 0.6 to 1.6)
        aspect = fw / fh
        if aspect < 0.6 or aspect > 1.6:
            continue
        valid_faces.append((x, y, fw, fh))

    return valid_faces


def detect_rotation_by_faces(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect correct rotation by finding faces.

    Tries each 90° rotation and returns the one where faces are detected.
    Faces are only detected when upright, making this very reliable.
    """
    try:
        cascade = _load_face_cascade()
    except RuntimeError:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    img_area = gray.shape[0] * gray.shape[1]

    results = []

    for rotation in [0, 90, 180, 270]:
        rotated = rotate_image_90(gray, rotation)
        faces = _detect_faces_in_orientation(rotated, cascade)

        if faces:
            # Calculate total face area as confidence metric
            total_area = sum(w * h for x, y, w, h in faces)
            # Calculate largest face as percentage of image
            max_face_area = max(w * h for x, y, w, h in faces)
            face_percentage = max_face_area / img_area
            results.append((rotation, len(faces), total_area, face_percentage, faces))

    if not results:
        return None

    # Pick rotation with most faces, or largest face area if tie
    best = max(results, key=lambda x: (x[1], x[2]))
    rotation, num_faces, total_area, face_percentage, faces = best

    # Only return if we're confident this is a real face:
    # - Multiple faces detected, OR
    # - Single face that's reasonably large (>1% of image)
    if num_faces == 1 and face_percentage < 0.01:
        return None  # Single tiny "face" is likely false positive

    # Confidence based on detection quality
    if num_faces >= 2 or face_percentage > 0.03:
        confidence = RotationConfidence.HIGH
    elif face_percentage > 0.01:
        confidence = RotationConfidence.MEDIUM
    else:
        confidence = RotationConfidence.LOW

    return RotationResult(
        rotation=rotation,
        confidence=confidence,
        method="face_detection",
        details={
            "num_faces": num_faces,
            "total_face_area": total_area,
            "largest_face_percentage": round(face_percentage * 100, 2),
            "face_rects": faces
        }
    )


# =============================================================================
# PERSON DETECTION (HOG - built into OpenCV)
# =============================================================================

def detect_rotation_by_people(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect rotation by finding standing people using HOG detector.

    People should be upright (head at top, feet at bottom).
    Very reliable when people are fully visible.
    """
    hog = _get_hog_detector()

    img_area = img.shape[0] * img.shape[1]
    results = []

    for rotation in [0, 90, 180, 270]:
        rotated = rotate_image_90(img, rotation)

        # Resize for faster detection
        scale = 1.0
        if max(rotated.shape[:2]) > 600:
            scale = 600 / max(rotated.shape[:2])
            small = cv2.resize(rotated, None, fx=scale, fy=scale)
        else:
            small = rotated

        # Detect people
        boxes, weights = hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )

        if len(boxes) > 0:
            # Scale back and filter
            valid_detections = []
            for (x, y, w, h), weight in zip(boxes, weights):
                if scale != 1.0:
                    x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)

                # People are taller than wide
                aspect = h / w if w > 0 else 0
                if aspect > 1.2 and weight > 0.3:
                    area = w * h
                    valid_detections.append((x, y, w, h, float(weight), area))

            if valid_detections:
                total_area = sum(d[5] for d in valid_detections)
                max_weight = max(d[4] for d in valid_detections)
                results.append((rotation, len(valid_detections), total_area, max_weight))

    if not results:
        return None

    # Pick rotation with most people, or highest confidence
    best = max(results, key=lambda x: (x[1], x[3]))
    rotation, num_people, total_area, max_weight = best

    # Confidence based on detection quality
    if num_people >= 2 or max_weight > 0.8:
        confidence = RotationConfidence.HIGH
    elif max_weight > 0.5:
        confidence = RotationConfidence.MEDIUM
    else:
        confidence = RotationConfidence.LOW

    return RotationResult(
        rotation=rotation,
        confidence=confidence,
        method="person_detection",
        details={
            "num_people": num_people,
            "max_confidence": round(max_weight, 2),
            "total_area": total_area
        }
    )


# =============================================================================
# UPPER BODY DETECTION
# =============================================================================

def detect_rotation_by_upperbody(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect rotation by finding upper bodies.

    More reliable than full face when person is at medium distance.
    """
    try:
        cascade = _load_upperbody_cascade()
    except RuntimeError:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    img_area = gray.shape[0] * gray.shape[1]

    results = []

    for rotation in [0, 90, 180, 270]:
        rotated = rotate_image_90(gray, rotation)

        # Resize for speed
        scale = 1.0
        if max(rotated.shape) > 800:
            scale = 800 / max(rotated.shape)
            small = cv2.resize(rotated, None, fx=scale, fy=scale)
        else:
            small = rotated

        bodies = cascade.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(bodies) > 0:
            # Scale back and calculate area
            valid_bodies = []
            for (x, y, w, h) in bodies:
                if scale != 1.0:
                    x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                area = w * h
                # Upper body is wider than tall usually
                aspect = w / h if h > 0 else 0
                if 0.5 < aspect < 2.0:
                    valid_bodies.append((x, y, w, h, area))

            if valid_bodies:
                total_area = sum(b[4] for b in valid_bodies)
                max_area_pct = max(b[4] for b in valid_bodies) / img_area
                results.append((rotation, len(valid_bodies), total_area, max_area_pct))

    if not results:
        return None

    best = max(results, key=lambda x: (x[1], x[2]))
    rotation, num_bodies, total_area, max_area_pct = best

    # Only trust if detection is reasonably large
    if max_area_pct < 0.01:
        return None

    if num_bodies >= 2 or max_area_pct > 0.05:
        confidence = RotationConfidence.MEDIUM
    else:
        confidence = RotationConfidence.LOW

    return RotationResult(
        rotation=rotation,
        confidence=confidence,
        method="upperbody_detection",
        details={
            "num_bodies": num_bodies,
            "largest_body_percentage": round(max_area_pct * 100, 2)
        }
    )


# =============================================================================
# GRADIENT/SKY DETECTION
# =============================================================================

def detect_rotation_by_gradient(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect rotation by analyzing brightness gradients.

    In most photos:
    - Sky/ceiling is at the top (brighter)
    - Ground/floor is at the bottom (darker)
    - Vertical gradients are more common than horizontal
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = gray.astype(np.float32)

    h, w = gray.shape

    # Divide image into regions
    top_region = gray[:h//4, :]
    bottom_region = gray[3*h//4:, :]
    left_region = gray[:, :w//4]
    right_region = gray[:, 3*w//4:]

    # Calculate mean brightness for each region
    top_brightness = np.mean(top_region)
    bottom_brightness = np.mean(bottom_region)
    left_brightness = np.mean(left_region)
    right_brightness = np.mean(right_region)

    # Calculate differences
    vertical_diff = top_brightness - bottom_brightness  # positive = top brighter
    horizontal_diff = left_brightness - right_brightness

    # Threshold for significance (needs noticeable difference)
    threshold = 15.0

    details = {
        "top_brightness": float(top_brightness),
        "bottom_brightness": float(bottom_brightness),
        "left_brightness": float(left_brightness),
        "right_brightness": float(right_brightness),
        "vertical_diff": float(vertical_diff),
        "horizontal_diff": float(horizontal_diff)
    }

    # Determine strongest gradient direction
    abs_v = abs(vertical_diff)
    abs_h = abs(horizontal_diff)

    if abs_v < threshold and abs_h < threshold:
        # No significant gradient detected
        return None

    if abs_v >= abs_h:
        # Vertical gradient is dominant
        if vertical_diff > threshold:
            # Top is brighter - correct orientation
            return RotationResult(
                rotation=0,
                confidence=RotationConfidence.LOW,
                method="gradient_vertical",
                details=details
            )
        elif vertical_diff < -threshold:
            # Bottom is brighter - upside down
            return RotationResult(
                rotation=180,
                confidence=RotationConfidence.LOW,
                method="gradient_vertical",
                details=details
            )
    else:
        # Horizontal gradient is dominant - image might be rotated 90°
        if horizontal_diff > threshold:
            # Left is brighter - rotated 90° CCW
            return RotationResult(
                rotation=90,
                confidence=RotationConfidence.LOW,
                method="gradient_horizontal",
                details=details
            )
        elif horizontal_diff < -threshold:
            # Right is brighter - rotated 90° CW
            return RotationResult(
                rotation=270,
                confidence=RotationConfidence.LOW,
                method="gradient_horizontal",
                details=details
            )

    return None


# =============================================================================
# EDGE DISTRIBUTION ANALYSIS
# =============================================================================

def detect_rotation_by_edges(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect rotation by analyzing edge distribution.

    Natural images tend to have:
    - More horizontal edges at the bottom (ground, horizon)
    - More vertical edges on the sides (trees, buildings, people)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape

    # Calculate edge density in each region
    top_edges = np.sum(edges[:h//3, :]) / (h//3 * w)
    bottom_edges = np.sum(edges[2*h//3:, :]) / (h//3 * w)

    # In natural images, bottom typically has more edges (ground detail)
    edge_ratio = bottom_edges / (top_edges + 1e-6)

    details = {
        "top_edge_density": float(top_edges),
        "bottom_edge_density": float(bottom_edges),
        "ratio": float(edge_ratio)
    }

    # This is a weak signal, only use if ratio is significant
    if edge_ratio > 1.5:
        return RotationResult(
            rotation=0,
            confidence=RotationConfidence.LOW,
            method="edge_distribution",
            details=details
        )
    elif edge_ratio < 0.67:
        return RotationResult(
            rotation=180,
            confidence=RotationConfidence.LOW,
            method="edge_distribution",
            details=details
        )

    return None


# =============================================================================
# COLOR DISTRIBUTION (Sky detection)
# =============================================================================

def detect_rotation_by_sky(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect rotation by looking for sky-like colors (blue) at the top.

    Works well for outdoor photos with visible sky.
    """
    if len(img.shape) != 3:
        return None

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = img.shape[:2]

    # Define sky-blue color range in HSV
    # Hue: 90-130 (blue range), Saturation: 30-255, Value: 100-255
    lower_blue = np.array([90, 30, 100])
    upper_blue = np.array([130, 255, 255])

    # Check each edge region for sky-like pixels
    regions = {
        0: hsv[:h//4, :],      # top
        180: hsv[3*h//4:, :],  # bottom
        270: hsv[:, :w//4],    # left
        90: hsv[:, 3*w//4:]    # right
    }

    sky_ratios = {}
    for rotation, region in regions.items():
        mask = cv2.inRange(region, lower_blue, upper_blue)
        sky_ratio = np.sum(mask > 0) / mask.size
        sky_ratios[rotation] = sky_ratio

    details = {"sky_ratios": {k: float(v) for k, v in sky_ratios.items()}}

    # Find region with most sky-like pixels
    best_rotation = max(sky_ratios, key=sky_ratios.get)
    best_ratio = sky_ratios[best_rotation]

    # Need significant sky presence (at least 10% of region)
    if best_ratio < 0.10:
        return None

    # Sky should be significantly more present than other edges
    other_ratios = [v for k, v in sky_ratios.items() if k != best_rotation]
    if other_ratios and best_ratio < 2 * max(other_ratios):
        return None

    return RotationResult(
        rotation=best_rotation,
        confidence=RotationConfidence.MEDIUM if best_ratio > 0.20 else RotationConfidence.LOW,
        method="sky_detection",
        details=details
    )


# =============================================================================
# EYE DETECTION (more reliable than face alone)
# =============================================================================

def _detect_eyes_in_orientation(
    gray: np.ndarray,
    eye_cascade: cv2.CascadeClassifier,
    min_eye_size: Tuple[int, int] = (25, 25)
) -> List[Tuple[int, int, int, int]]:
    """Detect eyes in a grayscale image."""
    h, w = gray.shape[:2]

    # Resize for faster detection
    scale = 1.0
    if max(gray.shape) > 800:
        scale = 800 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
        min_eye_size = (int(min_eye_size[0] * scale), int(min_eye_size[1] * scale))

    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,  # Stricter to reduce false positives
        minSize=min_eye_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(eyes) == 0:
        return []

    # Scale back
    if scale != 1.0:
        eyes = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale))
                for x, y, w, h in eyes]
    else:
        eyes = [tuple(e) for e in eyes]

    # Filter: eyes should be roughly square and not tiny
    min_dim = min(h, w) * 0.02
    valid_eyes = []
    for x, y, ew, eh in eyes:
        if ew < min_dim or eh < min_dim:
            continue
        aspect = ew / eh
        if aspect < 0.5 or aspect > 2.0:
            continue
        valid_eyes.append((x, y, ew, eh))

    return valid_eyes


def detect_rotation_by_eyes(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect rotation by finding pairs of eyes.

    Eyes should be:
    - Horizontally aligned (not vertically)
    - In the upper portion of a face region
    """
    try:
        eye_cascade = _load_eye_cascade()
    except RuntimeError:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    results = []

    for rotation in [0, 90, 180, 270]:
        rotated = rotate_image_90(gray, rotation)
        eyes = _detect_eyes_in_orientation(rotated, eye_cascade)

        if len(eyes) >= 2:
            # Check if any pair of eyes is horizontally aligned
            # (y coordinates similar, x coordinates different)
            for i, (x1, y1, w1, h1) in enumerate(eyes):
                for x2, y2, w2, h2 in eyes[i+1:]:
                    cy1, cy2 = y1 + h1//2, y2 + h2//2
                    cx1, cx2 = x1 + w1//2, x2 + w2//2

                    # Eyes should be horizontally aligned
                    vertical_dist = abs(cy1 - cy2)
                    horizontal_dist = abs(cx1 - cx2)

                    # Good eye pair: horizontal distance > vertical distance
                    if horizontal_dist > vertical_dist * 2 and horizontal_dist > 30:
                        total_area = w1 * h1 + w2 * h2
                        results.append((rotation, len(eyes), total_area))
                        break
                else:
                    continue
                break

    if not results:
        return None

    # Pick rotation with most/best eye pairs
    best = max(results, key=lambda x: (x[1], x[2]))
    rotation, num_eyes, total_area = best

    return RotationResult(
        rotation=rotation,
        confidence=RotationConfidence.HIGH,
        method="eye_detection",
        details={
            "num_eyes": num_eyes,
            "total_eye_area": total_area
        }
    )


# =============================================================================
# COLOR TEMPERATURE DISTRIBUTION
# =============================================================================

def detect_rotation_by_color_temperature(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect rotation by analyzing color temperature distribution.

    In natural outdoor photos:
    - Sky (top) tends to be cooler (blue-ish)
    - Ground (bottom) tends to be warmer (yellow/brown/green)

    This is more robust than simple brightness for many scenes.
    """
    if len(img.shape) != 3:
        return None

    # Convert to LAB color space
    # L = lightness, A = green-red, B = blue-yellow
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    h, w = img.shape[:2]

    # Extract the B channel (blue-yellow axis)
    # Lower B = more blue (cooler), Higher B = more yellow (warmer)
    b_channel = lab[:, :, 2].astype(np.float32)

    # Sample regions
    top_temp = np.mean(b_channel[:h//4, :])
    bottom_temp = np.mean(b_channel[3*h//4:, :])
    left_temp = np.mean(b_channel[:, :w//4])
    right_temp = np.mean(b_channel[:, 3*w//4:])

    # Calculate differences (positive = top is warmer than bottom)
    vertical_diff = top_temp - bottom_temp
    horizontal_diff = left_temp - right_temp

    # Threshold - needs noticeable temperature difference
    threshold = 5.0

    details = {
        "top_temp": float(top_temp),
        "bottom_temp": float(bottom_temp),
        "left_temp": float(left_temp),
        "right_temp": float(right_temp),
        "vertical_diff": float(vertical_diff),
        "horizontal_diff": float(horizontal_diff)
    }

    abs_v = abs(vertical_diff)
    abs_h = abs(horizontal_diff)

    if abs_v < threshold and abs_h < threshold:
        return None

    if abs_v >= abs_h:
        if vertical_diff < -threshold:
            # Top is cooler (more blue) - correct orientation
            return RotationResult(
                rotation=0,
                confidence=RotationConfidence.LOW,
                method="color_temperature",
                details=details
            )
        elif vertical_diff > threshold:
            # Bottom is cooler - upside down
            return RotationResult(
                rotation=180,
                confidence=RotationConfidence.LOW,
                method="color_temperature",
                details=details
            )
    else:
        if horizontal_diff < -threshold:
            # Left is cooler - rotated
            return RotationResult(
                rotation=90,
                confidence=RotationConfidence.LOW,
                method="color_temperature",
                details=details
            )
        elif horizontal_diff > threshold:
            # Right is cooler
            return RotationResult(
                rotation=270,
                confidence=RotationConfidence.LOW,
                method="color_temperature",
                details=details
            )

    return None


# =============================================================================
# TEXT DETECTION (using Tesseract if available)
# =============================================================================

# Check if pytesseract is available
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def detect_rotation_by_text(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect rotation using OCR to find text orientation.

    Text must be readable in the correct orientation.
    Requires: pip install pytesseract
    And tesseract installed: brew install tesseract
    """
    if not HAS_TESSERACT:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    try:
        # Use Tesseract's orientation and script detection
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)

        rotation = osd.get('rotate', 0)
        confidence = osd.get('orientation_conf', 0)

        if confidence < 1.0:
            return None

        # Tesseract returns the rotation needed to make text upright
        # We need to apply this rotation
        rotation = int(rotation) % 360

        details = {
            "tesseract_rotation": rotation,
            "tesseract_confidence": confidence,
            "script": osd.get('script', 'unknown')
        }

        if rotation == 0:
            return RotationResult(
                rotation=0,
                confidence=RotationConfidence.HIGH if confidence > 5 else RotationConfidence.MEDIUM,
                method="text_detection",
                details=details
            )
        else:
            return RotationResult(
                rotation=rotation,
                confidence=RotationConfidence.HIGH if confidence > 5 else RotationConfidence.MEDIUM,
                method="text_detection",
                details=details
            )

    except Exception:
        return None


# =============================================================================
# HORIZON LINE DETECTION
# =============================================================================

def detect_rotation_by_horizon(img: np.ndarray) -> Optional[RotationResult]:
    """
    Detect rotation by finding dominant horizontal lines (horizon).

    In properly oriented landscape photos, the horizon should be horizontal.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=min(gray.shape) // 4,
        maxLineGap=20
    )

    if lines is None or len(lines) < 3:
        return None

    # Analyze line orientations
    horizontal_lengths = []  # Lines close to 0° or 180°
    vertical_lengths = []    # Lines close to 90°

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

        # Normalize to 0-90 range
        if angle > 90:
            angle = 180 - angle

        if angle < 20:  # Horizontal-ish
            horizontal_lengths.append(length)
        elif angle > 70:  # Vertical-ish
            vertical_lengths.append(length)

    total_h = sum(horizontal_lengths)
    total_v = sum(vertical_lengths)

    details = {
        "horizontal_line_length": float(total_h),
        "vertical_line_length": float(total_v),
        "h_count": len(horizontal_lengths),
        "v_count": len(vertical_lengths)
    }

    # If there are significantly more horizontal lines, image is likely upright
    if total_h > total_v * 1.5 and total_h > 500:
        return RotationResult(
            rotation=0,
            confidence=RotationConfidence.LOW,
            method="horizon_detection",
            details=details
        )
    elif total_v > total_h * 1.5 and total_v > 500:
        # Dominant vertical lines suggest 90° rotation needed
        return RotationResult(
            rotation=90,
            confidence=RotationConfidence.LOW,
            method="horizon_detection",
            details=details
        )

    return None


# =============================================================================
# COMBINED DETECTION
# =============================================================================

def detect_auto_rotation(
    img: np.ndarray,
    use_faces: bool = True,
    use_people: bool = True,
    use_upperbody: bool = True,
    use_eyes: bool = True,
    use_text: bool = True,
    use_gradient: bool = True,
    use_color_temp: bool = True,
    use_edges: bool = True,
    use_sky: bool = True,
    use_horizon: bool = True
) -> RotationResult:
    """
    Detect the correct rotation using multiple methods.

    Priority (immediate return if HIGH confidence):
    1. Face detection (most reliable when faces present)
    2. Person detection (HOG - full body standing people)
    3. Upper body detection (catches more cases than face)
    4. Eye detection (catches partial faces)
    5. Text detection (unambiguous when text present)

    Vote-based (combined):
    6. Sky detection (good for outdoor photos)
    7. Color temperature (outdoor scenes)
    8. Horizon detection (landscapes)
    9. Gradient analysis (general purpose)
    10. Edge distribution (weak fallback)

    Args:
        img: BGR image (as from cv2.imread or after frame extraction)
        use_faces: Enable face detection
        use_people: Enable HOG person detection
        use_upperbody: Enable upper body detection
        use_eyes: Enable eye detection
        use_text: Enable text/OCR detection
        use_gradient: Enable gradient analysis
        use_color_temp: Enable color temperature analysis
        use_edges: Enable edge distribution analysis
        use_sky: Enable sky color detection
        use_horizon: Enable horizon line detection

    Returns:
        RotationResult with suggested rotation and confidence
    """
    # Convert float32 (0-1) to uint8 for all detection methods
    if img.dtype == np.float32:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    results = []

    # Method 1: Face detection (highest priority)
    if use_faces:
        face_result = detect_rotation_by_faces(img)
        if face_result and face_result.confidence == RotationConfidence.HIGH:
            return face_result
        elif face_result:
            results.append(face_result)

    # Method 2: Person detection (HOG - full standing people)
    if use_people:
        people_result = detect_rotation_by_people(img)
        if people_result and people_result.confidence == RotationConfidence.HIGH:
            return people_result
        elif people_result:
            results.append(people_result)

    # Method 3: Upper body detection
    if use_upperbody:
        upperbody_result = detect_rotation_by_upperbody(img)
        if upperbody_result:
            results.append(upperbody_result)

    # Method 4: Eye detection (catches partial faces)
    if use_eyes:
        eye_result = detect_rotation_by_eyes(img)
        if eye_result and eye_result.confidence == RotationConfidence.HIGH:
            return eye_result
        elif eye_result:
            results.append(eye_result)

    # Method 5: Text detection (unambiguous when present)
    if use_text:
        text_result = detect_rotation_by_text(img)
        if text_result and text_result.confidence == RotationConfidence.HIGH:
            return text_result
        elif text_result:
            results.append(text_result)

    # Method 6: Sky detection
    if use_sky:
        sky_result = detect_rotation_by_sky(img)
        if sky_result:
            results.append(sky_result)

    # Method 7: Color temperature
    if use_color_temp:
        color_temp_result = detect_rotation_by_color_temperature(img)
        if color_temp_result:
            results.append(color_temp_result)

    # Method 6: Horizon detection
    if use_horizon:
        horizon_result = detect_rotation_by_horizon(img)
        if horizon_result:
            results.append(horizon_result)

    # Method 7: Gradient analysis
    if use_gradient:
        gradient_result = detect_rotation_by_gradient(img)
        if gradient_result:
            results.append(gradient_result)

    # Method 8: Edge distribution
    if use_edges:
        edge_result = detect_rotation_by_edges(img)
        if edge_result:
            results.append(edge_result)

    if not results:
        # No detection possible
        return RotationResult(
            rotation=0,
            confidence=RotationConfidence.NONE,
            method="none",
            details={"message": "No reliable rotation signal detected"}
        )

    # Vote among results
    rotation_votes = {}
    for result in results:
        rot = result.rotation
        # Weight by confidence
        weight = {
            RotationConfidence.HIGH: 3,
            RotationConfidence.MEDIUM: 2,
            RotationConfidence.LOW: 1,
            RotationConfidence.NONE: 0
        }[result.confidence]

        rotation_votes[rot] = rotation_votes.get(rot, 0) + weight

    # Find winning rotation
    best_rotation = max(rotation_votes, key=rotation_votes.get)
    best_votes = rotation_votes[best_rotation]

    # Determine combined confidence
    if best_votes >= 3:
        confidence = RotationConfidence.MEDIUM
    elif best_votes >= 2:
        confidence = RotationConfidence.LOW
    else:
        confidence = RotationConfidence.LOW

    # Find which methods contributed to winning rotation
    contributing_methods = [r.method for r in results if r.rotation == best_rotation]

    return RotationResult(
        rotation=best_rotation,
        confidence=confidence,
        method=f"combined({','.join(contributing_methods)})",
        details={
            "votes": rotation_votes,
            "individual_results": [
                {"rotation": r.rotation, "method": r.method, "confidence": r.confidence.value}
                for r in results
            ]
        }
    )


# =============================================================================
# CLI TESTING
# =============================================================================

def main():
    """Test auto-rotation on an image."""
    import argparse

    parser = argparse.ArgumentParser(description="Test auto-rotation detection")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--no-faces", action="store_true", help="Disable face detection")
    parser.add_argument("--no-gradient", action="store_true", help="Disable gradient analysis")
    parser.add_argument("--no-edges", action="store_true", help="Disable edge analysis")
    parser.add_argument("--no-sky", action="store_true", help="Disable sky detection")
    parser.add_argument("--output", "-o", help="Save rotated image to path")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load {args.image}")
        return 1

    print(f"Image: {args.image} ({img.shape[1]}x{img.shape[0]})")
    print()

    result = detect_auto_rotation(
        img,
        use_faces=not args.no_faces,
        use_gradient=not args.no_gradient,
        use_edges=not args.no_edges,
        use_sky=not args.no_sky
    )

    print(f"Suggested rotation: {result.rotation}°")
    print(f"Confidence: {result.confidence.value}")
    print(f"Method: {result.method}")
    print(f"Details: {result.details}")

    if args.output and result.rotation != 0:
        rotated = rotate_image_90(img, result.rotation)
        cv2.imwrite(args.output, rotated)
        print(f"\nSaved rotated image to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
