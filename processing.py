"""
SUPER NEGATIVE PROCESSING SYSTEM - Image Processing Core

Frame detection, negative inversion, and image processing functions.
Handles RAW formats, film base color sampling, and frame extraction.
"""

import cv2
import numpy as np
from pathlib import Path
import hashlib
from scipy import optimize

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

# RAW file extensions supported by rawpy/LibRaw
RAW_EXTENSIONS = {'.nef', '.cr2', '.cr3', '.arw', '.dng', '.orf', '.rw2', '.raf', '.pef', '.srw'}


def _compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file for cache identification."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image(path: str, use_cache: bool = True) -> np.ndarray:
    """Load image as float32 BGR (0.0-1.0 range). Supports RAW formats (NEF, CR2, etc.) via rawpy.

    For RAW files, caches the demosaiced result to disk as 16-bit PNG for faster subsequent loads.
    All images are returned as float32 normalized to 0.0-1.0 range to preserve full bit depth.
    """
    path_obj = Path(path)
    ext = path_obj.suffix.lower()

    if ext in RAW_EXTENSIONS:
        if not HAS_RAWPY:
            raise ImportError(f"rawpy is required to open {ext.upper()} files. Install with: pip install rawpy")

        # Try to load from cache first
        if use_cache:
            try:
                import storage
                file_hash = _compute_file_hash(path)
                cached = storage.get_storage().load_raw_cache(file_hash)
                if cached is not None:
                    return cached  # Already float32 0-1
            except ImportError:
                pass  # storage module not available (e.g., CLI usage)

        # Demosaic the RAW file at 16-bit
        with rawpy.imread(path) as raw:
            # postprocess with 16-bit output to preserve RAW dynamic range
            rgb = raw.postprocess(output_bps=16)
            # Convert RGB to BGR for OpenCV compatibility
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Normalize to float32 0.0-1.0 range
            img = img.astype(np.float32) / 65535.0

        # Save to cache for next time
        if use_cache:
            try:
                import storage
                storage.get_storage().save_raw_cache(file_hash, img)
            except (ImportError, NameError):
                pass  # storage module not available or hash not computed
    else:
        # Load standard formats (TIFF, PNG, JPEG) preserving bit depth
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Normalize based on original bit depth
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            # float32 images are assumed to already be normalized

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def isolate_negative_from_background(img: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Separate the film negative from the black scanner background.

    Args:
        img: BGR image as float32 (0-1) or uint8 (0-255)
        threshold: Brightness threshold. If None, auto-detects based on dtype.
                   For float32: use 0-1 range (default ~0.06)
                   For uint8: use 0-255 range (default ~15)

    Returns a binary mask where 255 = negative, 0 = background.
    """
    # Convert to grayscale - handle both float32 and uint8
    if img.dtype == np.float32:
        # Convert float32 (0-1) to uint8 for grayscale conversion
        img_8bit = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
        # Default threshold for float32 input (15/255 ≈ 0.06)
        if threshold is None:
            threshold = 15
        else:
            # If threshold given as 0-1 float, convert to 0-255
            if threshold <= 1.0:
                threshold = int(threshold * 255)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if threshold is None:
            threshold = 15

    # Simple threshold - black background is very dark
    _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def sample_base_color(img: np.ndarray, negative_mask: np.ndarray, border_width: int = 50) -> np.ndarray:
    """
    Sample the film base color from the border region of the negative.

    Strategy: Erode the negative mask to get the inner region, then XOR
    with original to get just the border. Sample median color from there.

    Args:
        img: BGR image as float32 (0-1) or uint8 (0-255)

    Returns:
        Base color in LAB as uint8 array (for compatibility with cv2.cvtColor)
    """
    # Erode to get inner region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (border_width, border_width))
    inner_mask = cv2.erode(negative_mask, kernel)

    # Border = negative minus inner
    border_mask = cv2.bitwise_and(negative_mask, cv2.bitwise_not(inner_mask))

    # Convert to LAB for better color representation
    # OpenCV LAB conversion requires uint8 input
    if img.dtype == np.float32:
        img_8bit = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Sample pixels from border region
    border_pixels = lab[border_mask > 0]

    if len(border_pixels) == 0:
        raise ValueError("No border pixels found - check negative mask")

    # Use median to resist outliers (dust, scratches, etc.)
    base_color_lab = np.median(border_pixels, axis=0).astype(np.uint8)

    return base_color_lab


def invert_negative(img: np.ndarray, base_color_bgr: np.ndarray) -> np.ndarray:
    """
    Invert a color negative using the base (orange mask) color.

    Args:
        img: BGR image as float32 (0-1 range)
        base_color_bgr: Base color as float32 (0-1) or uint8 (0-255) BGR array

    Returns:
        Inverted image as float32 (0-1 range)
    """
    # Ensure base_color is in 0-1 range
    if base_color_bgr.dtype == np.uint8 or np.max(base_color_bgr) > 1.0:
        base_f = base_color_bgr.astype(np.float32) / 255.0
    else:
        base_f = base_color_bgr.astype(np.float32)

    # Avoid division by zero
    base_f = np.maximum(base_f, 1.0 / 65535.0)

    # Normalize by base color and invert
    scale = 0.5 / base_f
    result = img.astype(np.float32).copy()
    for c in range(3):
        result[:, :, c] = 1.0 - np.clip(result[:, :, c] * scale[c], 0, 1)

    # Auto-levels per channel (1st to 99th percentile)
    h, w = result.shape[:2]
    step = max(1, max(h, w) // 400)
    sampled = result[::step, ::step]

    for c in range(3):
        p1, p99 = np.percentile(sampled[:, :, c], [1, 99])
        range_val = max(p99 - p1, 1.0 / 65535.0)
        result[:, :, c] = (result[:, :, c] - p1) / range_val

    return np.clip(result, 0, 1)


def detect_frame_angle(img: np.ndarray, negative_mask: np.ndarray = None,
                       canny_low: int = 50, canny_high: int = 150) -> float:
    """
    Detect the rotation angle of a film frame using Hough Lines.

    Args:
        img: BGR image as float32 (0-1) or uint8 (0-255)
        negative_mask: Optional mask to constrain detection
        canny_low: Lower Canny threshold
        canny_high: Upper Canny threshold

    Returns:
        Detected rotation angle in degrees
    """
    h, w = img.shape[:2]

    # Work on small copy for speed
    scale = 800 / max(h, w)
    if scale >= 1:
        scale = 1

    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # Convert float32 to uint8 for edge detection
    if small.dtype == np.float32:
        small = (np.clip(small, 0, 1) * 255).astype(np.uint8)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    # Apply negative mask if provided
    if negative_mask is not None:
        small_mask = cv2.resize(negative_mask, (small.shape[1], small.shape[0]))
        edges = cv2.bitwise_and(edges, small_mask)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=50, maxLineGap=10)

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            angle = 90
        else:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize to deviation from horizontal or vertical
        if -45 <= angle <= 45:
            angles.append(angle)
        elif 45 < angle <= 135:
            angles.append(angle - 90)
        elif -135 <= angle < -45:
            angles.append(angle + 90)

    return np.median(angles) if angles else 0.0


def extract_and_straighten(img: np.ndarray, negative_mask: np.ndarray = None,
                           rotation: float = 0.0, margin: float = 0.05,
                           crop_adjustment: dict = None) -> np.ndarray:
    """
    Straighten and crop the film frame.

    Args:
        img: BGR image as float32 (0-1) or uint8 (0-255)
        negative_mask: Optional mask to constrain detection
        rotation: Total rotation angle in degrees
        margin: Fractional margin to crop inward (0.05 = 5%)
        crop_adjustment: Optional dict with 'left', 'right', 'top', 'bottom' adjustments

    Returns:
        Extracted and straightened image
    """
    h, w = img.shape[:2]

    if crop_adjustment is None:
        crop_adjustment = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}

    # Straighten the full image
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # Find frame bounds on straightened image
    # Work on small copy for speed
    scale = 800 / max(h, w)
    if scale >= 1:
        scale = 1

    small = cv2.resize(rotated, (0, 0), fx=scale, fy=scale)
    if small.dtype == np.float32:
        small_8bit = (np.clip(small, 0, 1) * 255).astype(np.uint8)
    else:
        small_8bit = small

    gray = cv2.cvtColor(small_8bit, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return rotated

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < (small.shape[0] * small.shape[1] * 0.1):
        return rotated

    # Get bounding rect and scale back to full resolution
    x_s, y_s, w_s, h_s = cv2.boundingRect(c)
    x = int(x_s / scale)
    y = int(y_s / scale)
    bw = int(w_s / scale)
    bh = int(h_s / scale)

    # Apply margin
    margin_x = int(bw * margin)
    margin_y = int(bh * margin)
    x += margin_x
    y += margin_y
    bw -= 2 * margin_x
    bh -= 2 * margin_y

    # Apply crop adjustments
    x += crop_adjustment['left']
    y += crop_adjustment['top']
    bw -= crop_adjustment['left'] + crop_adjustment['right']
    bh -= crop_adjustment['top'] + crop_adjustment['bottom']

    # Ensure valid bounds
    x = max(0, x)
    y = max(0, y)
    bw = min(bw, rotated.shape[1] - x)
    bh = min(bh, rotated.shape[0] - y)

    if bw <= 0 or bh <= 0:
        return rotated

    return rotated[y:y+bh, x:x+bw]


def process_negative_image(img: np.ndarray, bg_threshold: int = 15,
                           sensitivity: int = 50, fine_rotation: float = 0.0,
                           crop_adjustment: dict = None, base_pos: tuple = None) -> np.ndarray:
    """
    Full processing pipeline for a film negative image.

    1. Isolate negative from scanner background
    2. Detect rotation angle
    3. Straighten and crop
    4. Sample base color and invert

    Args:
        img: BGR image as float32 (0-1 range)
        bg_threshold: Background isolation threshold (0-255)
        sensitivity: Detection sensitivity (1-100)
        fine_rotation: Additional manual rotation adjustment
        crop_adjustment: Optional dict with crop adjustments
        base_pos: Optional (x, y) tuple (0-1 range) for base color sampling position

    Returns:
        Processed (inverted, cropped, straightened) image as float32 (0-1)
    """
    try:
        # 1. Isolate negative from background
        mask = isolate_negative_from_background(img, bg_threshold)

        # 2. Map sensitivity to detection parameters
        canny_low = max(20, int(150 - sensitivity * 1.2))
        canny_high = max(50, int(300 - sensitivity * 2.5))
        margin = 0.01 + (sensitivity / 100) * 0.09

        # 3. Detect rotation
        angle = detect_frame_angle(img, mask, canny_low, canny_high)
        total_rotation = angle + fine_rotation

        # 4. Extract and straighten
        extracted = extract_and_straighten(img, mask, total_rotation, margin, crop_adjustment)

        # 5. Sample base color
        if base_pos is not None:
            # Sample at specific position
            h, w = img.shape[:2]
            px = int(base_pos[0] * w)
            py = int(base_pos[1] * h)
            # Sample a small region around the position
            region_size = 10
            x1 = max(0, px - region_size)
            x2 = min(w, px + region_size)
            y1 = max(0, py - region_size)
            y2 = min(h, py + region_size)
            region = img[y1:y2, x1:x2]
            base_color_bgr = np.median(region.reshape(-1, 3), axis=0)
        else:
            # Auto-detect from border
            base_color_lab = sample_base_color(img, mask)
            # Convert LAB to BGR
            base_bgr_img = np.zeros((1, 1, 3), dtype=np.uint8)
            base_bgr_img[0, 0] = base_color_lab
            base_color_bgr = cv2.cvtColor(base_bgr_img, cv2.COLOR_LAB2BGR)[0, 0]
            base_color_bgr = base_color_bgr.astype(np.float32) / 255.0

        # 6. Invert
        inverted = invert_negative(extracted, base_color_bgr)

        return inverted

    except Exception:
        # Fallback: simple inversion
        return 1.0 - img if img.dtype == np.float32 else 255 - img


def create_content_mask(img: np.ndarray, base_color_lab: np.ndarray,
                         negative_mask: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """
    Create a mask of pixels that differ significantly from the film base color.

    Uses LAB color space for perceptually uniform distance calculation.

    Args:
        img: BGR image as float32 (0-1) or uint8 (0-255)
        base_color_lab: Base color in LAB as uint8 array
    """
    # Convert image to LAB
    # OpenCV LAB conversion requires uint8 input
    if img.dtype == np.float32:
        img_8bit = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Calculate color distance from base for each pixel
    # ΔE = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)
    diff = lab.astype(np.float32) - base_color_lab.astype(np.float32)
    delta_e = np.sqrt(np.sum(diff ** 2, axis=2))

    # Threshold to get content mask
    content_mask = (delta_e > threshold).astype(np.uint8) * 255

    # Only consider pixels within the negative area
    content_mask = cv2.bitwise_and(content_mask, negative_mask)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel)

    return content_mask


# =============================================================================
# PRECISION CORNER/EDGE DETECTION
# =============================================================================

def detect_edges_canny(img: np.ndarray, negative_mask: np.ndarray) -> np.ndarray:
    """
    Detect edges using Canny within the negative region.

    Args:
        img: BGR image as float32 (0-1) or uint8 (0-255)
    """
    # Canny requires uint8 input
    if img.dtype == np.float32:
        img_8bit = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny edge detection with automatic thresholds based on median
    median_val = np.median(blurred[negative_mask > 0])
    lower = int(max(0, 0.5 * median_val))
    upper = int(min(255, 1.5 * median_val))

    edges = cv2.Canny(blurred, lower, upper)

    # Mask to negative region only
    edges = cv2.bitwise_and(edges, negative_mask)

    return edges


def detect_hough_lines(edges: np.ndarray, min_line_length: int = 50) -> list:
    """
    Detect lines using probabilistic Hough transform.
    Returns list of (x1, y1, x2, y2, angle, length) tuples.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=10
    )

    if lines is None:
        return []

    result = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        result.append((x1, y1, x2, y2, angle, length))

    return result


def cluster_lines_by_orientation(lines: list) -> dict:
    """
    Cluster lines into 4 groups: top, bottom, left, right edges.
    Returns dict with 'horizontal' and 'vertical' line groups.
    """
    horizontal = []  # roughly 0° or 180°
    vertical = []    # roughly 90° or -90°

    for x1, y1, x2, y2, angle, length in lines:
        # Normalize angle to 0-180 range
        norm_angle = angle % 180

        if norm_angle < 30 or norm_angle > 150:  # horizontal
            mid_y = (y1 + y2) / 2
            horizontal.append((x1, y1, x2, y2, angle, length, mid_y))
        elif 60 < norm_angle < 120:  # vertical
            mid_x = (x1 + x2) / 2
            vertical.append((x1, y1, x2, y2, angle, length, mid_x))

    return {'horizontal': horizontal, 'vertical': vertical}


def ransac_fit_line(points: np.ndarray, n_iterations: int = 100, threshold: float = 2.0) -> tuple:
    """
    RANSAC line fitting for robust estimation.
    Returns (vx, vy, x0, y0) - direction vector and point on line.
    """
    if len(points) < 2:
        return None

    best_inliers = 0
    best_line = None

    for _ in range(n_iterations):
        # Random sample of 2 points
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]

        # Line direction
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction = direction / length

        # Count inliers (points within threshold distance of line)
        # Distance from point to line: |cross(direction, p - p1)|
        diff = points - p1
        distances = np.abs(diff[:, 0] * direction[1] - diff[:, 1] * direction[0])
        inliers = np.sum(distances < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            inlier_mask = distances < threshold
            # Refit using all inliers
            inlier_points = points[inlier_mask]
            if len(inlier_points) >= 2:
                line = cv2.fitLine(inlier_points.reshape(-1, 1, 2).astype(np.float32),
                                   cv2.DIST_L2, 0, 0.01, 0.01)
                best_line = (line[0][0], line[1][0], line[2][0], line[3][0])

    return best_line


def fit_edge_lines(lines: list, img_shape: tuple) -> dict:
    """
    Fit precise lines to each edge group using RANSAC.
    Returns dict with fitted lines for top, bottom, left, right.
    """
    clustered = cluster_lines_by_orientation(lines)
    h, w = img_shape[:2]
    center_y, center_x = h / 2, w / 2

    # Split horizontal into top/bottom based on y position
    top_lines = [l for l in clustered['horizontal'] if l[6] < center_y]
    bottom_lines = [l for l in clustered['horizontal'] if l[6] >= center_y]

    # Split vertical into left/right based on x position
    left_lines = [l for l in clustered['vertical'] if l[6] < center_x]
    right_lines = [l for l in clustered['vertical'] if l[6] >= center_x]

    fitted = {}

    for name, line_group in [('top', top_lines), ('bottom', bottom_lines),
                              ('left', left_lines), ('right', right_lines)]:
        if not line_group:
            fitted[name] = None
            continue

        # Collect all endpoints as points for fitting
        points = []
        for x1, y1, x2, y2, *_ in line_group:
            points.extend([(x1, y1), (x2, y2)])
        points = np.array(points, dtype=np.float32)

        fitted[name] = ransac_fit_line(points)

    return fitted


def line_intersection(line1: tuple, line2: tuple) -> tuple:
    """
    Compute intersection of two lines.
    Each line is (vx, vy, x0, y0) - direction vector and point on line.
    Returns (x, y) intersection point or None if parallel.
    """
    if line1 is None or line2 is None:
        return None

    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2

    # Solve: p1 + t*v1 = p2 + s*v2
    # t*vx1 - s*vx2 = x2 - x1
    # t*vy1 - s*vy2 = y2 - y1

    det = vx1 * (-vy2) - vy1 * (-vx2)
    if abs(det) < 1e-10:
        return None  # Parallel lines

    dx, dy = x2 - x1, y2 - y1
    t = (dx * (-vy2) - dy * (-vx2)) / det

    x = x1 + t * vx1
    y = y1 + t * vy1

    return (float(x), float(y))


def compute_corners_from_lines(fitted_lines: dict) -> list:
    """
    Compute 4 corners from fitted edge lines.
    Returns list of 4 corners: [top-left, top-right, bottom-right, bottom-left]
    """
    corners = []

    # Top-left: intersection of top and left
    tl = line_intersection(fitted_lines.get('top'), fitted_lines.get('left'))
    # Top-right: intersection of top and right
    tr = line_intersection(fitted_lines.get('top'), fitted_lines.get('right'))
    # Bottom-right: intersection of bottom and right
    br = line_intersection(fitted_lines.get('bottom'), fitted_lines.get('right'))
    # Bottom-left: intersection of bottom and left
    bl = line_intersection(fitted_lines.get('bottom'), fitted_lines.get('left'))

    return [tl, tr, br, bl]


def detect_shi_tomasi_corners(mask: np.ndarray, max_corners: int = 20) -> np.ndarray:
    """
    Detect corners using Shi-Tomasi (Good Features to Track).
    """
    corners = cv2.goodFeaturesToTrack(
        mask,
        maxCorners=max_corners,
        qualityLevel=0.1,
        minDistance=30,
        blockSize=7
    )

    if corners is None:
        return np.array([])

    return corners.reshape(-1, 2)


def refine_corners_subpixel(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Refine corner positions to sub-pixel accuracy.
    """
    if len(corners) == 0:
        return corners

    corners_input = corners.reshape(-1, 1, 2).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    refined = cv2.cornerSubPix(gray, corners_input, (5, 5), (-1, -1), criteria)

    return refined.reshape(-1, 2)


def find_closest_corners(line_corners: list, detected_corners: np.ndarray,
                         max_distance: float = 50.0) -> list:
    """
    Match line intersection corners with detected corners for validation.
    Returns refined corners using detected corners where available.
    """
    result = []

    for lc in line_corners:
        if lc is None:
            result.append(None)
            continue

        if len(detected_corners) == 0:
            result.append(lc)
            continue

        # Find closest detected corner
        distances = np.sqrt(np.sum((detected_corners - np.array(lc))**2, axis=1))
        min_idx = np.argmin(distances)

        if distances[min_idx] < max_distance:
            # Use detected corner (more accurate)
            result.append(tuple(detected_corners[min_idx]))
        else:
            # Keep line intersection corner
            result.append(lc)

    return result


def optimize_rectangle(corners: list, aspect_ratio: float = 1.5) -> np.ndarray:
    """
    Optimize corner positions to form a perfect rectangle.
    Uses least squares to find best-fit rectangle constrained to valid corners.

    aspect_ratio: expected width/height ratio (35mm film = 36/24 = 1.5)
    """
    # Filter out None corners
    valid_corners = [(i, c) for i, c in enumerate(corners) if c is not None]

    if len(valid_corners) < 3:
        return None

    # Initial estimate: center and size from valid corners
    points = np.array([c for _, c in valid_corners])
    center = np.mean(points, axis=0)

    # Estimate size from point spread
    width = np.max(points[:, 0]) - np.min(points[:, 0])
    height = np.max(points[:, 1]) - np.min(points[:, 1])

    # Initial parameters: [center_x, center_y, width, height, angle]
    init_params = [center[0], center[1], width, height, 0.0]

    def rectangle_corners(params):
        """Generate 4 corners from rectangle parameters."""
        cx, cy, w, h, angle = params
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Half dimensions
        hw, hh = w / 2, h / 2

        # Corner offsets (before rotation)
        offsets = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])

        # Rotate and translate
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = offsets @ rotation.T
        corners = rotated + np.array([cx, cy])

        return corners

    def cost_function(params):
        """Cost: sum of squared distances from valid corners to nearest rectangle corner."""
        rect_corners = rectangle_corners(params)
        total_cost = 0.0

        for idx, corner in valid_corners:
            # Distance to the corresponding rectangle corner
            dist = np.sum((rect_corners[idx] - np.array(corner))**2)
            total_cost += dist

        return total_cost

    # Optimize
    result = optimize.minimize(cost_function, init_params, method='Powell')

    if result.success:
        return rectangle_corners(result.x)
    else:
        return rectangle_corners(init_params)


def find_frame_corners_precision(img: np.ndarray, content_mask: np.ndarray,
                                  negative_mask: np.ndarray) -> np.ndarray:
    """
    Find frame corners using precision edge/corner detection.

    Args:
        img: BGR image as float32 (0-1) or uint8 (0-255)

    Returns: 4x2 array of corner coordinates [TL, TR, BR, BL]
    """
    # Corner refinement requires uint8
    if img.dtype == np.float32:
        img_8bit = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Get edges from the CONTENT MASK boundary (not the image)
    # This focuses on the frame boundary, not internal image details
    edges = cv2.Canny(content_mask, 50, 150)

    # Step 2: Hough line detection
    lines = detect_hough_lines(edges, min_line_length=50)

    # Step 3: Fit lines to each edge using RANSAC
    fitted_lines = fit_edge_lines(lines, img.shape)

    # Step 4: Compute corners from line intersections
    line_corners = compute_corners_from_lines(fitted_lines)

    # Step 5: Detect Shi-Tomasi corners on content mask
    detected_corners = detect_shi_tomasi_corners(content_mask)

    # Step 6: Refine detected corners to sub-pixel
    if len(detected_corners) > 0:
        detected_corners = refine_corners_subpixel(gray, detected_corners)

    # Step 7: Match and merge corners
    merged_corners = find_closest_corners(line_corners, detected_corners)

    # Step 8: Optimize to rectangle constraint
    optimized = optimize_rectangle(merged_corners)

    if optimized is not None:
        return optimized

    # Fallback: return line corners as array
    valid = [c for c in merged_corners if c is not None]
    if len(valid) >= 3:
        return np.array(valid)

    return None


def corners_to_rect(corners: np.ndarray) -> tuple:
    """
    Convert 4 corners to OpenCV RotatedRect format.
    Returns: ((center_x, center_y), (width, height), angle)
    """
    if corners is None or len(corners) < 4:
        return None

    # Use minAreaRect on the corners
    corners_int = corners.astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(corners_int)

    return rect


def find_frame_rectangle(content_mask: np.ndarray, negative_mask: np.ndarray,
                         img: np.ndarray = None, use_precision: bool = True) -> tuple:
    """
    Find the rotated rectangle that best fits the frame content.

    Returns: ((center_x, center_y), (width, height), angle)
    """
    # Try precision detection first
    if use_precision and img is not None:
        try:
            corners = find_frame_corners_precision(img, content_mask, negative_mask)
            if corners is not None and len(corners) >= 4:
                rect = corners_to_rect(corners)
                if rect is not None:
                    return rect
        except Exception as e:
            print(f"    Precision detection failed: {e}, falling back to contour method")

    # Fallback: contour-based detection
    contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in content mask")

    # Get the largest contour (should be the frame)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit minimum area rotated rectangle
    rect = cv2.minAreaRect(largest_contour)

    return rect


def extract_frame(img: np.ndarray, rect: tuple, padding: int = 0) -> np.ndarray:
    """
    Extract and deskew the frame from the image using the detected rectangle.
    """
    center, size, angle = rect
    width, height = size

    # Add padding
    width += padding * 2
    height += padding * 2

    # Ensure width > height (landscape orientation for consistency)
    if width < height:
        width, height = height, width
        angle += 90

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the entire image
    rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

    # Crop the rectangle
    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)

    # Clamp to image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(int(width), rotated.shape[1] - x)
    h = min(int(height), rotated.shape[0] - y)

    return rotated[y:y+h, x:x+w]


def draw_detection_visualization(img: np.ndarray, negative_mask: np.ndarray,
                                  content_mask: np.ndarray, rect: tuple) -> np.ndarray:
    """
    Create a visualization showing the detection steps.
    """
    vis = img.copy()

    # Draw the detected rectangle in green
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(vis, [box], 0, (0, 255, 0), 3)

    # Draw corner points
    for point in box:
        cv2.circle(vis, tuple(point), 8, (0, 0, 255), -1)

    return vis


def process_negative(input_path: str, output_dir: str = None,
                     base_threshold: float = 20.0, debug: bool = True) -> dict:
    """
    Main processing pipeline for a single negative.

    Returns dict with detected rectangle and extracted frame.
    """
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Processing: {input_path.name}")

    # Load image
    img = load_image(str(input_path))
    print(f"  Image size: {img.shape[1]}x{img.shape[0]}")

    # Step 1: Isolate negative from black background
    print("  Step 1: Isolating negative from background...")
    negative_mask = isolate_negative_from_background(img)

    # Step 2: Sample base color from border
    print("  Step 2: Sampling film base color...")
    base_color_lab = sample_base_color(img, negative_mask)
    # Convert back to BGR for display
    base_color_bgr = cv2.cvtColor(np.array([[base_color_lab]], dtype=np.uint8), cv2.COLOR_LAB2BGR)[0, 0]
    print(f"  Base color (BGR): {base_color_bgr}")

    # Step 3: Create content mask
    print(f"  Step 3: Creating content mask (threshold={base_threshold})...")
    content_mask = create_content_mask(img, base_color_lab, negative_mask, threshold=base_threshold)

    # Step 4: Find frame rectangle
    print("  Step 4: Finding frame rectangle...")
    rect = find_frame_rectangle(content_mask, negative_mask, img)
    center, size, angle = rect
    print(f"  Detected rectangle: center={center}, size={size}, angle={angle:.2f}°")

    # Step 5: Extract frame
    print("  Step 5: Extracting frame...")
    extracted = extract_frame(img, rect)
    print(f"  Extracted size: {extracted.shape[1]}x{extracted.shape[0]}")

    # Save outputs
    stem = input_path.stem

    if debug:
        # Save visualization
        vis = draw_detection_visualization(img, negative_mask, content_mask, rect)
        cv2.imwrite(str(output_dir / f"{stem}_detection.png"), vis)

        # Save masks for debugging
        cv2.imwrite(str(output_dir / f"{stem}_mask_negative.png"), negative_mask)
        cv2.imwrite(str(output_dir / f"{stem}_mask_content.png"), content_mask)

    # Save extracted frame
    cv2.imwrite(str(output_dir / f"{stem}_extracted.png"), extracted)

    print(f"  Outputs saved to: {output_dir}")

    return {
        "rect": rect,
        "extracted": extracted,
        "base_color_bgr": base_color_bgr,
    }
