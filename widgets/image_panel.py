"""
SUPER NEGATIVE PROCESSING SYSTEM - Image Panel Widgets

Main image display with zoom/pan and crop functionality.
"""

from PySide6.QtWidgets import QWidget, QPushButton, QSizePolicy
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
import numpy as np
import cv2


class CropWidget(QWidget):
    """Widget for interactive cropping with draggable edges/corners."""

    cropChanged = Signal(tuple)  # Emits (left, top, right, bottom) as percentages

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._image = None
        self._pixmap = None
        self._image_rect = None  # Where the image is drawn

        # Crop region as percentages (0-1) of image
        self._crop = [0.0, 0.0, 1.0, 1.0]  # left, top, right, bottom

        # Drag state
        self._dragging = None  # 'left', 'right', 'top', 'bottom', 'tl', 'tr', 'bl', 'br'
        self._drag_start = None
        self._crop_start = None

        self._handle_size = 8

        # Grid overlay state
        self._grid_enabled = False
        self._grid_divisions = 3

    def set_grid_enabled(self, enabled: bool):
        """Enable or disable grid overlay."""
        self._grid_enabled = enabled
        self.update()

    def set_grid_divisions(self, divisions: int):
        """Set number of grid divisions."""
        self._grid_divisions = max(2, min(20, divisions))
        self.update()

    def set_image(self, img: np.ndarray, reset_crop: bool = False):
        """Set the image to display. Preserves crop unless reset_crop=True or image is None."""
        if img is None:
            self._image = None
            self._pixmap = None
            self._crop = [0.0, 0.0, 1.0, 1.0]
            self.update()
            return

        self._image = img.copy()

        # Only reset crop if explicitly requested
        if reset_crop:
            self._crop = [0.0, 0.0, 1.0, 1.0]

        # Convert float32 (0-1) to uint8 (0-255) for display
        display_img = img
        if img.dtype == np.float32:
            display_img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Convert to QPixmap
        if len(display_img.shape) == 2:
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def get_crop_rect(self) -> tuple:
        """Get crop as (x, y, w, h) in pixels of original image."""
        if self._image is None:
            return None
        h, w = self._image.shape[:2]
        x1 = int(self._crop[0] * w)
        y1 = int(self._crop[1] * h)
        x2 = int(self._crop[2] * w)
        y2 = int(self._crop[3] * h)
        return (x1, y1, x2 - x1, y2 - y1)

    def get_cropped_image(self) -> np.ndarray:
        """Get the cropped portion of the image."""
        if self._image is None:
            return None
        rect = self.get_crop_rect()
        if rect is None:
            return None
        x, y, w, h = rect
        return self._image[y:y+h, x:x+w]

    def set_crop(self, crop: tuple):
        """Set the crop region (left, top, right, bottom as percentages 0-1)."""
        if crop and len(crop) == 4:
            self._crop = list(crop)
            self.update()

    def get_crop(self) -> tuple:
        """Get current crop as tuple."""
        return tuple(self._crop)

    def reset_crop(self):
        """Reset crop to full image."""
        self._crop = [0.0, 0.0, 1.0, 1.0]
        self.update()
        self.cropChanged.emit(tuple(self._crop))

    def _get_image_rect(self) -> tuple:
        """Calculate where the image is drawn (scaled to fit)."""
        if self._pixmap is None:
            return None

        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()

        scale = min(ww / pw, wh / ph)
        scaled_w = int(pw * scale)
        scaled_h = int(ph * scale)

        x = (ww - scaled_w) // 2
        y = (wh - scaled_h) // 2

        return (x, y, scaled_w, scaled_h)

    def _crop_to_widget(self, crop_x, crop_y) -> tuple:
        """Convert crop percentage to widget coordinates."""
        if self._image_rect is None:
            return (0, 0)
        ix, iy, iw, ih = self._image_rect
        return (ix + crop_x * iw, iy + crop_y * ih)

    def _widget_to_crop(self, wx, wy) -> tuple:
        """Convert widget coordinates to crop percentage."""
        if self._image_rect is None:
            return (0, 0)
        ix, iy, iw, ih = self._image_rect
        cx = max(0, min(1, (wx - ix) / iw))
        cy = max(0, min(1, (wy - iy) / ih))
        return (cx, cy)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(42, 42, 42))

        if self._pixmap is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "Extracted Frame")
            return

        # Draw scaled image
        self._image_rect = self._get_image_rect()
        ix, iy, iw, ih = self._image_rect
        scaled = self._pixmap.scaled(iw, ih, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(ix, iy, scaled)

        # Draw darkened areas outside crop
        l, t, r, b = self._crop
        cl, ct = self._crop_to_widget(l, t)
        cr, cb = self._crop_to_widget(r, b)

        overlay = QColor(0, 0, 0, 128)
        # Top
        painter.fillRect(int(ix), int(iy), int(iw), int(ct - iy), overlay)
        # Bottom
        painter.fillRect(int(ix), int(cb), int(iw), int(iy + ih - cb), overlay)
        # Left
        painter.fillRect(int(ix), int(ct), int(cl - ix), int(cb - ct), overlay)
        # Right
        painter.fillRect(int(cr), int(ct), int(ix + iw - cr), int(cb - ct), overlay)

        # Draw crop rectangle
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.drawRect(int(cl), int(ct), int(cr - cl), int(cb - ct))

        # Draw grid overlay within crop area
        if self._grid_enabled:
            self._draw_grid(painter, cl, ct, cr, cb)

        # Draw handles
        self._draw_handles(painter, cl, ct, cr, cb)

    def _draw_handles(self, painter, cl, ct, cr, cb):
        """Draw drag handles on edges and corners."""
        hs = self._handle_size
        painter.setBrush(QColor(255, 0, 0))
        painter.setPen(QPen(QColor(0, 0, 0), 1))

        mx, my = (cl + cr) / 2, (ct + cb) / 2

        # Edge handles
        handles = [
            (cl - hs/2, my - hs/2, hs, hs),  # left
            (cr - hs/2, my - hs/2, hs, hs),  # right
            (mx - hs/2, ct - hs/2, hs, hs),  # top
            (mx - hs/2, cb - hs/2, hs, hs),  # bottom
        ]
        # Corner handles
        handles += [
            (cl - hs/2, ct - hs/2, hs, hs),  # tl
            (cr - hs/2, ct - hs/2, hs, hs),  # tr
            (cl - hs/2, cb - hs/2, hs, hs),  # bl
            (cr - hs/2, cb - hs/2, hs, hs),  # br
        ]

        for hx, hy, hw, hh in handles:
            painter.drawRect(int(hx), int(hy), int(hw), int(hh))

    def _draw_grid(self, painter, cl, ct, cr, cb):
        """Draw grid overlay within the crop area."""
        width = cr - cl
        height = cb - ct

        for i in range(1, self._grid_divisions):
            # Calculate line positions
            x = cl + (width * i / self._grid_divisions)
            y = ct + (height * i / self._grid_divisions)

            # Draw dark outline first (3px black)
            painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
            painter.drawLine(int(x), int(ct), int(x), int(cb))
            painter.drawLine(int(cl), int(y), int(cr), int(y))

            # Draw white center line (1px white)
            painter.setPen(QPen(QColor(255, 255, 255, 230), 1))
            painter.drawLine(int(x), int(ct), int(x), int(cb))
            painter.drawLine(int(cl), int(y), int(cr), int(y))

    def _get_handle_at(self, pos) -> str:
        """Get which handle is at the given position."""
        if self._image_rect is None:
            return None

        l, t, r, b = self._crop
        cl, ct = self._crop_to_widget(l, t)
        cr, cb = self._crop_to_widget(r, b)
        mx, my = (cl + cr) / 2, (ct + cb) / 2

        hs = self._handle_size + 4  # Slightly larger hit area
        x, y = pos.x(), pos.y()

        # Check corners first (they overlap edges)
        if abs(x - cl) < hs and abs(y - ct) < hs:
            return 'tl'
        if abs(x - cr) < hs and abs(y - ct) < hs:
            return 'tr'
        if abs(x - cl) < hs and abs(y - cb) < hs:
            return 'bl'
        if abs(x - cr) < hs and abs(y - cb) < hs:
            return 'br'

        # Check edges
        if abs(x - cl) < hs and ct < y < cb:
            return 'left'
        if abs(x - cr) < hs and ct < y < cb:
            return 'right'
        if abs(y - ct) < hs and cl < x < cr:
            return 'top'
        if abs(y - cb) < hs and cl < x < cr:
            return 'bottom'

        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            handle = self._get_handle_at(pos)
            if handle:
                self._dragging = handle
                self._drag_start = pos
                self._crop_start = list(self._crop)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._dragging:
            self._update_crop(pos)
        else:
            # Update cursor based on handle
            handle = self._get_handle_at(pos)
            if handle in ('left', 'right'):
                self.setCursor(Qt.SizeHorCursor)
            elif handle in ('top', 'bottom'):
                self.setCursor(Qt.SizeVerCursor)
            elif handle in ('tl', 'br'):
                self.setCursor(Qt.SizeFDiagCursor)
            elif handle in ('tr', 'bl'):
                self.setCursor(Qt.SizeBDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = None
            self.cropChanged.emit(tuple(self._crop))

    def _update_crop(self, pos):
        """Update crop based on drag."""
        cx, cy = self._widget_to_crop(pos.x(), pos.y())
        l, t, r, b = self._crop_start

        min_size = 0.05  # Minimum 5% of image

        if self._dragging == 'left':
            self._crop[0] = min(cx, r - min_size)
        elif self._dragging == 'right':
            self._crop[2] = max(cx, l + min_size)
        elif self._dragging == 'top':
            self._crop[1] = min(cy, b - min_size)
        elif self._dragging == 'bottom':
            self._crop[3] = max(cy, t + min_size)
        elif self._dragging in ('tl', 'tr', 'bl', 'br'):
            # Corner drag - maintain aspect ratio
            orig_w = r - l
            orig_h = b - t
            aspect = orig_w / orig_h if orig_h > 0 else 1

            if self._dragging == 'br':
                new_w = max(min_size, cx - l)
                new_h = new_w / aspect
                self._crop[2] = l + new_w
                self._crop[3] = min(1.0, t + new_h)
            elif self._dragging == 'tl':
                new_w = max(min_size, r - cx)
                new_h = new_w / aspect
                self._crop[0] = r - new_w
                self._crop[1] = max(0.0, b - new_h)
            elif self._dragging == 'tr':
                new_w = max(min_size, cx - l)
                new_h = new_w / aspect
                self._crop[2] = l + new_w
                self._crop[1] = max(0.0, b - new_h)
            elif self._dragging == 'bl':
                new_w = max(min_size, r - cx)
                new_h = new_w / aspect
                self._crop[0] = r - new_w
                self._crop[3] = min(1.0, t + new_h)

        # Clamp values
        self._crop[0] = max(0, self._crop[0])
        self._crop[1] = max(0, self._crop[1])
        self._crop[2] = min(1, self._crop[2])
        self._crop[3] = min(1, self._crop[3])

        self.update()


class BaseSelectionWidget(QWidget):
    """Widget for selecting base color with a movable crosshair."""

    positionChanged = Signal(tuple)  # Emits (x, y) as percentages (0-1)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._image = None
        self._pixmap = None
        self._image_rect = None

        # Position as percentages (0-1) of image
        self._pos = [0.5, 0.5]  # x, y - default to center
        self._dragging = False
        self._crosshair_size = 20
        self._sampled_color_bgr = None

        # Debug: sample boxes from auto-detect (list of (x%, y%, radius%, variance, is_best))
        self._debug_sample_boxes = []

    def set_image(self, img: np.ndarray):
        """Set the image to display."""
        if img is None:
            self._image = None
            self._pixmap = None
            self.update()
            return

        self._image = img.copy()
        # Reset position to center when new image is set
        self._pos = [0.5, 0.5]

        # Convert float32 (0-1) to uint8 (0-255) for display
        display_img = img
        if img.dtype == np.float32:
            display_img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Convert to QPixmap
        if len(display_img.shape) == 2:
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._sample_color()
        self.update()

    def set_position(self, pos: tuple):
        """Set crosshair position (x, y as percentages 0-1)."""
        if pos and len(pos) == 2:
            self._pos = list(pos)
            self._sample_color()
            self.update()

    def get_position(self) -> tuple:
        """Get current position as tuple."""
        return tuple(self._pos)

    def get_sampled_color_bgr(self) -> np.ndarray:
        """Get the BGR color at the crosshair position."""
        return self._sampled_color_bgr

    def auto_detect_base(self, negative_mask: np.ndarray, brightness_threshold: int = 80):
        """Auto-detect base color position from the film border region."""
        if self._image is None or negative_mask is None:
            return

        h, w = self._image.shape[:2]

        # Scale brightness threshold for float32 images (0-1 range)
        if self._image.dtype == np.float32:
            brightness_threshold = brightness_threshold / 255.0

        # 1. Find border region via double erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        outer_mask = cv2.erode(negative_mask, kernel, iterations=3)
        inner_mask = cv2.erode(negative_mask, kernel, iterations=15)
        border_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

        # 2. Get all candidate points within the border region
        border_points = np.where(border_mask > 0)
        if len(border_points[0]) == 0:
            self.update()
            return

        # 3. Sample ~30 random points from the border region
        num_candidates = min(30, len(border_points[0]))
        indices = np.linspace(0, len(border_points[0]) - 1, num_candidates, dtype=int)

        best_pos = None
        best_variance = float('inf')
        best_idx = -1
        sample_radius = 15

        self._debug_sample_boxes = []
        sample_boxes_temp = []

        for i, idx in enumerate(indices):
            py, px = border_points[0][idx], border_points[1][idx]

            x1, y1 = max(0, px - sample_radius), max(0, py - sample_radius)
            x2, y2 = min(w, px + sample_radius), min(h, py + sample_radius)
            patch = self._image[y1:y2, x1:x2]

            if patch.size == 0:
                continue

            mean_brightness = np.mean(patch)
            if mean_brightness < brightness_threshold:
                sample_boxes_temp.append((px / w, py / h, sample_radius / w, sample_radius / h, float('inf')))
                continue

            variance = np.var(patch)
            sample_boxes_temp.append((px / w, py / h, sample_radius / w, sample_radius / h, variance))

            if variance < best_variance:
                best_variance = variance
                best_pos = (px / w, py / h)
                best_idx = len(sample_boxes_temp) - 1

        for i, box in enumerate(sample_boxes_temp):
            x_pct, y_pct, r_w, r_h, var = box
            is_best = (i == best_idx)
            self._debug_sample_boxes.append((x_pct, y_pct, r_w, r_h, var, is_best))

        if best_pos is not None:
            self._pos = list(best_pos)
            self._sample_color()
            self.positionChanged.emit(tuple(self._pos))

        self.update()

    def _sample_color(self):
        """Sample the color at the current crosshair position."""
        if self._image is None:
            self._sampled_color_bgr = None
            return

        h, w = self._image.shape[:2]
        x = int(self._pos[0] * w)
        y = int(self._pos[1] * h)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        x1, x2 = max(0, x - 1), min(w, x + 2)
        y1, y2 = max(0, y - 1), min(h, y + 2)
        region = self._image[y1:y2, x1:x2]

        if self._image.dtype == np.float32:
            self._sampled_color_bgr = np.mean(region, axis=(0, 1)).astype(np.float32)
        else:
            self._sampled_color_bgr = np.mean(region, axis=(0, 1)).astype(np.uint8)

    def _get_image_rect(self) -> tuple:
        """Calculate where the image is drawn (scaled to fit)."""
        if self._pixmap is None:
            return None

        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()

        scale = min(ww / pw, wh / ph)
        scaled_w = int(pw * scale)
        scaled_h = int(ph * scale)

        x = (ww - scaled_w) // 2
        y = (wh - scaled_h) // 2

        return (x, y, scaled_w, scaled_h)

    def _pos_to_widget(self, px, py) -> tuple:
        """Convert position percentage to widget coordinates."""
        if self._image_rect is None:
            return (0, 0)
        ix, iy, iw, ih = self._image_rect
        return (ix + px * iw, iy + py * ih)

    def _widget_to_pos(self, wx, wy) -> tuple:
        """Convert widget coordinates to position percentage."""
        if self._image_rect is None:
            return (0.5, 0.5)
        ix, iy, iw, ih = self._image_rect
        px = max(0, min(1, (wx - ix) / iw))
        py = max(0, min(1, (wy - iy) / ih))
        return (px, py)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(42, 42, 42))

        if self._pixmap is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "Base Selection")
            return

        # Draw scaled image
        self._image_rect = self._get_image_rect()
        ix, iy, iw, ih = self._image_rect
        scaled = self._pixmap.scaled(iw, ih, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(ix, iy, scaled)

        # Draw debug sample boxes from auto-detect
        for box in self._debug_sample_boxes:
            x_pct, y_pct, r_w, r_h, variance, is_best = box
            bx, by = self._pos_to_widget(x_pct, y_pct)
            box_w = r_w * iw * 2
            box_h = r_h * ih * 2

            if is_best:
                painter.setPen(QPen(QColor(0, 255, 0), 2))
            else:
                painter.setPen(QPen(QColor(255, 0, 0, 150), 1))

            painter.setBrush(Qt.NoBrush)
            painter.drawRect(int(bx - box_w/2), int(by - box_h/2), int(box_w), int(box_h))

        # Draw crosshair
        cx, cy = self._pos_to_widget(self._pos[0], self._pos[1])
        cs = self._crosshair_size

        # White outline for visibility
        painter.setPen(QPen(QColor(0, 0, 0), 3))
        painter.drawLine(int(cx - cs), int(cy), int(cx + cs), int(cy))
        painter.drawLine(int(cx), int(cy - cs), int(cx), int(cy + cs))

        # Yellow crosshair
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.drawLine(int(cx - cs), int(cy), int(cx + cs), int(cy))
        painter.drawLine(int(cx), int(cy - cs), int(cx), int(cy + cs))

        # Draw sampled color swatch
        if self._sampled_color_bgr is not None:
            b, g, r = self._sampled_color_bgr
            if self._image is not None and self._image.dtype == np.float32:
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
            else:
                r, g, b = int(r), int(g), int(b)
            painter.setBrush(QColor(r, g, b))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawRect(int(cx + cs + 5), int(cy - 10), 20, 20)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            if self._image_rect:
                ix, iy, iw, ih = self._image_rect
                if ix <= pos.x() <= ix + iw and iy <= pos.y() <= iy + ih:
                    self._dragging = True
                    self._update_position(pos)

    def mouseMoveEvent(self, event):
        if self._dragging:
            pos = event.position().toPoint()
            self._update_position(pos)
        else:
            if self._image_rect:
                pos = event.position().toPoint()
                ix, iy, iw, ih = self._image_rect
                if ix <= pos.x() <= ix + iw and iy <= pos.y() <= iy + ih:
                    self.setCursor(Qt.CrossCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.positionChanged.emit(tuple(self._pos))

    def _update_position(self, widget_pos):
        """Update crosshair position from widget coordinates."""
        px, py = self._widget_to_pos(widget_pos.x(), widget_pos.y())
        self._pos = [px, py]
        self._sample_color()
        self.update()


class ImagePanel(QWidget):
    """A panel that displays an image with a title and optional grid overlay."""

    # Signal emitted when crop edge is dragged: (edge_name, delta_pixels)
    cropEdgeDragged = Signal(str, int)
    # Signal emitted when crop corner drag starts: (corner_name,)
    cropCornerDragStarted = Signal(str)
    # Signal emitted when crop corner is dragged: (corner_name, cumulative_delta_x, cumulative_delta_y)
    cropCornerDragged = Signal(str, int, int)
    # Signal emitted when crop box is moved: (delta_x, delta_y) in image pixels
    cropBoxMoved = Signal(int, int)

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._original_image = None
        self._pixmap = None
        self._image_rect = None  # (x, y, w, h) of drawn image

        # Grid overlay state
        self._grid_enabled = False
        self._grid_divisions = 3

        # Crop mode state
        self._crop_mode_active = False
        self._crop_full_image = None
        self._crop_full_pixmap = None
        self._crop_full_pixmap_inverted = None
        self._crop_bounds = None
        self._crop_adjustment = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
        self._flash_edge = None
        self._crop_inverted = False
        self._border_flash_edge = None

        # Crop edge dragging state
        self._dragging_edge = None
        self._dragging_box = False
        self._drag_start_pos = None
        self._drag_start_value = 0
        self._drag_last_pos = None
        self.setMouseTracking(True)

        # Invert button (shown when in crop mode)
        self._invert_btn = QPushButton("Invert", self)
        self._invert_btn.setCheckable(True)
        self._invert_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 60, 200);
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 80, 220);
            }
            QPushButton:checked {
                background-color: rgba(80, 120, 80, 220);
                border: 1px solid #6a6;
            }
        """)
        self._invert_btn.setToolTip("Toggle inverted preview (V)")
        self._invert_btn.clicked.connect(self._toggle_invert)
        self._invert_btn.hide()

    def set_grid_enabled(self, enabled: bool):
        """Enable or disable grid overlay."""
        self._grid_enabled = enabled
        self.update()

    def set_grid_divisions(self, divisions: int):
        """Set number of grid divisions."""
        self._grid_divisions = max(2, min(20, divisions))
        self.update()

    def get_image(self) -> np.ndarray:
        """Return the currently displayed image, or None if no image."""
        return self._original_image

    def set_crop_mode(self, active: bool, full_image: np.ndarray = None,
                      inverted_image: np.ndarray = None,
                      bounds: tuple = None, adjustment: dict = None,
                      flash_edge: str = None):
        """Set crop mode state."""
        self._crop_mode_active = active
        self._flash_edge = flash_edge
        if full_image is not None:
            self._crop_full_image = full_image.copy()
            display_img = full_image
            if full_image.dtype == np.float32:
                display_img = (np.clip(full_image, 0, 1) * 255).astype(np.uint8)
            if len(display_img.shape) == 2:
                img_rgb = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            bytes_per_line = 3 * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self._crop_full_pixmap = QPixmap.fromImage(qimg)
        if inverted_image is not None:
            display_inv = inverted_image
            if inverted_image.dtype == np.float32:
                display_inv = (np.clip(inverted_image, 0, 1) * 255).astype(np.uint8)
            if len(display_inv.shape) == 2:
                img_inv_rgb = cv2.cvtColor(display_inv, cv2.COLOR_GRAY2RGB)
            else:
                img_inv_rgb = cv2.cvtColor(display_inv, cv2.COLOR_BGR2RGB)
            h, w = img_inv_rgb.shape[:2]
            bytes_per_line = 3 * w
            qimg_inv = QImage(img_inv_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self._crop_full_pixmap_inverted = QPixmap.fromImage(qimg_inv)
        if bounds is not None:
            self._crop_bounds = bounds
        if adjustment is not None:
            self._crop_adjustment = adjustment.copy()
        if active:
            self._invert_btn.show()
            self._position_invert_button()
        else:
            self._invert_btn.hide()
            self._crop_inverted = False
            self._invert_btn.setChecked(False)
        self.update()

    def _toggle_invert(self):
        """Toggle inverted colors in crop preview."""
        self._crop_inverted = self._invert_btn.isChecked()
        self.update()

    def _position_invert_button(self):
        """Position the invert button in the top-left corner."""
        margin = 10
        self._invert_btn.move(margin, margin)

    def set_border_flash(self, edge: str = None):
        """Set which edge to flash on the border (for normal view feedback)."""
        self._border_flash_edge = edge
        self.update()

    def set_image(self, img: np.ndarray, is_mask: bool = False):
        """Set the image to display."""
        if img is None:
            self._original_image = None
            self._pixmap = None
            self.update()
            return

        self._original_image = img.copy()
        self._update_pixmap()
        self.update()

    def _update_pixmap(self):
        """Convert stored image to QPixmap."""
        if self._original_image is None:
            self._pixmap = None
            return

        img = self._original_image

        if img.dtype == np.float32:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)

    def _get_image_rect(self):
        """Calculate where the image should be drawn (centered, aspect-ratio preserved)."""
        if self._pixmap is None:
            return None

        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()

        scale = min(ww / pw, wh / ph)
        iw, ih = int(pw * scale), int(ph * scale)
        ix = (ww - iw) // 2
        iy = (wh - ih) // 2

        return (ix, iy, iw, ih)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(42, 42, 42))

        painter.setPen(QPen(QColor(68, 68, 68), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        if self._crop_mode_active and self._crop_full_pixmap is not None:
            self._draw_crop_mode(painter)
            return

        if self._pixmap is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, self.title)
            return

        self._image_rect = self._get_image_rect()
        ix, iy, iw, ih = self._image_rect
        scaled = self._pixmap.scaled(iw, ih, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(ix, iy, scaled)

        if self._grid_enabled:
            self._draw_grid(painter, ix, iy, iw, ih)

        if self._border_flash_edge:
            flash_pen = QPen(QColor(255, 0, 0), 3)
            flash_pen.setDashPattern([8, 8])
            painter.setPen(flash_pen)
            if self._border_flash_edge == 'left':
                painter.drawLine(ix, iy, ix, iy + ih)
            elif self._border_flash_edge == 'right':
                painter.drawLine(ix + iw, iy, ix + iw, iy + ih)
            elif self._border_flash_edge == 'top':
                painter.drawLine(ix, iy, ix + iw, iy)
            elif self._border_flash_edge == 'bottom':
                painter.drawLine(ix, iy + ih, ix + iw, iy + ih)

    def _draw_crop_mode(self, painter):
        """Draw crop mode visualization with full image and crop overlay."""
        pixmap = self._crop_full_pixmap_inverted if self._crop_inverted else self._crop_full_pixmap
        pw, ph = pixmap.width(), pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        iw, ih = int(pw * scale), int(ph * scale)
        ix = (ww - iw) // 2
        iy = (wh - ih) // 2

        scaled = pixmap.scaled(iw, ih, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(ix, iy, scaled)

        if self._crop_bounds is not None:
            cx, cy, cw, ch = self._crop_bounds
            adj = self._crop_adjustment

            ax = cx - adj['left']
            ay = cy - adj['top']
            aw = cw + adj['left'] + adj['right']
            ah = ch + adj['top'] + adj['bottom']

            crop_x = ix + int(ax * scale)
            crop_y = iy + int(ay * scale)
            crop_w = int(aw * scale)
            crop_h = int(ah * scale)

            overlay = QColor(0, 0, 0, 160)
            painter.fillRect(ix, iy, iw, crop_y - iy, overlay)
            painter.fillRect(ix, crop_y + crop_h, iw, iy + ih - crop_y - crop_h, overlay)
            painter.fillRect(ix, crop_y, crop_x - ix, crop_h, overlay)
            painter.fillRect(crop_x + crop_w, crop_y, ix + iw - crop_x - crop_w, crop_h, overlay)

            normal_color = QColor(0, 255, 0)
            flash_color = QColor(255, 0, 0)

            def make_crop_pen(color):
                pen = QPen(color, 1)
                pen.setDashPattern([8, 8])
                return pen

            is_flash = self._flash_edge == 'left'
            painter.setPen(make_crop_pen(flash_color if is_flash else normal_color))
            painter.drawLine(crop_x, crop_y, crop_x, crop_y + crop_h)

            is_flash = self._flash_edge == 'right'
            painter.setPen(make_crop_pen(flash_color if is_flash else normal_color))
            painter.drawLine(crop_x + crop_w, crop_y, crop_x + crop_w, crop_y + crop_h)

            is_flash = self._flash_edge == 'top'
            painter.setPen(make_crop_pen(flash_color if is_flash else normal_color))
            painter.drawLine(crop_x, crop_y, crop_x + crop_w, crop_y)

            is_flash = self._flash_edge == 'bottom'
            painter.setPen(make_crop_pen(flash_color if is_flash else normal_color))
            painter.drawLine(crop_x, crop_y + crop_h, crop_x + crop_w, crop_y + crop_h)

            handle_size = 10
            painter.setPen(QPen(QColor(0, 0, 0), 1))

            handle_color = flash_color if self._flash_edge == 'left' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x - handle_size // 2, crop_y + crop_h // 2 - handle_size // 2,
                           handle_size, handle_size)
            handle_color = flash_color if self._flash_edge == 'right' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x + crop_w - handle_size // 2, crop_y + crop_h // 2 - handle_size // 2,
                           handle_size, handle_size)
            handle_color = flash_color if self._flash_edge == 'top' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x + crop_w // 2 - handle_size // 2, crop_y - handle_size // 2,
                           handle_size, handle_size)
            handle_color = flash_color if self._flash_edge == 'bottom' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x + crop_w // 2 - handle_size // 2, crop_y + crop_h - handle_size // 2,
                           handle_size, handle_size)

            painter.setBrush(normal_color)
            painter.drawRect(crop_x - handle_size // 2, crop_y - handle_size // 2,
                           handle_size, handle_size)
            painter.drawRect(crop_x + crop_w - handle_size // 2, crop_y - handle_size // 2,
                           handle_size, handle_size)
            painter.drawRect(crop_x - handle_size // 2, crop_y + crop_h - handle_size // 2,
                           handle_size, handle_size)
            painter.drawRect(crop_x + crop_w - handle_size // 2, crop_y + crop_h - handle_size // 2,
                           handle_size, handle_size)

    def _draw_grid(self, painter, ix, iy, iw, ih):
        """Draw grid overlay on the image."""
        for i in range(1, self._grid_divisions):
            x = ix + (iw * i / self._grid_divisions)
            y = iy + (ih * i / self._grid_divisions)

            painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
            painter.drawLine(int(x), iy, int(x), iy + ih)
            painter.drawLine(ix, int(y), ix + iw, int(y))

            painter.setPen(QPen(QColor(255, 255, 255, 230), 1))
            painter.drawLine(int(x), iy, int(x), iy + ih)
            painter.drawLine(ix, int(y), ix + iw, int(y))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def _get_crop_edge_at_pos(self, pos):
        """Return which crop edge handle is at the given position, or None."""
        if not self._crop_mode_active or self._crop_full_pixmap is None or self._crop_bounds is None:
            return None

        pw, ph = self._crop_full_pixmap.width(), self._crop_full_pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        iw, ih = int(pw * scale), int(ph * scale)
        ix = (ww - iw) // 2
        iy = (wh - ih) // 2

        cx, cy, cw, ch = self._crop_bounds
        adj = self._crop_adjustment
        ax = cx - adj['left']
        ay = cy - adj['top']
        aw = cw + adj['left'] + adj['right']
        ah = ch + adj['top'] + adj['bottom']

        crop_x = ix + int(ax * scale)
        crop_y = iy + int(ay * scale)
        crop_w = int(aw * scale)
        crop_h = int(ah * scale)

        handle_radius = 15
        x, y = pos.x(), pos.y()

        if abs(x - crop_x) < handle_radius and abs(y - crop_y) < handle_radius:
            return 'top_left'
        if abs(x - (crop_x + crop_w)) < handle_radius and abs(y - crop_y) < handle_radius:
            return 'top_right'
        if abs(x - crop_x) < handle_radius and abs(y - (crop_y + crop_h)) < handle_radius:
            return 'bottom_left'
        if abs(x - (crop_x + crop_w)) < handle_radius and abs(y - (crop_y + crop_h)) < handle_radius:
            return 'bottom_right'

        if abs(x - crop_x) < handle_radius and abs(y - (crop_y + crop_h // 2)) < handle_radius:
            return 'left'
        if abs(x - (crop_x + crop_w)) < handle_radius and abs(y - (crop_y + crop_h // 2)) < handle_radius:
            return 'right'
        if abs(x - (crop_x + crop_w // 2)) < handle_radius and abs(y - crop_y) < handle_radius:
            return 'top'
        if abs(x - (crop_x + crop_w // 2)) < handle_radius and abs(y - (crop_y + crop_h)) < handle_radius:
            return 'bottom'

        return None

    def _is_inside_crop_box(self, pos):
        """Check if position is inside the crop box (but not on an edge handle)."""
        if not self._crop_mode_active or self._crop_full_pixmap is None or self._crop_bounds is None:
            return False

        pw, ph = self._crop_full_pixmap.width(), self._crop_full_pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        iw, ih = int(pw * scale), int(ph * scale)
        ix = (ww - iw) // 2
        iy = (wh - ih) // 2

        cx, cy, cw, ch = self._crop_bounds
        adj = self._crop_adjustment
        ax = cx - adj['left']
        ay = cy - adj['top']
        aw = cw + adj['left'] + adj['right']
        ah = ch + adj['top'] + adj['bottom']

        crop_x = ix + int(ax * scale)
        crop_y = iy + int(ay * scale)
        crop_w = int(aw * scale)
        crop_h = int(ah * scale)

        x, y = pos.x(), pos.y()
        return crop_x < x < crop_x + crop_w and crop_y < y < crop_y + crop_h

    def mousePressEvent(self, event):
        """Start dragging a crop edge, corner, or box if clicked."""
        if event.button() == Qt.LeftButton and self._crop_mode_active:
            handle = self._get_crop_edge_at_pos(event.position())
            if handle:
                self._dragging_edge = handle
                self._drag_start_pos = event.position()
                self._drag_last_pos = event.position()
                if handle in ('left', 'right', 'top', 'bottom'):
                    self._drag_start_value = self._crop_adjustment[handle]
                elif handle in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
                    self.cropCornerDragStarted.emit(handle)
                event.accept()
                return
            if self._is_inside_crop_box(event.position()):
                self._dragging_box = True
                self._drag_start_pos = event.position()
                self._drag_last_pos = event.position()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle edge/corner/box dragging or update cursor for handles."""
        if self._dragging_edge:
            pw, ph = self._crop_full_pixmap.width(), self._crop_full_pixmap.height()
            ww, wh = self.width(), self.height()
            scale = min(ww / pw, wh / ph)

            if self._dragging_edge in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
                delta_x_widget = event.position().x() - self._drag_start_pos.x()
                delta_y_widget = event.position().y() - self._drag_start_pos.y()
                delta_x_image = round(delta_x_widget / scale)
                delta_y_image = round(delta_y_widget / scale)
                self.cropCornerDragged.emit(self._dragging_edge, delta_x_image, delta_y_image)
                event.accept()
                return

            if self._dragging_edge in ('left', 'right'):
                delta_widget = event.position().x() - self._drag_start_pos.x()
                delta_image = int(delta_widget / scale)
                if self._dragging_edge == 'left':
                    new_value = self._drag_start_value - delta_image
                else:
                    new_value = self._drag_start_value + delta_image
            else:
                delta_widget = event.position().y() - self._drag_start_pos.y()
                delta_image = int(delta_widget / scale)
                if self._dragging_edge == 'top':
                    new_value = self._drag_start_value - delta_image
                else:
                    new_value = self._drag_start_value + delta_image

            delta = new_value - self._crop_adjustment[self._dragging_edge]
            if delta != 0:
                self.cropEdgeDragged.emit(self._dragging_edge, delta)
            event.accept()
            return

        if self._dragging_box:
            pw, ph = self._crop_full_pixmap.width(), self._crop_full_pixmap.height()
            ww, wh = self.width(), self.height()
            scale = min(ww / pw, wh / ph)

            delta_x_widget = event.position().x() - self._drag_last_pos.x()
            delta_y_widget = event.position().y() - self._drag_last_pos.y()
            delta_x_image = int(delta_x_widget / scale)
            delta_y_image = int(delta_y_widget / scale)

            if delta_x_image != 0 or delta_y_image != 0:
                self.cropBoxMoved.emit(delta_x_image, delta_y_image)
                self._drag_last_pos = event.position()
            event.accept()
            return

        if self._crop_mode_active:
            handle = self._get_crop_edge_at_pos(event.position())
            if handle in ('left', 'right'):
                self.setCursor(Qt.SizeHorCursor)
            elif handle in ('top', 'bottom'):
                self.setCursor(Qt.SizeVerCursor)
            elif handle in ('top_left', 'bottom_right'):
                self.setCursor(Qt.SizeFDiagCursor)
            elif handle in ('top_right', 'bottom_left'):
                self.setCursor(Qt.SizeBDiagCursor)
            elif self._is_inside_crop_box(event.position()):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Stop dragging crop edge or box."""
        if event.button() == Qt.LeftButton:
            if self._dragging_edge:
                self._dragging_edge = None
                self._drag_start_pos = None
                event.accept()
                return
            if self._dragging_box:
                self._dragging_box = False
                self._drag_start_pos = None
                self._drag_last_pos = None
                event.accept()
                return
        super().mouseReleaseEvent(event)
