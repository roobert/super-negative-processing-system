"""
SUPER NEGATIVE PROCESSING SYSTEM - Adjustment Widgets

Image adjustment controls: curves editor, preview panel, and full adjustment view.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QGroupBox, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
import numpy as np
import cv2
from scipy.interpolate import PchipInterpolator

import storage
import presets
from state import TransformState
from widgets.controls import SliderWithButtons


class CurvesWidget(QWidget):
    """Interactive curves editor with RGB and per-channel control."""

    curveChanged = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 340)

        # Control points per channel: list of (x, y) tuples
        # Default is identity line (diagonal)
        self._curves = {
            'rgb': [(0, 0), (255, 255)],
            'r': [(0, 0), (255, 255)],
            'g': [(0, 0), (255, 255)],
            'b': [(0, 0), (255, 255)],
        }
        self._current_channel = 'rgb'

        # Preset default curves (None = no preset active)
        # Mirrors SliderWithButtons._preset_default pattern
        self._preset_default_curves = None

        # Cached LUTs
        self._luts = {}
        self._luts_dirty = True

        # Interaction state
        self._dragging_point = None
        self._hover_point = None
        self._canvas_rect = None
        self._point_radius = 6

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Channel selector buttons
        btn_layout = QHBoxLayout()
        self._channel_group = QButtonGroup(self)

        channels = [('RGB', 'rgb'), ('R', 'r'), ('G', 'g'), ('B', 'b')]
        colors = ['#888', '#ff6666', '#66ff66', '#6666ff']

        for i, (label, channel) in enumerate(channels):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedWidth(40)
            btn.setStyleSheet(f"""
                QPushButton {{
                    border: 1px solid #555;
                    border-radius: 3px;
                    padding: 4px;
                }}
                QPushButton:checked {{
                    background-color: {colors[i]};
                    border: 2px solid #fff;
                }}
            """)
            if channel == 'rgb':
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, ch=channel: self._set_channel(ch))
            self._channel_group.addButton(btn)
            btn_layout.addWidget(btn)

        # Single reset button for current channel
        self._reset_channel_btn = QPushButton("↺")
        self._reset_channel_btn.setFixedSize(24, 24)
        self._reset_channel_btn.setToolTip("Reset current channel")
        self._reset_channel_btn.clicked.connect(self._reset_current_channel)
        btn_layout.addWidget(self._reset_channel_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Curves canvas (drawn in paintEvent)
        layout.addStretch()

        # Reset all button
        self._reset_all_btn = QPushButton("Reset All Curves")
        self._reset_all_btn.clicked.connect(self.reset)
        layout.addWidget(self._reset_all_btn)

        self._update_reset_button_style()
        self._update_reset_all_button_style()

    def _set_channel(self, channel: str):
        self._current_channel = channel
        self._update_reset_button_style()
        self.update()

    def get_curves(self) -> dict:
        """Get all curve data for persistence."""
        return {ch: list(pts) for ch, pts in self._curves.items()}

    def set_curves(self, curves: dict):
        """Restore curve data from persistence."""
        for ch in ['rgb', 'r', 'g', 'b']:
            if ch in curves:
                self._curves[ch] = [tuple(p) for p in curves[ch]]
        self._luts_dirty = True
        self._update_reset_button_style()
        self._update_reset_all_button_style()
        self.update()
        self.curveChanged.emit()

    def set_preset_default_curves(self, curves):
        """Set the preset's default curves. None means no preset active.

        Mirrors SliderWithButtons.set_preset_default() pattern.
        """
        if curves is None:
            self._preset_default_curves = None
        else:
            # Deep copy to avoid mutation issues
            self._preset_default_curves = {
                ch: [tuple(p) for p in pts] for ch, pts in curves.items()
            }
        self._update_reset_button_style()
        self._update_reset_all_button_style()

    def _identity_curve(self) -> list:
        """Return the identity curve (absolute default)."""
        return [(0, 0), (255, 255)]

    def _channel_curves_equal(self, a: list, b: list) -> bool:
        """Check if two channel curves are equal."""
        if len(a) != len(b):
            return False
        return all(p1 == p2 for p1, p2 in zip(a, b))

    def reset(self):
        """Reset all curves to identity."""
        for ch in self._curves:
            self._curves[ch] = [(0, 0), (255, 255)]
        self._luts_dirty = True
        self._update_reset_button_style()
        self._update_reset_all_button_style()
        self.update()
        self.curveChanged.emit()

    def _reset_current_channel(self):
        """Reset current channel to appropriate value based on state.

        Mirrors SliderWithButtons._reset() logic:
        - No preset: reset to identity (absolute default)
        - At identity, not at preset: go to preset default (Blue state)
        - At preset, not at identity: go to identity (Orange state)
        - Tweaked away from both: go to preset default (Red state)
        """
        ch = self._current_channel
        current = self._curves[ch]
        identity = self._identity_curve()
        at_absolute = self._channel_curves_equal(current, identity)

        preset_active = self._preset_default_curves is not None
        preset_curve = self._preset_default_curves.get(ch, identity) if preset_active else None
        at_preset = preset_active and self._channel_curves_equal(current, preset_curve)

        if not preset_active:
            # No preset: just reset to identity
            target = identity
        elif at_absolute and not at_preset:
            # Blue state: at identity, go to preset
            target = preset_curve
        elif at_preset and not at_absolute:
            # Orange state: at preset, go to identity
            target = identity
        else:
            # Red state: tweaked, go to preset default
            target = preset_curve

        self._curves[ch] = [tuple(p) for p in target]
        self._luts_dirty = True
        self._update_reset_button_style()
        self._update_reset_all_button_style()
        self.update()
        self.curveChanged.emit()

    def _update_reset_button_style(self):
        """Update reset button appearance based on current state.

        Mirrors SliderWithButtons._update_reset_style() with 4 states:
        - Hidden: at default (no color)
        - Orange: at preset default, can go to identity
        - Blue: at identity, can go to preset default
        - Red: tweaked away from preset, can go to preset
        """
        ch = self._current_channel
        current = self._curves[ch]
        identity = self._identity_curve()
        at_absolute = self._channel_curves_equal(current, identity)

        preset_active = self._preset_default_curves is not None
        preset_curve = self._preset_default_curves.get(ch, identity) if preset_active else None
        at_preset = preset_active and self._channel_curves_equal(current, preset_curve)

        # Helper for curve description in tooltips
        def curve_desc(curve):
            if curve == identity:
                return "identity"
            return f"{len(curve)} pts"

        if not preset_active:
            # No preset mode: simple orange when modified
            if at_absolute:
                self._reset_channel_btn.setStyleSheet("")
                self._reset_channel_btn.setToolTip(f"At default: {curve_desc(current)}")
            else:
                self._reset_channel_btn.setStyleSheet(
                    "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
                self._reset_channel_btn.setToolTip(f"Reset → {curve_desc(identity)}")
        elif at_absolute and at_preset:
            # At both defaults (they're equal) - no reset needed
            self._reset_channel_btn.setStyleSheet("")
            self._reset_channel_btn.setToolTip(f"At default: {curve_desc(current)}")
        elif at_preset and not at_absolute:
            # Orange: at preset default, can go to absolute
            self._reset_channel_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
            self._reset_channel_btn.setToolTip(f"Reset to absolute → {curve_desc(identity)}")
        elif at_absolute and not at_preset:
            # Blue: at absolute default, can go to preset
            self._reset_channel_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
            self._reset_channel_btn.setToolTip(f"Reset to preset → {curve_desc(preset_curve)}")
        else:
            # Red: tweaked away from preset, can go to preset
            self._reset_channel_btn.setStyleSheet(
                "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; }")
            self._reset_channel_btn.setToolTip(f"Reset to preset → {curve_desc(preset_curve)}")

    def _update_reset_all_button_style(self):
        """Update Reset All Curves button appearance based on whether any curves are modified."""
        identity = self._identity_curve()
        any_modified = any(
            not self._channel_curves_equal(pts, identity)
            for pts in self._curves.values()
        )
        if any_modified:
            self._reset_all_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
            self._reset_all_btn.setToolTip("Reset all curves to identity (some curves modified)")
        else:
            self._reset_all_btn.setStyleSheet("")
            self._reset_all_btn.setToolTip("All curves at identity")

    def has_changes(self) -> bool:
        """Check if any curve differs from identity."""
        for ch, points in self._curves.items():
            if points != [(0, 0), (255, 255)]:
                return True
        return False

    def _build_lut(self, channel: str) -> np.ndarray:
        """Build 256-element LUT from control points using PCHIP interpolation."""
        points = sorted(self._curves[channel], key=lambda p: p[0])

        if len(points) < 2:
            return np.arange(256, dtype=np.uint8)

        # Remove duplicate x values (keep last occurrence)
        unique_points = {}
        for p in points:
            unique_points[p[0]] = p[1]
        points = sorted(unique_points.items())

        if len(points) < 2:
            return np.arange(256, dtype=np.uint8)

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        # Use PCHIP for monotonic interpolation (prevents overshoot)
        interp = PchipInterpolator(xs, ys)
        lut = interp(np.arange(256))
        return np.clip(lut, 0, 255).astype(np.uint8)

    def get_lut(self, channel: str) -> np.ndarray:
        """Get LUT for a channel, rebuilding if dirty."""
        if self._luts_dirty:
            for ch in self._curves:
                self._luts[ch] = self._build_lut(ch)
            self._luts_dirty = False
        return self._luts.get(channel, np.arange(256, dtype=np.uint8))

    def apply_curves(self, img: np.ndarray) -> np.ndarray:
        """Apply all curves to a BGR image.

        Supports both uint8 (0-255) and float32 (0-1) input.
        For float32, uses interpolation instead of LUT for full precision.
        """
        if not self.has_changes():
            return img

        result = img.copy()

        if img.dtype == np.float32:
            # Float32 path: use interpolation for full precision
            # Apply per-channel curves (BGR order in OpenCV)
            for i, channel in enumerate(['b', 'g', 'r']):
                points = sorted(self._curves[channel], key=lambda p: p[0])
                if len(points) >= 2:
                    # Check if curve is non-trivial (not just identity)
                    if not (len(points) == 2 and points[0] == (0, 0) and points[1] == (255, 255)):
                        # Convert points from 0-255 to 0-1 range
                        xs = np.array([p[0] / 255.0 for p in points])
                        ys = np.array([p[1] / 255.0 for p in points])
                        interp = PchipInterpolator(xs, ys)
                        result[:, :, i] = np.clip(interp(result[:, :, i]), 0, 1)

            # Apply RGB (master) curve to all channels
            rgb_points = sorted(self._curves['rgb'], key=lambda p: p[0])
            if len(rgb_points) >= 2:
                if not (len(rgb_points) == 2 and rgb_points[0] == (0, 0) and rgb_points[1] == (255, 255)):
                    xs = np.array([p[0] / 255.0 for p in rgb_points])
                    ys = np.array([p[1] / 255.0 for p in rgb_points])
                    interp = PchipInterpolator(xs, ys)
                    result = np.clip(interp(result), 0, 1)
        else:
            # uint8 path: use LUT for speed
            for i, channel in enumerate(['b', 'g', 'r']):
                lut = self.get_lut(channel)
                if not np.array_equal(lut, np.arange(256)):
                    result[:, :, i] = lut[result[:, :, i]]

            # Apply RGB (master) curve to all channels
            rgb_lut = self.get_lut('rgb')
            if not np.array_equal(rgb_lut, np.arange(256)):
                result = rgb_lut[result]

        return result

    def _canvas_to_curve(self, pos) -> tuple:
        """Convert canvas position to curve coordinates (0-255)."""
        if self._canvas_rect is None:
            return None
        rx, ry, rw, rh = self._canvas_rect
        x = int((pos.x() - rx) / rw * 255)
        y = int((1 - (pos.y() - ry) / rh) * 255)
        return (max(0, min(255, x)), max(0, min(255, y)))

    def _curve_to_canvas(self, point: tuple) -> tuple:
        """Convert curve coordinates to canvas position."""
        if self._canvas_rect is None:
            return None
        rx, ry, rw, rh = self._canvas_rect
        x = rx + point[0] / 255 * rw
        y = ry + (1 - point[1] / 255) * rh
        return (x, y)

    def _find_point_at(self, pos) -> int:
        """Find index of point near position, or None."""
        curve_pos = self._canvas_to_curve(pos)
        if curve_pos is None:
            return None

        points = self._curves[self._current_channel]
        for i, p in enumerate(points):
            dx = abs(p[0] - curve_pos[0])
            dy = abs(p[1] - curve_pos[1])
            if dx < 15 and dy < 15:
                return i
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            idx = self._find_point_at(event.position().toPoint())
            if idx is not None:
                # Start dragging existing point
                self._dragging_point = idx
            else:
                # Add new point
                curve_pos = self._canvas_to_curve(event.position().toPoint())
                if curve_pos:
                    points = self._curves[self._current_channel]
                    points.append(curve_pos)
                    points.sort(key=lambda p: p[0])
                    self._dragging_point = points.index(curve_pos)
                    self._luts_dirty = True
                    self._update_reset_button_style()
                    self._update_reset_all_button_style()
                    self.update()
                    self.curveChanged.emit()

    def mouseMoveEvent(self, event):
        if self._dragging_point is not None:
            curve_pos = self._canvas_to_curve(event.position().toPoint())
            if curve_pos:
                points = self._curves[self._current_channel]
                # Don't allow moving endpoints horizontally
                if self._dragging_point == 0:
                    curve_pos = (0, curve_pos[1])
                elif self._dragging_point == len(points) - 1:
                    curve_pos = (255, curve_pos[1])
                points[self._dragging_point] = curve_pos
                self._luts_dirty = True
                self._update_reset_button_style()
                self._update_reset_all_button_style()
                self.update()
                self.curveChanged.emit()
        else:
            # Update hover state
            idx = self._find_point_at(event.position().toPoint())
            if idx != self._hover_point:
                self._hover_point = idx
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging_point is not None:
            # Check if point was dragged off canvas - remove it (except endpoints)
            points = self._curves[self._current_channel]
            if self._dragging_point > 0 and self._dragging_point < len(points) - 1:
                curve_pos = self._canvas_to_curve(event.position().toPoint())
                if curve_pos is None or curve_pos[1] < -20 or curve_pos[1] > 275:
                    points.pop(self._dragging_point)
                    self._luts_dirty = True
                    self._update_reset_button_style()
                    self._update_reset_all_button_style()
                    self.curveChanged.emit()
            self._dragging_point = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        """Double-click to remove a point (except endpoints)."""
        if event.button() == Qt.LeftButton:
            idx = self._find_point_at(event.position().toPoint())
            if idx is not None:
                points = self._curves[self._current_channel]
                if 0 < idx < len(points) - 1:
                    points.pop(idx)
                    self._luts_dirty = True
                    self._update_reset_button_style()
                    self._update_reset_all_button_style()
                    self.update()
                    self.curveChanged.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate canvas area (square, below buttons)
        margin = 10
        top_margin = 45  # Space for channel buttons
        size = min(self.width() - 2 * margin, self.height() - top_margin - 50)
        canvas_x = (self.width() - size) // 2
        canvas_y = top_margin
        self._canvas_rect = (canvas_x, canvas_y, size, size)  # (x, y, width, height)
        x, y = canvas_x, canvas_y

        # Draw background
        painter.fillRect(x, y, size, size, QColor(30, 30, 30))

        # Draw grid
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(1, 4):
            gx = x + i * size // 4
            gy = y + i * size // 4
            painter.drawLine(gx, y, gx, y + size)
            painter.drawLine(x, gy, x + size, gy)

        # Draw diagonal (identity line)
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
        painter.drawLine(x, y + size, x + size, y)

        # Channel colors (full and dimmed versions)
        channel_colors = {
            'rgb': (QColor(200, 200, 200), QColor(100, 100, 100, 80)),
            'r': (QColor(255, 100, 100), QColor(255, 100, 100, 60)),
            'g': (QColor(100, 255, 100), QColor(100, 255, 100, 60)),
            'b': (QColor(100, 100, 255), QColor(100, 100, 255, 60)),
        }

        # Draw all curves (inactive ones first, dimmed)
        draw_order = [ch for ch in ['rgb', 'r', 'g', 'b'] if ch != self._current_channel]
        draw_order.append(self._current_channel)  # Draw active channel last (on top)

        for channel in draw_order:
            is_active = (channel == self._current_channel)
            color_full, color_dim = channel_colors[channel]
            color = color_full if is_active else color_dim
            line_width = 2 if is_active else 1

            painter.setPen(QPen(color, line_width))
            lut = self.get_lut(channel)
            prev = None
            for i in range(256):
                cx = x + i / 255 * size
                cy = y + (1 - lut[i] / 255) * size
                if prev:
                    painter.drawLine(int(prev[0]), int(prev[1]), int(cx), int(cy))
                prev = (cx, cy)

        # Draw control points (only for active channel)
        color = channel_colors[self._current_channel][0]
        points = self._curves[self._current_channel]
        for i, p in enumerate(points):
            cx, cy = self._curve_to_canvas(p)
            radius = self._point_radius + 2 if i == self._hover_point else self._point_radius
            painter.setBrush(color)
            painter.setPen(QPen(Qt.white, 2))
            painter.drawEllipse(int(cx - radius), int(cy - radius), radius * 2, radius * 2)

        # Draw border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(x, y, size, size)


class AdjustmentsPreview(QWidget):
    """Large image preview with optional grid overlay and zoom/pan support."""

    # Signal emitted when crop edge is dragged: (edge_name, delta_pixels)
    cropEdgeDragged = Signal(str, int)
    # Signal emitted when crop corner drag starts: (corner_name,)
    cropCornerDragStarted = Signal(str)
    # Signal emitted when crop corner is dragged: (corner_name, cumulative_delta_x, cumulative_delta_y)
    cropCornerDragged = Signal(str, int, int)
    # Signal emitted when crop box is moved: (delta_x, delta_y) in image pixels
    cropBoxMoved = Signal(int, int)
    # Signal emitted when WB picker clicks on a point: (image_x, image_y, r, g, b)
    wbPickerClicked = Signal(int, int, float, float, float)

    def __init__(self):
        super().__init__()
        self._pixmap = None
        self._grid_enabled = False
        self._grid_divisions = 3

        # Crop mode state
        self._crop_mode_active = False
        self._crop_full_image = None
        self._crop_full_pixmap = None
        self._crop_full_pixmap_inverted = None  # Inverted version for toggle
        self._crop_bounds = None
        self._crop_adjustment = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
        self._flash_edge = None  # Edge currently flashing red
        self._crop_inverted = False  # Show inverted colors in crop preview
        self._border_flash_edge = None  # Edge to flash on border in normal view

        # White balance picker state
        self._wb_picker_active = False
        self._wb_picker_image = None  # Reference to the base image for sampling

        # Crop edge dragging state
        self._dragging_edge = None  # 'left', 'right', 'top', 'bottom', or None
        self._dragging_box = False  # True when dragging entire box
        self._drag_start_pos = None
        self._drag_start_value = 0
        self._drag_last_pos = None  # For box dragging incremental updates

        # Zoom and pan state
        self._zoom = 1.0  # 1.0 = fit to widget
        self._zoom_speed = 0.05  # 5% per wheel step
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._pan_x = 0.0  # Horizontal offset from center
        self._pan_y = 0.0  # Vertical offset from center
        self._panning = False
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._pan_start_offset_x = 0.0
        self._pan_start_offset_y = 0.0

        self.setStyleSheet("background-color: #2a2a2a; border: 1px solid #333;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Fit to screen button (shown when zoomed/panned)
        self._fit_btn = QPushButton("Fit to Screen", self)
        self._fit_btn.setStyleSheet("""
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
        """)
        self._fit_btn.clicked.connect(self.reset_view)
        self._fit_btn.hide()
        self._update_fit_button_visibility()

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

        # Preset name flash overlay
        self._flash_label = QLabel(self)
        self._flash_label.setAlignment(Qt.AlignCenter)
        self._flash_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        self._flash_label.hide()
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._flash_label.hide)

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.update()

    def set_grid_enabled(self, enabled: bool):
        self._grid_enabled = enabled
        self.update()

    def set_grid_divisions(self, divisions: int):
        self._grid_divisions = max(2, min(20, divisions))
        self.update()

    def set_border_flash(self, edge: str = None):
        """Set which edge to flash on the border (for normal view feedback)."""
        self._border_flash_edge = edge
        self.update()

    def set_wb_picker_active(self, active: bool):
        """Enable/disable white balance picker mode."""
        self._wb_picker_active = active
        if active:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_wb_picker_image(self, image: np.ndarray):
        """Set the base image to sample from for white balance."""
        self._wb_picker_image = image

    def _widget_to_image_coords(self, widget_x: float, widget_y: float) -> tuple:
        """Convert widget coordinates to image coordinates."""
        if self._pixmap is None:
            return None, None

        widget_w = self.width()
        widget_h = self.height()
        pixmap_w = self._pixmap.width()
        pixmap_h = self._pixmap.height()

        base_scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
        effective_scale = base_scale * self._zoom

        # Calculate image position in widget (centered)
        img_x = (widget_w - pixmap_w * effective_scale) / 2 + self._pan_x
        img_y = (widget_h - pixmap_h * effective_scale) / 2 + self._pan_y

        # Convert to image coordinates
        image_x = int((widget_x - img_x) / effective_scale)
        image_y = int((widget_y - img_y) / effective_scale)

        return image_x, image_y

    def set_crop_mode(self, active: bool, full_image: np.ndarray = None,
                      inverted_image: np.ndarray = None,
                      bounds: tuple = None, adjustment: dict = None,
                      flash_edge: str = None):
        """Set crop mode state for visualization."""
        self._crop_mode_active = active
        self._flash_edge = flash_edge
        if full_image is not None:
            self._crop_full_image = full_image.copy()
            # Convert float32 to uint8 for display
            display_img = full_image
            if full_image.dtype == np.float32:
                display_img = (np.clip(full_image, 0, 1) * 255).astype(np.uint8)
            # Convert to pixmap (normal/negative)
            if len(display_img.shape) == 2:
                img_rgb = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            bytes_per_line = 3 * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self._crop_full_pixmap = QPixmap.fromImage(qimg)
        if inverted_image is not None:
            # Convert float32 to uint8 for display
            display_inv = inverted_image
            if inverted_image.dtype == np.float32:
                display_inv = (np.clip(inverted_image, 0, 1) * 255).astype(np.uint8)
            # Use the properly inverted image (with orange mask removed)
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
        # Show/hide invert button
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

    def flash_preset_name(self, name: str, duration_ms: int = 1000):
        """Flash the preset name on screen briefly."""
        # Check if setting is enabled
        if not storage.get_storage().get_show_preset_name_on_change():
            return

        self._flash_label.setText(name)
        self._flash_label.adjustSize()
        # Position at top left
        self._flash_label.move(20, 20)
        self._flash_label.show()
        self._flash_label.raise_()
        self._flash_timer.start(duration_ms)

    def _is_default_view(self):
        """Check if view is at default (fit to screen) state."""
        return self._zoom == 1.0 and self._pan_x == 0.0 and self._pan_y == 0.0

    def _update_fit_button_visibility(self):
        """Show or hide the fit button based on zoom/pan state."""
        if self._is_default_view():
            self._fit_btn.hide()
        else:
            self._fit_btn.show()

    def _position_fit_button(self):
        """Position the fit button in the top-right corner."""
        margin = 10
        btn_width = self._fit_btn.sizeHint().width()
        btn_height = self._fit_btn.sizeHint().height()
        self._fit_btn.move(self.width() - btn_width - margin, margin)

    def resizeEvent(self, event):
        """Reposition button when widget is resized."""
        super().resizeEvent(event)
        self._position_fit_button()

    def reset_view(self):
        """Reset zoom and pan to default values."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._update_fit_button_visibility()
        self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if self._pixmap is None:
            return

        # Get wheel delta (120 units = 1 step)
        delta = event.angleDelta().y() / 120.0

        # Calculate zoom change at 5% per step
        zoom_change = delta * self._zoom_speed
        old_zoom = self._zoom
        self._zoom = max(self._min_zoom, min(self._max_zoom, self._zoom + zoom_change))

        # Zoom toward mouse position
        if old_zoom != self._zoom:
            # Get mouse position relative to widget center
            mouse_x = event.position().x() - self.width() / 2
            mouse_y = event.position().y() - self.height() / 2

            # Adjust pan to keep the point under the mouse stationary
            zoom_ratio = self._zoom / old_zoom
            self._pan_x = mouse_x - (mouse_x - self._pan_x) * zoom_ratio
            self._pan_y = mouse_y - (mouse_y - self._pan_y) * zoom_ratio

        self._update_fit_button_visibility()
        self.update()
        event.accept()

    def _get_crop_edge_at_pos(self, pos):
        """Return which crop edge handle is at the given position, or None."""
        if not self._crop_mode_active or self._crop_full_pixmap is None or self._crop_bounds is None:
            return None

        # Calculate image position and scale (same as in _draw_crop_mode)
        widget_w = self.width()
        widget_h = self.height()
        pixmap_w = self._crop_full_pixmap.width()
        pixmap_h = self._crop_full_pixmap.height()

        base_scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
        effective_scale = base_scale * self._zoom

        scaled_w = int(pixmap_w * effective_scale)
        scaled_h = int(pixmap_h * effective_scale)

        ix = int((widget_w - scaled_w) / 2 + self._pan_x)
        iy = int((widget_h - scaled_h) / 2 + self._pan_y)

        # Calculate crop rectangle in widget coordinates
        cx, cy, cw, ch = self._crop_bounds
        adj = self._crop_adjustment
        ax = cx - adj['left']
        ay = cy - adj['top']
        aw = cw + adj['left'] + adj['right']
        ah = ch + adj['top'] + adj['bottom']

        crop_x = ix + int(ax * effective_scale)
        crop_y = iy + int(ay * effective_scale)
        crop_w = int(aw * effective_scale)
        crop_h = int(ah * effective_scale)

        # Check if position is near any handle (within 15 pixels)
        handle_radius = 15
        x, y = pos.x(), pos.y()

        # Check corners first (they take priority over edges)
        # Top-left corner
        if abs(x - crop_x) < handle_radius and abs(y - crop_y) < handle_radius:
            return 'top_left'
        # Top-right corner
        if abs(x - (crop_x + crop_w)) < handle_radius and abs(y - crop_y) < handle_radius:
            return 'top_right'
        # Bottom-left corner
        if abs(x - crop_x) < handle_radius and abs(y - (crop_y + crop_h)) < handle_radius:
            return 'bottom_left'
        # Bottom-right corner
        if abs(x - (crop_x + crop_w)) < handle_radius and abs(y - (crop_y + crop_h)) < handle_radius:
            return 'bottom_right'

        # Edge handles (midpoints)
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

        # Calculate image position and scale (same as in _draw_crop_mode)
        widget_w = self.width()
        widget_h = self.height()
        pixmap_w = self._crop_full_pixmap.width()
        pixmap_h = self._crop_full_pixmap.height()

        base_scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
        effective_scale = base_scale * self._zoom

        scaled_w = int(pixmap_w * effective_scale)
        scaled_h = int(pixmap_h * effective_scale)

        ix = int((widget_w - scaled_w) / 2 + self._pan_x)
        iy = int((widget_h - scaled_h) / 2 + self._pan_y)

        # Calculate crop rectangle in widget coordinates
        cx, cy, cw, ch = self._crop_bounds
        adj = self._crop_adjustment
        ax = cx - adj['left']
        ay = cy - adj['top']
        aw = cw + adj['left'] + adj['right']
        ah = ch + adj['top'] + adj['bottom']

        crop_x = ix + int(ax * effective_scale)
        crop_y = iy + int(ay * effective_scale)
        crop_w = int(aw * effective_scale)
        crop_h = int(ah * effective_scale)

        x, y = pos.x(), pos.y()
        return crop_x < x < crop_x + crop_w and crop_y < y < crop_y + crop_h

    def mousePressEvent(self, event):
        """Start panning or crop edge/box dragging on left mouse button press."""
        if event.button() == Qt.LeftButton:
            # Check for WB picker mode first
            if self._wb_picker_active and self._wb_picker_image is not None:
                img_x, img_y = self._widget_to_image_coords(
                    event.position().x(), event.position().y()
                )
                if img_x is not None and img_y is not None:
                    h, w = self._wb_picker_image.shape[:2]
                    if 0 <= img_x < w and 0 <= img_y < h:
                        # Sample a 5x5 region around the click point
                        radius = 2
                        y1 = max(0, img_y - radius)
                        y2 = min(h, img_y + radius + 1)
                        x1 = max(0, img_x - radius)
                        x2 = min(w, img_x + radius + 1)

                        region = self._wb_picker_image[y1:y2, x1:x2].astype(np.float32)
                        # BGR order in OpenCV
                        mean_b = float(np.mean(region[:, :, 0]))
                        mean_g = float(np.mean(region[:, :, 1]))
                        mean_r = float(np.mean(region[:, :, 2]))

                        # Emit signal with sampled RGB values
                        self.wbPickerClicked.emit(img_x, img_y, mean_r, mean_g, mean_b)
                        event.accept()
                        return

            # Check for crop edge/corner dragging
            if self._crop_mode_active:
                handle = self._get_crop_edge_at_pos(event.position())
                if handle:
                    self._dragging_edge = handle
                    self._drag_start_pos = event.position()
                    self._drag_last_pos = event.position()
                    # Store start value for single edges
                    if handle in ('left', 'right', 'top', 'bottom'):
                        self._drag_start_value = self._crop_adjustment[handle]
                    elif handle in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
                        # Emit corner drag started so main window can capture start values
                        self.cropCornerDragStarted.emit(handle)
                    event.accept()
                    return
                # Check for box dragging (inside box but not on edge)
                if self._is_inside_crop_box(event.position()):
                    self._dragging_box = True
                    self._drag_start_pos = event.position()
                    self._drag_last_pos = event.position()
                    self.setCursor(Qt.SizeAllCursor)
                    event.accept()
                    return

            # Otherwise start panning
            if self._pixmap is not None or (self._crop_mode_active and self._crop_full_pixmap is not None):
                self._panning = True
                self._pan_start_x = event.position().x()
                self._pan_start_y = event.position().y()
                self._pan_start_offset_x = self._pan_x
                self._pan_start_offset_y = self._pan_y
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()

    def mouseMoveEvent(self, event):
        """Handle panning or crop edge/corner/box dragging while mouse is pressed."""
        # Handle crop edge/corner dragging
        if self._dragging_edge:
            widget_w = self.width()
            widget_h = self.height()
            pixmap_w = self._crop_full_pixmap.width()
            pixmap_h = self._crop_full_pixmap.height()
            base_scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
            effective_scale = base_scale * self._zoom

            # Handle corner dragging - emit raw image-space deltas from drag start
            # Main window will track start values and apply constrained values absolutely
            if self._dragging_edge in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
                delta_x_widget = event.position().x() - self._drag_start_pos.x()
                delta_y_widget = event.position().y() - self._drag_start_pos.y()
                # Convert to image space - these are CUMULATIVE deltas from drag start
                delta_x_image = round(delta_x_widget / effective_scale)
                delta_y_image = round(delta_y_widget / effective_scale)
                self.cropCornerDragged.emit(self._dragging_edge, delta_x_image, delta_y_image)
                event.accept()
                return

            # Handle single edge dragging
            if self._dragging_edge in ('left', 'right'):
                delta_widget = event.position().x() - self._drag_start_pos.x()
                delta_image = int(delta_widget / effective_scale)
                if self._dragging_edge == 'left':
                    new_value = self._drag_start_value - delta_image
                else:
                    new_value = self._drag_start_value + delta_image
            else:
                delta_widget = event.position().y() - self._drag_start_pos.y()
                delta_image = int(delta_widget / effective_scale)
                if self._dragging_edge == 'top':
                    new_value = self._drag_start_value - delta_image
                else:
                    new_value = self._drag_start_value + delta_image

            delta = new_value - self._crop_adjustment[self._dragging_edge]
            if delta != 0:
                self.cropEdgeDragged.emit(self._dragging_edge, delta)
            event.accept()
            return

        # Handle crop box dragging
        if self._dragging_box:
            widget_w = self.width()
            widget_h = self.height()
            pixmap_w = self._crop_full_pixmap.width()
            pixmap_h = self._crop_full_pixmap.height()
            base_scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
            effective_scale = base_scale * self._zoom

            delta_x_widget = event.position().x() - self._drag_last_pos.x()
            delta_y_widget = event.position().y() - self._drag_last_pos.y()
            delta_x_image = int(delta_x_widget / effective_scale)
            delta_y_image = int(delta_y_widget / effective_scale)

            if delta_x_image != 0 or delta_y_image != 0:
                self.cropBoxMoved.emit(delta_x_image, delta_y_image)
                self._drag_last_pos = event.position()
            event.accept()
            return

        # Handle panning
        if self._panning:
            dx = event.position().x() - self._pan_start_x
            dy = event.position().y() - self._pan_start_y
            self._pan_x = self._pan_start_offset_x + dx
            self._pan_y = self._pan_start_offset_y + dy
            self._update_fit_button_visibility()
            self.update()
            event.accept()
            return

        # Update cursor based on position (but not if WB picker is active)
        if self._wb_picker_active:
            self.setCursor(Qt.CrossCursor)
        elif self._crop_mode_active:
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
            elif self._pixmap is not None or self._crop_full_pixmap is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        elif self._pixmap is not None:
            self.setCursor(Qt.OpenHandCursor)

    def mouseReleaseEvent(self, event):
        """Stop panning or crop edge/corner/box dragging on mouse release."""
        if event.button() == Qt.LeftButton:
            if self._dragging_edge:
                self._dragging_edge = None
                self._drag_start_pos = None
                self._drag_last_pos = None
            if self._dragging_box:
                self._dragging_box = False
                self._drag_start_pos = None
                self._drag_last_pos = None
            self._panning = False
            # Update cursor (but not if WB picker is active)
            if self._wb_picker_active:
                self.setCursor(Qt.CrossCursor)
            elif self._crop_mode_active:
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
                    self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.OpenHandCursor)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """Reset view on double-click."""
        if event.button() == Qt.LeftButton:
            self.reset_view()
            event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(42, 42, 42))

        # Crop mode: draw full image with crop overlay
        if self._crop_mode_active and self._crop_full_pixmap is not None:
            self._draw_crop_mode(painter)
            return

        if self._pixmap is None:
            return

        # Calculate base scale (fit to widget while maintaining aspect ratio)
        widget_w = self.width()
        widget_h = self.height()
        pixmap_w = self._pixmap.width()
        pixmap_h = self._pixmap.height()

        base_scale = min(widget_w / pixmap_w, widget_h / pixmap_h)

        # Apply zoom factor
        effective_scale = base_scale * self._zoom

        # Calculate scaled dimensions
        scaled_w = int(pixmap_w * effective_scale)
        scaled_h = int(pixmap_h * effective_scale)

        # Calculate position (centered + pan offset)
        x = int((widget_w - scaled_w) / 2 + self._pan_x)
        y = int((widget_h - scaled_h) / 2 + self._pan_y)

        # Scale and draw the pixmap
        scaled = self._pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(x, y, scaled)

        # Draw grid overlay if enabled
        if self._grid_enabled and self._grid_divisions > 0:
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
            for i in range(1, self._grid_divisions):
                # Vertical lines
                gx = x + i * scaled_w // self._grid_divisions
                painter.drawLine(gx, y, gx, y + scaled_h)
                # Horizontal lines
                gy = y + i * scaled_h // self._grid_divisions
                painter.drawLine(x, gy, x + scaled_w, gy)

        # Draw border flash for crop edge feedback in normal view
        if self._border_flash_edge:
            flash_pen = QPen(QColor(255, 0, 0), 3)
            flash_pen.setDashPattern([8, 8])
            painter.setPen(flash_pen)
            if self._border_flash_edge == 'left':
                painter.drawLine(x, y, x, y + scaled_h)
            elif self._border_flash_edge == 'right':
                painter.drawLine(x + scaled_w, y, x + scaled_w, y + scaled_h)
            elif self._border_flash_edge == 'top':
                painter.drawLine(x, y, x + scaled_w, y)
            elif self._border_flash_edge == 'bottom':
                painter.drawLine(x, y + scaled_h, x + scaled_w, y + scaled_h)

    def _draw_crop_mode(self, painter):
        """Draw crop mode visualization with full image and crop overlay."""
        widget_w = self.width()
        widget_h = self.height()
        # Use inverted pixmap if toggled
        pixmap = self._crop_full_pixmap_inverted if self._crop_inverted else self._crop_full_pixmap
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()

        base_scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
        effective_scale = base_scale * self._zoom

        scaled_w = int(pixmap_w * effective_scale)
        scaled_h = int(pixmap_h * effective_scale)

        x = int((widget_w - scaled_w) / 2 + self._pan_x)
        y = int((widget_h - scaled_h) / 2 + self._pan_y)

        # Draw the full rotated image (normal or inverted)
        scaled = pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(x, y, scaled)

        # Calculate adjusted crop bounds in widget coordinates
        if self._crop_bounds is not None:
            cx, cy, cw, ch = self._crop_bounds
            adj = self._crop_adjustment

            # Apply adjustments
            ax = cx - adj['left']
            ay = cy - adj['top']
            aw = cw + adj['left'] + adj['right']
            ah = ch + adj['top'] + adj['bottom']

            # Scale to widget coordinates
            crop_x = x + int(ax * effective_scale)
            crop_y = y + int(ay * effective_scale)
            crop_w = int(aw * effective_scale)
            crop_h = int(ah * effective_scale)

            # Draw darkened overlay outside crop area
            overlay = QColor(0, 0, 0, 160)
            painter.fillRect(x, y, scaled_w, crop_y - y, overlay)
            painter.fillRect(x, crop_y + crop_h, scaled_w, y + scaled_h - crop_y - crop_h, overlay)
            painter.fillRect(x, crop_y, crop_x - x, crop_h, overlay)
            painter.fillRect(crop_x + crop_w, crop_y, x + scaled_w - crop_x - crop_w, crop_h, overlay)

            # Draw crop rectangle border (draw each edge separately for flash effect)
            normal_color = QColor(0, 255, 0)
            flash_color = QColor(255, 0, 0)

            def make_crop_pen(color):
                pen = QPen(color, 1)
                pen.setDashPattern([8, 8])  # Equal dash and gap length
                return pen

            # Left edge
            is_flash = self._flash_edge == 'left'
            painter.setPen(make_crop_pen(flash_color if is_flash else normal_color))
            painter.drawLine(crop_x, crop_y, crop_x, crop_y + crop_h)

            # Right edge
            is_flash = self._flash_edge == 'right'
            painter.setPen(make_crop_pen(flash_color if is_flash else normal_color))
            painter.drawLine(crop_x + crop_w, crop_y, crop_x + crop_w, crop_y + crop_h)

            # Top edge
            is_flash = self._flash_edge == 'top'
            painter.setPen(make_crop_pen(flash_color if is_flash else normal_color))
            painter.drawLine(crop_x, crop_y, crop_x + crop_w, crop_y)

            # Bottom edge
            is_flash = self._flash_edge == 'bottom'
            painter.setPen(make_crop_pen(flash_color if is_flash else normal_color))
            painter.drawLine(crop_x, crop_y + crop_h, crop_x + crop_w, crop_y + crop_h)

            # Draw edge handles
            handle_size = 10

            # Left handle
            handle_color = flash_color if self._flash_edge == 'left' else normal_color
            painter.setBrush(handle_color)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawRect(crop_x - handle_size // 2, crop_y + crop_h // 2 - handle_size // 2, handle_size, handle_size)

            # Right handle
            handle_color = flash_color if self._flash_edge == 'right' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x + crop_w - handle_size // 2, crop_y + crop_h // 2 - handle_size // 2, handle_size, handle_size)

            # Top handle
            handle_color = flash_color if self._flash_edge == 'top' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x + crop_w // 2 - handle_size // 2, crop_y - handle_size // 2, handle_size, handle_size)

            # Bottom handle
            handle_color = flash_color if self._flash_edge == 'bottom' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x + crop_w // 2 - handle_size // 2, crop_y + crop_h - handle_size // 2, handle_size, handle_size)

            # Draw corner handles
            corner_size = 8

            # Top-left corner
            painter.setBrush(normal_color)
            painter.drawRect(crop_x - corner_size // 2, crop_y - corner_size // 2, corner_size, corner_size)

            # Top-right corner
            painter.drawRect(crop_x + crop_w - corner_size // 2, crop_y - corner_size // 2, corner_size, corner_size)

            # Bottom-left corner
            painter.drawRect(crop_x - corner_size // 2, crop_y + crop_h - corner_size // 2, corner_size, corner_size)

            # Bottom-right corner
            painter.drawRect(crop_x + crop_w - corner_size // 2, crop_y + crop_h - corner_size // 2, corner_size, corner_size)


class AdjustmentsView(QWidget):
    """Full-screen view for image adjustments with curves editor."""

    # Signals for changes
    adjustmentsChanged = Signal()  # Emitted when any adjustment changes
    presetSelected = Signal(str)  # Emitted when a preset is selected

    def __init__(self, transform_state: TransformState):
        super().__init__()
        self._transform_state = transform_state
        self._image = None
        self._adjusted_image = None
        self._current_preset = 'none'
        self._preset_states = {}  # Maps preset_key -> {'adjustments': {...}, 'curves': {...}}
        self._loading_preset = False  # Flag to prevent modification tracking during preset load

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Import here to avoid circular imports
        from widgets.panels import CollapsiblePresetPanel, CollapsibleAdjustmentsPanel

        # Left: Collapsible preset bar with thumbnails
        self._preset_panel = CollapsiblePresetPanel()
        self._preset_panel.presetSelected.connect(self._on_preset_selected)
        self._preset_panel.fullModeChanged.connect(self._on_preset_full_mode_changed)
        layout.addWidget(self._preset_panel)

        # Middle: Preview area (full height, transform is now in right panel)
        self._preview = AdjustmentsPreview()
        self._preview.wbPickerClicked.connect(self._on_wb_sample)

        # Connect transform state to preview for grid updates
        self._transform_state.gridEnabledChanged.connect(self._preview.set_grid_enabled)
        self._transform_state.gridDivisionsChanged.connect(self._preview.set_grid_divisions)

        layout.addWidget(self._preview, stretch=1)

        # Right: Scrollable controls panel (wrapped in collapsible container)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(10, 10, 10, 0)

        # White Balance section (technical correction - before exposure)
        wb_group = QGroupBox("White Balance")
        wb_layout = QVBoxLayout(wb_group)

        wb_header = QHBoxLayout()
        wb_header.addStretch()
        self.wb_picker_btn = QPushButton("⦿ Pick Neutral")
        self.wb_picker_btn.setToolTip("Click a neutral gray area in the image to set white balance")
        self.wb_picker_btn.setCheckable(True)
        self.wb_picker_btn.clicked.connect(self._on_wb_picker_clicked)
        wb_header.addWidget(self.wb_picker_btn)
        wb_layout.addLayout(wb_header)

        self.wb_r_slider = SliderWithButtons("R", 0.5, 2.0, 1.0, step=0.01, decimals=2,
            info_text="<b>When to use:</b> Correct color casts. Use 'Pick Neutral' button first.<br><br>"
                      "<b>How it works:</b> Multiplies red channel. >1 adds red, <1 adds cyan.<br><br>"
                      "<b>Tip:</b> If image is too cyan/blue, increase R. Too red? Decrease R.")
        self.wb_r_slider.setToolTip("Red channel multiplier")
        self.wb_r_slider.set_gradient('#00cccc', '#ff4444')  # Cyan → Red
        self.wb_r_slider.valueChanged.connect(self._on_adjustment_changed)
        wb_layout.addWidget(self.wb_r_slider)

        self.wb_g_slider = SliderWithButtons("G", 0.5, 2.0, 1.0, step=0.01, decimals=2,
            info_text="<b>When to use:</b> Correct color casts, especially green/magenta.<br><br>"
                      "<b>How it works:</b> Multiplies green channel. >1 adds green, <1 adds magenta.<br><br>"
                      "<b>Tip:</b> Fluorescent lights often need G reduced (adds magenta).")
        self.wb_g_slider.setToolTip("Green channel multiplier")
        self.wb_g_slider.set_gradient('#cc44cc', '#44cc44')  # Magenta → Green
        self.wb_g_slider.valueChanged.connect(self._on_adjustment_changed)
        wb_layout.addWidget(self.wb_g_slider)

        self.wb_b_slider = SliderWithButtons("B", 0.5, 2.0, 1.0, step=0.01, decimals=2,
            info_text="<b>When to use:</b> Correct color casts, especially warm/cool.<br><br>"
                      "<b>How it works:</b> Multiplies blue channel. >1 adds blue, <1 adds yellow.<br><br>"
                      "<b>Tip:</b> Tungsten/indoor shots often need B increased (removes yellow).")
        self.wb_b_slider.setToolTip("Blue channel multiplier")
        self.wb_b_slider.set_gradient('#cccc44', '#4444ff')  # Yellow → Blue
        self.wb_b_slider.valueChanged.connect(self._on_adjustment_changed)
        wb_layout.addWidget(self.wb_b_slider)

        controls_layout.addWidget(wb_group)

        # Basic adjustments section
        basic_group = QGroupBox("Basic Adjustments")
        basic_layout = QVBoxLayout(basic_group)

        self.exposure_slider = SliderWithButtons("Exposure", -2.0, 2.0, 0, step=0.05, decimals=2,
            info_text="<b>When to use:</b> First adjustment to make. Fix overall brightness.<br><br>"
                      "<b>How it works:</b> Multiplies all pixels by 2^value (like camera stops).<br><br>"
                      "<b>Tip:</b> +1.0 = double brightness, -1.0 = half brightness.")
        self.exposure_slider.setToolTip("Adjust overall brightness (stops)")
        self.exposure_slider.valueChanged.connect(self._on_adjustment_changed)
        basic_layout.addWidget(self.exposure_slider)

        self.contrast_slider = SliderWithButtons("Contrast", -100, 100, 0, step=1, decimals=0,
            info_text="<b>When to use:</b> After exposure. Makes darks darker and lights lighter.<br><br>"
                      "<b>How it works:</b> S-curve around midtones - pushes values away from middle gray.<br><br>"
                      "<b>Tip:</b> Negative values create a flatter, faded look.")
        self.contrast_slider.setToolTip("Adjust contrast")
        self.contrast_slider.valueChanged.connect(self._on_adjustment_changed)
        basic_layout.addWidget(self.contrast_slider)

        self.highlights_slider = SliderWithButtons("Highlights", -100, 100, 0, step=1, decimals=0,
            info_text="<b>When to use:</b> Recover detail in bright areas (sky, windows).<br><br>"
                      "<b>How it works:</b> Only affects pixels brighter than middle gray.<br><br>"
                      "<b>Tip:</b> Negative values bring back blown-out highlights.")
        self.highlights_slider.setToolTip("Recover highlight detail")
        self.highlights_slider.valueChanged.connect(self._on_adjustment_changed)
        basic_layout.addWidget(self.highlights_slider)

        self.shadows_slider = SliderWithButtons("Shadows", -100, 100, 0, step=1, decimals=0,
            info_text="<b>When to use:</b> Recover detail in dark areas without affecting highlights.<br><br>"
                      "<b>How it works:</b> Only affects pixels darker than middle gray.<br><br>"
                      "<b>Tip:</b> Positive values lift shadows, revealing hidden detail.")
        self.shadows_slider.setToolTip("Recover shadow detail")
        self.shadows_slider.valueChanged.connect(self._on_adjustment_changed)
        basic_layout.addWidget(self.shadows_slider)

        controls_layout.addWidget(basic_group)

        # Levels section
        levels_group = QGroupBox("Levels")
        levels_layout = QVBoxLayout(levels_group)

        self.blacks_slider = SliderWithButtons("Blacks", 0, 100, 0, step=1, decimals=0,
            info_text="<b>When to use:</b> Set the black point - clip the darkest shadows.<br><br>"
                      "<b>How it works:</b> Values below threshold become pure black (0).<br><br>"
                      "<b>Tip:</b> Increases contrast by crushing blacks. Use with histogram visible.")
        self.blacks_slider.setToolTip("Set black point (clip shadows)")
        self.blacks_slider.valueChanged.connect(self._on_adjustment_changed)
        levels_layout.addWidget(self.blacks_slider)

        self.whites_slider = SliderWithButtons("Whites", 0, 100, 0, step=1, decimals=0,
            info_text="<b>When to use:</b> Set the white point - clip the brightest highlights.<br><br>"
                      "<b>How it works:</b> Values above threshold become pure white (255).<br><br>"
                      "<b>Tip:</b> Use to maximize dynamic range. Watch for blown highlights.")
        self.whites_slider.setToolTip("Set white point (clip highlights)")
        self.whites_slider.valueChanged.connect(self._on_adjustment_changed)
        levels_layout.addWidget(self.whites_slider)

        self.gamma_slider = SliderWithButtons("Gamma", 0.2, 3.0, 1.0, step=0.05, decimals=2,
            info_text="<b>When to use:</b> Adjust midtone brightness without affecting black/white points.<br><br>"
                      "<b>How it works:</b> Non-linear brightness curve. <1 = darker midtones, >1 = brighter.<br><br>"
                      "<b>Tip:</b> Similar to Exposure but preserves highlights/shadows better.")
        self.gamma_slider.setToolTip("Midtone adjustment")
        self.gamma_slider.valueChanged.connect(self._on_adjustment_changed)
        levels_layout.addWidget(self.gamma_slider)

        controls_layout.addWidget(levels_group)

        # Curves section (fine tonal control)
        curves_group = QGroupBox("Curves")
        curves_group_layout = QVBoxLayout(curves_group)

        self.curves_widget = CurvesWidget()
        self.curves_widget.curveChanged.connect(self._on_curves_changed)
        curves_group_layout.addWidget(self.curves_widget)

        controls_layout.addWidget(curves_group)

        # Color Grading section (creative - after technical corrections)
        color_group = QGroupBox("Color Grading")
        color_layout = QVBoxLayout(color_group)

        self.temperature_slider = SliderWithButtons("Temperature", -100, 100, 0, step=1, decimals=0,
            info_text="<b>When to use:</b> Creative color grading AFTER white balance is set.<br><br>"
                      "<b>How it works:</b> Adds fixed amount to red/blue channels (additive shift).<br><br>"
                      "<b>Tip:</b> Use WB sliders for correction, this for creative warmth/coolness.")
        self.temperature_slider.setToolTip("Warm (yellow) ↔ Cool (blue)")
        self.temperature_slider.valueChanged.connect(self._on_adjustment_changed)
        color_layout.addWidget(self.temperature_slider)

        self.vibrance_slider = SliderWithButtons("Vibrance", -100, 100, 0, step=1, decimals=0,
            info_text="<b>When to use:</b> Boost colors without oversaturating skin tones.<br><br>"
                      "<b>How it works:</b> 'Smart' saturation - affects muted colors more than already-saturated ones.<br><br>"
                      "<b>Tip:</b> Safer than Saturation for portraits. Boost landscapes without neon skin.")
        self.vibrance_slider.setToolTip("Smart saturation (protects skin tones)")
        self.vibrance_slider.valueChanged.connect(self._on_adjustment_changed)
        color_layout.addWidget(self.vibrance_slider)

        self.saturation_slider = SliderWithButtons("Saturation", -100, 100, 0, step=1, decimals=0,
            info_text="<b>When to use:</b> Global color intensity adjustment.<br><br>"
                      "<b>How it works:</b> Uniformly scales all color intensity. -100 = grayscale.<br><br>"
                      "<b>Tip:</b> Use sparingly. Vibrance is usually better for natural-looking results.")
        self.saturation_slider.setToolTip("Color intensity")
        self.saturation_slider.valueChanged.connect(self._on_adjustment_changed)
        color_layout.addWidget(self.saturation_slider)

        controls_layout.addWidget(color_group)

        # Detail section (applied last)
        detail_group = QGroupBox("Detail")
        detail_layout = QVBoxLayout(detail_group)

        self.sharpening_slider = SliderWithButtons("Sharpening", 0, 100, 0, step=1, decimals=0,
            info_text="<b>When to use:</b> Final step - enhance edge detail and perceived sharpness.<br><br>"
                      "<b>How it works:</b> Unsharp mask - increases contrast at edges.<br><br>"
                      "<b>Tip:</b> Apply last, after all other adjustments. Too much creates halos.")
        self.sharpening_slider.setToolTip("Enhance edge detail")
        self.sharpening_slider.valueChanged.connect(self._on_adjustment_changed)
        detail_layout.addWidget(self.sharpening_slider)

        controls_layout.addWidget(detail_group)

        controls_layout.addStretch()
        scroll_area.setWidget(controls_panel)

        # Wrap scroll area in collapsible panel
        self._adjustments_panel = CollapsibleAdjustmentsPanel(scroll_area)
        layout.addWidget(self._adjustments_panel)

    def resizeEvent(self, event):
        """Update preset panel full width on resize."""
        super().resizeEvent(event)
        self._update_preset_full_width()

    def _update_preset_full_width(self):
        """Calculate and set the full width for preset panel grid mode."""
        # Full width = total width - adjustments panel width - toggle button - margins
        adjustments_width = self._adjustments_panel.width() if self._adjustments_panel.is_expanded() else 28
        toggle_width = 28  # SplitVerticalToggleButton width
        margins = 30  # Layout margins and spacing
        full_width = self.width() - adjustments_width - toggle_width - margins
        # Ensure minimum width
        full_width = max(full_width, 400)
        self._preset_panel.set_full_width(full_width)

    def set_crop_mode_active(self, active: bool):
        """Update crop mode via shared transform state."""
        self._transform_state.crop_mode = active

    def toggle_preset_panel(self):
        """Toggle the preset panel visibility (for keyboard shortcut)."""
        self._preset_panel.toggle()

    def _on_preset_full_mode_changed(self, is_full: bool):
        """Handle preset panel entering/exiting full mode."""
        if is_full:
            # Hide preview and let preset panel take the space
            self._preview.hide()
            # Update full width synchronously so animation uses correct target
            self._update_preset_full_width()
        else:
            # Show preview again
            self._preview.show()

    def toggle_adjustments_panel(self):
        """Toggle the adjustments panel visibility (for keyboard shortcut)."""
        self._adjustments_panel.toggle()
        # Update preset panel full width after toggle animation
        QTimer.singleShot(300, self._update_preset_full_width)

    def apply_preset_by_number(self, number: int):
        """Apply a preset by its shortcut number (1-9)."""
        preset_key = self._preset_panel.get_preset_key_by_number(number)
        if preset_key:
            self._preset_panel.select_preset(preset_key)
            self._on_preset_selected(preset_key)

    def move_current_preset_up(self):
        """Move current preset up in the list."""
        self._preset_panel.move_current_preset_up()

    def move_current_preset_down(self):
        """Move current preset down in the list."""
        self._preset_panel.move_current_preset_down()

    def select_previous_preset(self):
        """Select the previous preset in the list."""
        new_key = self._preset_panel.select_previous_preset()
        if new_key:
            self._on_preset_selected(new_key)

    def select_next_preset(self):
        """Select the next preset in the list."""
        new_key = self._preset_panel.select_next_preset()
        if new_key:
            self._on_preset_selected(new_key)

    def _on_adjustment_changed(self, value):
        """Handle any adjustment slider change."""
        if not self._loading_preset:
            self._update_preset_indicator()
        self._update_preview()
        self.adjustmentsChanged.emit()

    def _on_wb_picker_clicked(self, checked: bool):
        """Handle WB picker button click."""
        self._preview.set_wb_picker_active(checked)
        if checked and self._image is not None:
            self._preview.set_wb_picker_image(self._image)
        if not checked:
            self.wb_picker_btn.setChecked(False)

    def _on_wb_sample(self, img_x: int, img_y: int, mean_r: float, mean_g: float, mean_b: float):
        """Handle WB picker sample from preview.

        Calculates multipliers to make the sampled point neutral (R=G=B).
        Picker stays active so user can click multiple points to compare.
        """
        # Calculate neutral target (average of R, G, B)
        gray_target = (mean_r + mean_g + mean_b) / 3.0

        # Avoid division by zero
        epsilon = 1e-6

        # Calculate multipliers to make this point neutral
        wb_r = gray_target / max(mean_r, epsilon)
        wb_g = gray_target / max(mean_g, epsilon)
        wb_b = gray_target / max(mean_b, epsilon)

        # Clamp to reasonable range
        wb_r = max(0.5, min(2.0, wb_r))
        wb_g = max(0.5, min(2.0, wb_g))
        wb_b = max(0.5, min(2.0, wb_b))

        # Set the sliders (this will trigger _on_adjustment_changed and update preview)
        self.wb_r_slider.setValue(wb_r)
        self.wb_g_slider.setValue(wb_g)
        self.wb_b_slider.setValue(wb_b)

    def set_image(self, img: np.ndarray, image_hash: str = None):
        """Set the base image (inverted negative before adjustments).

        Args:
            img: The base image to use for adjustments.
            image_hash: Optional hash for caching preset thumbnails.
        """
        self._image = img.copy() if img is not None else None
        # Update preset bar with base image for thumbnails
        self._preset_panel.set_base_image(self._image, image_hash)
        self._update_preview()

    def set_crop_mode(self, active: bool, full_image: np.ndarray = None,
                      inverted_image: np.ndarray = None,
                      bounds: tuple = None, adjustment: dict = None,
                      flash_edge: str = None):
        """Set crop mode state for the preview."""
        self._preview.set_crop_mode(active, full_image, inverted_image, bounds, adjustment, flash_edge)

    def get_adjusted_image(self) -> np.ndarray:
        """Get the image with all adjustments applied."""
        return self._adjusted_image

    def get_adjustments(self) -> dict:
        """Get all adjustment values for persistence."""
        return {
            'exposure': self.exposure_slider.value(),
            'wb_r': self.wb_r_slider.value(),
            'wb_g': self.wb_g_slider.value(),
            'wb_b': self.wb_b_slider.value(),
            'contrast': self.contrast_slider.value(),
            'highlights': self.highlights_slider.value(),
            'shadows': self.shadows_slider.value(),
            'temperature': self.temperature_slider.value(),
            'vibrance': self.vibrance_slider.value(),
            'saturation': self.saturation_slider.value(),
            'blacks': self.blacks_slider.value(),
            'whites': self.whites_slider.value(),
            'gamma': self.gamma_slider.value(),
            'sharpening': self.sharpening_slider.value(),
        }

    def set_adjustments(self, adjustments: dict):
        """Restore adjustment values from persistence."""
        if not adjustments:
            self.reset_adjustments()
            return
        # Block signals during restore to avoid multiple updates
        sliders = [
            self.exposure_slider, self.wb_r_slider, self.wb_g_slider, self.wb_b_slider,
            self.contrast_slider, self.highlights_slider,
            self.shadows_slider, self.temperature_slider, self.vibrance_slider,
            self.saturation_slider, self.blacks_slider, self.whites_slider,
            self.gamma_slider, self.sharpening_slider
        ]
        for s in sliders:
            s.blockSignals(True)

        self.exposure_slider.setValue(adjustments.get('exposure', 0))
        self.wb_r_slider.setValue(adjustments.get('wb_r', 1.0))
        self.wb_g_slider.setValue(adjustments.get('wb_g', 1.0))
        self.wb_b_slider.setValue(adjustments.get('wb_b', 1.0))
        self.contrast_slider.setValue(adjustments.get('contrast', 0))
        self.highlights_slider.setValue(adjustments.get('highlights', 0))
        self.shadows_slider.setValue(adjustments.get('shadows', 0))
        self.temperature_slider.setValue(adjustments.get('temperature', 0))
        self.vibrance_slider.setValue(adjustments.get('vibrance', 0))
        self.saturation_slider.setValue(adjustments.get('saturation', 0))
        self.blacks_slider.setValue(adjustments.get('blacks', 0))
        self.whites_slider.setValue(adjustments.get('whites', 0))
        self.gamma_slider.setValue(adjustments.get('gamma', 1.0))
        self.sharpening_slider.setValue(adjustments.get('sharpening', 0))

        for s in sliders:
            s.blockSignals(False)
        self._update_preview()

    def reset_adjustments(self):
        """Reset all adjustments to defaults."""
        sliders = [
            self.exposure_slider, self.wb_r_slider, self.wb_g_slider, self.wb_b_slider,
            self.contrast_slider, self.highlights_slider,
            self.shadows_slider, self.temperature_slider, self.vibrance_slider,
            self.saturation_slider, self.blacks_slider, self.whites_slider,
            self.gamma_slider, self.sharpening_slider
        ]
        for s in sliders:
            s.blockSignals(True)
            s.setValue(s.default)
            s.blockSignals(False)
        self._update_preview()

    def _save_current_preset_state(self):
        """Save current adjustments/curves to the preset_states dict."""
        self._preset_states[self._current_preset] = {
            'adjustments': self.get_adjustments(),
            'curves': self.curves_widget.get_curves(),
        }

    def _on_preset_selected(self, preset_key: str):
        """Handle preset selection from the preset bar."""
        # Save current state before switching
        self._save_current_preset_state()
        self._load_preset(preset_key)
        self.presetSelected.emit(preset_key)

        # Flash the preset name
        preset = presets.get_preset(preset_key)
        self._preview.flash_preset_name(preset.get('name', preset_key))

    def _get_adjustment_sliders(self) -> dict:
        """Return mapping of adjustment names to slider widgets."""
        return {
            'exposure': self.exposure_slider,
            'contrast': self.contrast_slider,
            'highlights': self.highlights_slider,
            'shadows': self.shadows_slider,
            'temperature': self.temperature_slider,
            'vibrance': self.vibrance_slider,
            'saturation': self.saturation_slider,
            'blacks': self.blacks_slider,
            'whites': self.whites_slider,
            'gamma': self.gamma_slider,
            'sharpening': self.sharpening_slider,
        }

    def _update_slider_preset_defaults(self, preset_key: str):
        """Update all sliders with the preset's default values for reset button behavior."""
        if preset_key == 'none':
            # No preset - clear all preset defaults
            for slider in self._get_adjustment_sliders().values():
                slider.set_preset_default(None)
        else:
            # Set preset defaults on each slider
            preset = presets.get_preset(preset_key)
            preset_adjustments = preset.get('adjustments', {})
            for name, slider in self._get_adjustment_sliders().items():
                preset_val = preset_adjustments.get(name, slider.default)
                slider.set_preset_default(preset_val)

    def _update_curves_preset_default(self, preset_key: str):
        """Update curves widget with the preset's default curves for reset button behavior."""
        if preset_key == 'none':
            # No preset - clear preset default curves
            self.curves_widget.set_preset_default_curves(None)
        else:
            # Set preset default curves
            preset = presets.get_preset(preset_key)
            preset_curves = preset.get('curves', {})
            if preset_curves:
                self.curves_widget.set_preset_default_curves(preset_curves)
            else:
                # Preset has no curves defined - use identity as preset default
                identity_curves = {ch: [(0, 0), (255, 255)] for ch in ['rgb', 'r', 'g', 'b']}
                self.curves_widget.set_preset_default_curves(identity_curves)

    def _load_preset(self, preset_key: str):
        """Load a preset's adjustments and curves into the UI.

        If the user has previously modified this preset, load their saved state.
        Otherwise load the preset defaults.
        """
        self._current_preset = preset_key
        self._loading_preset = True  # Prevent modification tracking during load

        # Update slider and curves preset defaults for reset button behavior
        self._update_slider_preset_defaults(preset_key)
        self._update_curves_preset_default(preset_key)

        # Check if we have saved state for this preset
        if preset_key in self._preset_states:
            # Load user's saved state for this preset
            state = self._preset_states[preset_key]
            self.set_adjustments(state.get('adjustments', {}))
            curves = state.get('curves', {})
            if curves:
                self.curves_widget.set_curves(curves)
            else:
                self.curves_widget.reset()
        else:
            # Load preset defaults
            preset = presets.get_preset(preset_key)
            self.set_adjustments(preset.get('adjustments', {}))
            curves = preset.get('curves', {})
            if curves:
                self.curves_widget.set_curves(curves)
            else:
                self.curves_widget.reset()

        self._loading_preset = False
        self._update_preset_indicator()
        self._update_preview()

    def _adjustments_equal(self, a: dict, b: dict) -> bool:
        """Compare two adjustment dicts with tolerance for float precision."""
        if set(a.keys()) != set(b.keys()):
            return False
        for key in a:
            # Use small tolerance for float comparison
            if abs(float(a[key]) - float(b[key])) > 0.001:
                return False
        return True

    def _curves_equal(self, a: dict, b: dict) -> bool:
        """Compare two curve dicts (handles tuple vs list differences)."""
        if set(a.keys()) != set(b.keys()):
            return False
        for ch in a:
            pts_a = a[ch]
            pts_b = b[ch]
            if len(pts_a) != len(pts_b):
                return False
            for (x1, y1), (x2, y2) in zip(pts_a, pts_b):
                if x1 != x2 or y1 != y2:
                    return False
        return True

    def _is_preset_modified(self, preset_key: str, use_live_values: bool = False) -> bool:
        """Check if a preset's state differs from its defaults.

        Args:
            preset_key: The preset to check
            use_live_values: If True, use current UI values instead of saved state
        """
        defaults = presets.get_preset(preset_key)

        if use_live_values:
            # Compare current UI values against preset defaults
            current_adjustments = self.get_adjustments()
            if not self._adjustments_equal(current_adjustments, defaults.get('adjustments', {})):
                return True
            current_curves = self.curves_widget.get_curves()
            if not self._curves_equal(current_curves, defaults.get('curves', {})):
                return True
            return False
        else:
            # Compare saved state against preset defaults
            if preset_key not in self._preset_states:
                return False
            saved = self._preset_states[preset_key]
            if not self._adjustments_equal(saved.get('adjustments', {}), defaults.get('adjustments', {})):
                return True
            if not self._curves_equal(saved.get('curves', {}), defaults.get('curves', {})):
                return True
            return False

    def _update_preset_indicator(self):
        """Update the preset bar to show modified state for all presets."""
        self._preset_panel.clear_all_modified()

        # Check current preset using live UI values
        if self._current_preset != 'none':
            if self._is_preset_modified(self._current_preset, use_live_values=True):
                self._preset_panel.set_modified(self._current_preset, True)

        # Check other presets using saved states
        for preset_key in self._preset_states:
            if preset_key != self._current_preset:
                if self._is_preset_modified(preset_key, use_live_values=False):
                    self._preset_panel.set_modified(preset_key, True)

    def get_preset_state(self) -> dict:
        """Get current preset state for persistence.

        Returns dict with 'active_preset' and 'preset_states'.
        """
        # Save current state before returning
        self._save_current_preset_state()
        return {
            'active_preset': self._current_preset,
            'preset_states': self._preset_states.copy(),
        }

    def set_preset_state(self, state: dict):
        """Restore preset state from persistence.

        Accepts dict with 'active_preset' and 'preset_states'.
        Also handles legacy format (preset_key, modified) for migration.
        """
        if isinstance(state, dict):
            # New format
            self._current_preset = state.get('active_preset', 'none')
            self._preset_states = state.get('preset_states', {}).copy()
        else:
            # Legacy format: (preset_key, modified) - ignore, start fresh
            self._current_preset = 'none'
            self._preset_states = {}

        # Update slider and curves preset defaults for reset button behavior
        self._update_slider_preset_defaults(self._current_preset)
        self._update_curves_preset_default(self._current_preset)
        self._preset_panel.select_preset(self._current_preset)
        self._update_preset_indicator()

    def _on_curves_changed(self):
        if not self._loading_preset:
            self._update_preset_indicator()
        self._update_preview()

    def _update_preview(self):
        if self._image is None:
            self._preview.set_pixmap(None)
            self._adjusted_image = None
            return

        # Input is float32 (0-1 range), work in float32 throughout
        img = self._image.copy()

        # 1. Exposure (apply as multiplicative factor, in stops)
        exposure = self.exposure_slider.value()
        if exposure != 0:
            factor = 2.0 ** exposure
            img = img * factor

        # 1.5. White Balance Multipliers (proper color correction)
        wb_r = self.wb_r_slider.value()
        wb_g = self.wb_g_slider.value()
        wb_b = self.wb_b_slider.value()
        if wb_r != 1.0 or wb_g != 1.0 or wb_b != 1.0:
            img[:, :, 2] = img[:, :, 2] * wb_r  # Red (BGR order)
            img[:, :, 1] = img[:, :, 1] * wb_g  # Green
            img[:, :, 0] = img[:, :, 0] * wb_b  # Blue

        # 2. Temperature (creative color grading)
        # Scale: was 0.5 per unit on 0-255, now on 0-1 range
        temperature = self.temperature_slider.value()
        if temperature != 0:
            # Temperature: positive = warm (add yellow/red), negative = cool (add blue)
            temp_adjust = temperature * (0.5 / 255.0)  # Convert to 0-1 scale
            img[:, :, 0] = img[:, :, 0] - temp_adjust  # Blue
            img[:, :, 2] = img[:, :, 2] + temp_adjust  # Red

        # 3. Contrast (S-curve around midpoint)
        contrast = self.contrast_slider.value()
        if contrast != 0:
            factor = (100 + contrast) / 100.0
            img = (img - 0.5) * factor + 0.5  # Midpoint is 0.5 in 0-1 range

        # 4. Highlights & Shadows recovery
        highlights = self.highlights_slider.value()
        shadows = self.shadows_slider.value()
        if highlights != 0 or shadows != 0:
            # Convert to luminance for masking (values are 0-1)
            lum = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]

            if highlights != 0:
                # Affect bright areas (luminance > 0.5)
                highlight_mask = np.clip((lum - 0.5) / 0.5, 0, 1)
                adjustment = -highlights * (0.5 / 255.0) * highlight_mask  # Scale to 0-1
                for c in range(3):
                    img[:, :, c] = img[:, :, c] + adjustment

            if shadows != 0:
                # Affect dark areas (luminance < 0.5)
                shadow_mask = np.clip((0.5 - lum) / 0.5, 0, 1)
                adjustment = shadows * (0.5 / 255.0) * shadow_mask  # Scale to 0-1
                for c in range(3):
                    img[:, :, c] = img[:, :, c] + adjustment

        # 5. Levels (blacks, whites, gamma)
        blacks = self.blacks_slider.value()
        whites = self.whites_slider.value()
        gamma = self.gamma_slider.value()
        if blacks != 0 or whites != 0 or gamma != 1.0:
            # Map input range to output [0, 1]
            # blacks slider: 0-100 maps to 0-1 input clipping
            # whites slider: 0-100 maps to clipping at (1 - whites/100)
            in_min = blacks / 100.0
            in_max = 1.0 - whites / 100.0
            if in_max > in_min:
                img = (img - in_min) / (in_max - in_min)
            # Apply gamma
            if gamma != 1.0:
                img = np.clip(img, 0, 1)
                img = np.power(img, 1.0 / gamma)

        # 6. Vibrance (smart saturation - less effect on already saturated colors)
        vibrance = self.vibrance_slider.value()
        if vibrance != 0:
            # Calculate saturation of each pixel (values are 0-1)
            max_rgb = np.max(img, axis=2)
            min_rgb = np.min(img, axis=2)
            current_sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0)
            # Less effect on already saturated pixels
            vibrance_mask = 1.0 - current_sat
            gray = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
            amount = vibrance / 100.0
            for c in range(3):
                diff = img[:, :, c] - gray
                img[:, :, c] = img[:, :, c] + diff * amount * vibrance_mask

        # 7. Saturation
        saturation = self.saturation_slider.value()
        if saturation != 0:
            # Convert to HSV-like: blend with grayscale
            gray = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
            gray = np.stack([gray, gray, gray], axis=2)
            factor = (100 + saturation) / 100.0
            img = gray + factor * (img - gray)

        # 8. Apply curves (fine tonal control) - works with float32 0-1
        img = self.curves_widget.apply_curves(img)

        # 9. Sharpening (applied last, after all tonal/color work)
        sharpening = self.sharpening_slider.value()
        if sharpening > 0:
            # Unsharp mask sharpening (works with float32)
            amount = sharpening / 50.0  # Scale to reasonable range
            blurred = cv2.GaussianBlur(img, (0, 0), 1.5)
            img = img + amount * (img - blurred)

        # Store result as float32 (0-1) for export
        self._adjusted_image = np.clip(img, 0, 1)

        # Convert to uint8 for display only
        display_img = (self._adjusted_image * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Set pixmap (scaling is handled by AdjustmentsPreview)
        self._preview.set_pixmap(pixmap)
