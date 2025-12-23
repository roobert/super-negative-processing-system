"""
SUPER NEGATIVE PROCESSING SYSTEM - Preset Bar Widget

Sidebar for displaying and selecting film stock presets with live previews.
"""

from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QScrollArea, QHBoxLayout, QVBoxLayout, QPushButton
)
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint
from PySide6.QtGui import QPixmap, QImage, QDrag, QPainter, QPen, QColor
from scipy.interpolate import PchipInterpolator
import numpy as np
import cv2

import presets
import storage


class PresetThumbnailItem(QFrame):
    """A clickable thumbnail showing an image with a preset applied, with name label."""

    clicked = Signal(str)  # Emits preset key
    favoriteToggled = Signal(str, bool)  # Emits (preset_key, is_favorite)

    def __init__(self, preset_key: str, preset_name: str):
        super().__init__()
        self.preset_key = preset_key
        self.preset_name = preset_name
        self._is_modified = False
        self._is_selected = False
        self._is_favorite = False
        self._number = 0  # 0 = no number, 1-9 = shortcut number
        self._drag_start_pos = None  # For drag-and-drop

        self.setFixedSize(260, 195)
        self.setCursor(Qt.PointingHandCursor)
        self.setFrameStyle(QFrame.Box)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Thumbnail image container (for overlays)
        thumb_container = QWidget()
        thumb_container.setFixedSize(250, 168)
        thumb_layout = QVBoxLayout(thumb_container)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(0)

        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(250, 168)
        self._thumb_label.setAlignment(Qt.AlignCenter)
        self._thumb_label.setText("...")
        thumb_layout.addWidget(self._thumb_label)

        layout.addWidget(thumb_container)

        # Favorite star button (top-right corner overlay) - not shown for 'none'
        self._star_btn = QPushButton("☆", self)
        self._star_btn.setFixedSize(28, 28)
        self._star_btn.setCursor(Qt.PointingHandCursor)
        self._star_btn.clicked.connect(self._on_star_clicked)
        self._star_btn.move(228, 6)
        if preset_key == 'none':
            self._star_btn.hide()
        else:
            self._update_star_style()

        # Number label (top-left corner overlay)
        self._number_label = QLabel(self)
        self._number_label.setFixedSize(24, 24)
        self._number_label.setAlignment(Qt.AlignCenter)
        self._number_label.move(8, 8)
        self._update_number_label()

        # Name label
        self._name_label = QLabel(preset_name)
        self._name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._name_label)

        self._update_style()

    def _update_style(self):
        """Update visual style based on selection and modified state."""
        if self._is_selected:
            self.setStyleSheet("PresetThumbnailItem { background-color: #2a2a2a; border: 2px solid #e67e22; }")
        else:
            self.setStyleSheet("PresetThumbnailItem { background-color: #2a2a2a; border: 2px solid #444; }")
        self._update_name_label()

    def _update_name_label(self):
        """Update the name label with modified indicator if needed."""
        if self._is_modified:
            self._name_label.setText(f"{self.preset_name} (adjusted)")
            self._name_label.setStyleSheet("color: #e67e22; font-size: 11px; font-style: italic;")
            self.setToolTip(f"{self.preset_name}\nAdjusted - click to reset to preset defaults")
        else:
            self._name_label.setText(self.preset_name)
            self._name_label.setStyleSheet("color: #ccc; font-size: 11px;")
            self.setToolTip(self.preset_name)

    def _update_star_style(self):
        """Update star button appearance based on favorite state."""
        if self._is_favorite:
            self._star_btn.setText("★")
            self._star_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 0.6);
                    color: #f1c40f;
                    border: none;
                    border-radius: 4px;
                    font-size: 18px;
                }
                QPushButton:hover {
                    background-color: rgba(0, 0, 0, 0.8);
                }
            """)
        else:
            self._star_btn.setText("☆")
            self._star_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 0.4);
                    color: #888;
                    border: none;
                    border-radius: 4px;
                    font-size: 18px;
                }
                QPushButton:hover {
                    background-color: rgba(0, 0, 0, 0.7);
                    color: #f1c40f;
                }
            """)

    def _update_number_label(self):
        """Update number label visibility and text."""
        if self._number > 0:
            self._number_label.setText(str(self._number))
            self._number_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(230, 126, 34, 0.9);
                    color: white;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
            self._number_label.show()
        else:
            self._number_label.hide()

    def _on_star_clicked(self):
        """Handle star button click - toggle favorite state."""
        self._is_favorite = not self._is_favorite
        self._update_star_style()
        self.favoriteToggled.emit(self.preset_key, self._is_favorite)

    def set_favorite(self, is_favorite: bool):
        """Set the favorite state (without emitting signal)."""
        self._is_favorite = is_favorite
        self._update_star_style()

    def set_number(self, number: int):
        """Set the shortcut number (1-9) or 0 to hide."""
        self._number = number
        self._update_number_label()

    def set_image(self, img: np.ndarray):
        """Update thumbnail with a BGR numpy array (preset-applied image)."""
        if img is None:
            self._thumb_label.setText("...")
            self._thumb_label.setPixmap(QPixmap())
            return

        # Convert float32 (0-1) to uint8 (0-255) for QImage
        if img.dtype == np.float32:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        # Scale to fit the thumbnail area (250x168)
        scale = min(246 / w, 164 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_scaled = cv2.resize(img_rgb, (new_w, new_h))
        qimg = QImage(img_scaled.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self._thumb_label.setPixmap(QPixmap.fromImage(qimg))

    def set_modified(self, modified: bool):
        """Set the modified flag (shows indicator when preset has been tweaked)."""
        self._is_modified = modified
        self._update_name_label()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.preset_key != 'none':
            self._drag_start_pos = event.position().toPoint()
        self.clicked.emit(self.preset_key)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton) or self._drag_start_pos is None:
            return
        if self.preset_key == 'none':
            return  # Can't drag 'none'

        # Check if we've moved far enough to start a drag
        if (event.position().toPoint() - self._drag_start_pos).manhattanLength() < 10:
            return

        # Start drag
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.preset_key)
        drag.setMimeData(mime_data)

        # Create a semi-transparent pixmap of this widget for drag visual
        pixmap = self.grab()
        pixmap.setDevicePixelRatio(2.0)  # For retina displays
        drag.setPixmap(pixmap.scaled(200, 147, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        drag.setHotSpot(QPoint(100, 73))

        getattr(drag, 'exec')(Qt.MoveAction)
        self._drag_start_pos = None

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None

    def set_selected(self, selected: bool):
        self._is_selected = selected
        self._update_style()


class PresetBarContainer(QWidget):
    """Container widget that accepts drops for preset reordering."""

    dropRequested = Signal(str, int)  # Emits (preset_key, target_index)

    def __init__(self, layout: QVBoxLayout):
        super().__init__()
        self._layout = layout
        self.setAcceptDrops(True)
        self._drop_indicator_index = -1

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if not event.mimeData().hasText():
            return

        # Find which thumbnail we're hovering over
        pos = event.position().toPoint()
        target_index = self._get_drop_index(pos.y())

        if target_index != self._drop_indicator_index:
            self._drop_indicator_index = target_index
            self.update()

        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._drop_indicator_index = -1
        self.update()

    def dropEvent(self, event):
        if not event.mimeData().hasText():
            return

        preset_key = event.mimeData().text()
        target_index = self._get_drop_index(event.position().toPoint().y())

        self._drop_indicator_index = -1
        self.update()

        # Don't allow dropping at index 0 (that's 'none')
        if target_index <= 0:
            target_index = 1

        self.dropRequested.emit(preset_key, target_index)
        event.acceptProposedAction()

    def _get_drop_index(self, y: int) -> int:
        """Determine drop index based on y position."""
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                widget_y = widget.y()
                widget_h = widget.height()
                # If we're in the top half of this widget, insert before it
                if y < widget_y + widget_h // 2:
                    return i
        # If we're past all widgets, insert at end (before stretch)
        return self._layout.count() - 1  # -1 for the stretch

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_indicator_index >= 0:
            painter = QPainter(self)
            painter.setPen(QPen(QColor("#e67e22"), 3))

            # Draw line at the drop position
            y = 0
            if self._drop_indicator_index < self._layout.count():
                item = self._layout.itemAt(self._drop_indicator_index)
                if item and item.widget():
                    y = item.widget().y() - 2
            painter.drawLine(5, y, self.width() - 5, y)
            painter.end()


class PresetBar(QScrollArea):
    """Scrollable bar showing preset thumbnails with live previews.

    Supports two layout modes:
    - List mode (default): Single column vertical list
    - Grid mode: Multiple columns to fill available width
    """

    presetSelected = Signal(str)  # Emits preset key

    THUMB_WIDTH = 260  # Thumbnail width
    THUMB_SPACING = 5  # Spacing between thumbnails

    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; background-color: #1a1a1a; }")

        self._grid_mode = False
        self._available_width = 265  # Will be updated

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(3, 5, 3, 5)
        self.layout.setSpacing(5)

        self.container = PresetBarContainer(self.layout)
        self.container.setFixedWidth(265)
        self.container.setLayout(self.layout)
        self.container.dropRequested.connect(self._on_drop_requested)

        self.setWidget(self.container)
        self.thumbnails = {}  # preset_key -> PresetThumbnailItem
        self.current_key = 'none'
        self._base_image = None  # The image to apply presets to
        self._current_image_hash = None  # Hash for caching preset thumbnails
        self._favorites = set()  # Set of favorite preset keys
        self._custom_order = []  # User's custom order (excluding 'none')
        self._ordered_keys = []  # Current display order of preset keys

        # Load favorites and custom order from storage
        self._favorites = set(storage.get_storage().get_favorite_presets())
        self._custom_order = storage.get_storage().get_preset_order()

        self._setup_thumbnails()

    def set_grid_mode(self, enabled: bool):
        """Switch between list mode and grid mode."""
        if self._grid_mode == enabled:
            return
        self._grid_mode = enabled
        self._reorder_thumbnails()

    def setFixedWidth(self, width: int):
        """Override to track available width for grid layout."""
        super().setFixedWidth(width)
        self._available_width = width - 15  # Account for scrollbar and margins
        if hasattr(self, 'container'):
            self.container.setFixedWidth(self._available_width)
            if self._grid_mode:
                self._reorder_thumbnails()

    def _setup_thumbnails(self):
        """Create thumbnail items for all presets."""
        preset_list = presets.get_preset_list()
        for key, name, description in preset_list:
            thumb = PresetThumbnailItem(key, name)
            thumb.setToolTip(f"{name}\n{description}")
            thumb.clicked.connect(self._on_thumb_clicked)
            thumb.favoriteToggled.connect(self._on_favorite_toggled)
            thumb.set_favorite(key in self._favorites)
            self.thumbnails[key] = thumb

        # Add all thumbnails in sorted order
        self._reorder_thumbnails()

        # Select 'none' by default
        if 'none' in self.thumbnails:
            self.thumbnails['none'].set_selected(True)

    def _get_sorted_keys(self) -> list:
        """Get preset keys sorted: 'none' first, then by custom order/favorites."""
        preset_list = presets.get_preset_list()
        all_keys = [key for key, _, _ in preset_list]

        # 'none' is always first
        result = ['none'] if 'none' in all_keys else []

        # If we have a custom order, use it (it already excludes 'none')
        if self._custom_order:
            # Use custom order, but only include keys that still exist
            for key in self._custom_order:
                if key in all_keys and key != 'none':
                    result.append(key)
            # Add any new presets that aren't in custom order yet
            for key in all_keys:
                if key not in result and key != 'none':
                    result.append(key)
        else:
            # No custom order: favorites first, then rest
            favorite_keys = [k for k in all_keys if k in self._favorites and k != 'none']
            result.extend(favorite_keys)
            non_favorite_keys = [k for k in all_keys if k not in self._favorites and k != 'none']
            result.extend(non_favorite_keys)

        return result

    def _reorder_thumbnails(self):
        """Reorder thumbnails in the layout based on favorites and current mode."""
        # Remove all widgets/layouts from layout
        while self.layout.count():
            item = self.layout.takeAt(0)
            # If it's a layout (row), remove widgets from it too
            if item.layout():
                row_layout = item.layout()
                while row_layout.count():
                    row_layout.takeAt(0)

        # Get sorted order
        self._ordered_keys = self._get_sorted_keys()

        if self._grid_mode:
            # Grid layout: calculate columns based on available width
            cols = max(1, (self._available_width + self.THUMB_SPACING) // (self.THUMB_WIDTH + self.THUMB_SPACING))
            row_layout = None
            col_idx = 0

            for i, key in enumerate(self._ordered_keys):
                thumb = self.thumbnails[key]
                # Assign number 1-9 to first 9 presets
                if i < 9:
                    thumb.set_number(i + 1)
                else:
                    thumb.set_number(0)

                # Start new row if needed
                if col_idx == 0:
                    row_layout = QHBoxLayout()
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(self.THUMB_SPACING)

                row_layout.addWidget(thumb)
                col_idx += 1

                # If row is full, add to main layout
                if col_idx >= cols:
                    row_layout.addStretch()
                    self.layout.addLayout(row_layout)
                    col_idx = 0

            # Add any incomplete row
            if col_idx > 0 and row_layout:
                row_layout.addStretch()
                self.layout.addLayout(row_layout)
        else:
            # List layout: single column
            for i, key in enumerate(self._ordered_keys):
                thumb = self.thumbnails[key]
                self.layout.addWidget(thumb)
                # Assign number 1-9 to first 9 presets
                if i < 9:
                    thumb.set_number(i + 1)
                else:
                    thumb.set_number(0)

        self.layout.addStretch()

    def _on_favorite_toggled(self, preset_key: str, is_favorite: bool):
        """Handle favorite toggle from a thumbnail."""
        if is_favorite:
            self._favorites.add(preset_key)
        else:
            self._favorites.discard(preset_key)

        # Save to storage
        storage.get_storage().set_favorite_presets(list(self._favorites))

        # When toggling favorites, update custom order to reflect new position
        self._update_custom_order_from_current()

        # Reorder thumbnails
        self._reorder_thumbnails()

    def _on_drop_requested(self, preset_key: str, target_index: int):
        """Handle a preset being dropped at a new position."""
        if preset_key == 'none' or preset_key not in self.thumbnails:
            return

        # Get current order and find where the preset is
        current_order = self._ordered_keys.copy()
        if preset_key not in current_order:
            return

        old_index = current_order.index(preset_key)

        # Remove from old position
        current_order.remove(preset_key)

        # Adjust target index if needed (account for removal)
        if old_index < target_index:
            target_index -= 1

        # Clamp target index (can't go before 'none' at index 0)
        target_index = max(1, min(target_index, len(current_order)))

        # Insert at new position
        current_order.insert(target_index, preset_key)

        # Save new order (excluding 'none')
        self._custom_order = [k for k in current_order if k != 'none']
        storage.get_storage().set_preset_order(self._custom_order)

        # Reorder thumbnails
        self._reorder_thumbnails()

    def _update_custom_order_from_current(self):
        """Save current visual order as custom order."""
        self._custom_order = [k for k in self._ordered_keys if k != 'none']
        storage.get_storage().set_preset_order(self._custom_order)

    def get_preset_key_by_number(self, number: int) -> str:
        """Get the preset key for a given shortcut number (1-9)."""
        if 1 <= number <= 9 and number <= len(self._ordered_keys):
            return self._ordered_keys[number - 1]
        return None

    def move_current_preset_up(self):
        """Move the currently selected preset up in the list."""
        self._move_current_preset(-1)

    def move_current_preset_down(self):
        """Move the currently selected preset down in the list."""
        self._move_current_preset(1)

    def select_previous_preset(self) -> str:
        """Select the previous preset in the list. Returns the new preset key."""
        if not self._ordered_keys or self.current_key not in self._ordered_keys:
            return None
        current_index = self._ordered_keys.index(self.current_key)
        if current_index > 0:
            new_key = self._ordered_keys[current_index - 1]
            self.select_preset(new_key)
            self.presetSelected.emit(new_key)
            return new_key
        return None

    def select_next_preset(self) -> str:
        """Select the next preset in the list. Returns the new preset key."""
        if not self._ordered_keys or self.current_key not in self._ordered_keys:
            return None
        current_index = self._ordered_keys.index(self.current_key)
        if current_index < len(self._ordered_keys) - 1:
            new_key = self._ordered_keys[current_index + 1]
            self.select_preset(new_key)
            self.presetSelected.emit(new_key)
            return new_key
        return None

    def _move_current_preset(self, direction: int):
        """Move current preset by direction (-1 = up, 1 = down)."""
        if self.current_key == 'none' or self.current_key not in self._ordered_keys:
            return  # Can't move 'none'

        current_index = self._ordered_keys.index(self.current_key)
        new_index = current_index + direction

        # Can't move above index 1 (index 0 is 'none') or below end
        if new_index < 1 or new_index >= len(self._ordered_keys):
            return

        # Swap in ordered keys
        order = self._ordered_keys.copy()
        order[current_index], order[new_index] = order[new_index], order[current_index]

        # Save new order (excluding 'none')
        self._custom_order = [k for k in order if k != 'none']
        storage.get_storage().set_preset_order(self._custom_order)

        # Reorder thumbnails
        self._reorder_thumbnails()

        # Ensure the moved preset is visible
        if self.current_key in self.thumbnails:
            self.ensureWidgetVisible(self.thumbnails[self.current_key])

    def _on_thumb_clicked(self, preset_key: str):
        self.select_preset(preset_key)
        self.presetSelected.emit(preset_key)

    def select_preset(self, preset_key: str):
        """Highlight the selected preset thumbnail."""
        if self.current_key in self.thumbnails:
            self.thumbnails[self.current_key].set_selected(False)

        self.current_key = preset_key

        if preset_key in self.thumbnails:
            self.thumbnails[preset_key].set_selected(True)
            self.ensureWidgetVisible(self.thumbnails[preset_key])

    def set_base_image(self, img: np.ndarray, image_hash: str = None):
        """Set the base image and generate all preset preview thumbnails.

        Args:
            img: The base image (inverted negative) to apply presets to.
            image_hash: Optional hash for caching preset thumbnails.
        """
        self._base_image = img.copy() if img is not None else None
        self._current_image_hash = image_hash
        self._generate_previews()

    def _generate_previews(self):
        """Generate thumbnail previews for all presets, using cache when available."""
        if self._base_image is None:
            for thumb in self.thumbnails.values():
                thumb.set_image(None)
            return

        # Try to load cached thumbnails first
        cached = {}
        if self._current_image_hash:
            cached = storage.get_storage().load_all_preset_thumbnails(self._current_image_hash)

        # Downscale base image ONCE for all preset previews (massive speedup for large images)
        h, w = self._base_image.shape[:2]
        max_thumb_dim = 260  # Target thumbnail dimension
        if max(h, w) > max_thumb_dim:
            scale = max_thumb_dim / max(h, w)
            thumb_base = cv2.resize(self._base_image, (int(w * scale), int(h * scale)),
                                    interpolation=cv2.INTER_AREA)
        else:
            thumb_base = self._base_image

        # Generate preview for each preset
        for key, thumb in self.thumbnails.items():
            if key in cached:
                # Use cached thumbnail
                thumb.set_image(cached[key])
            else:
                # Generate and cache (using downscaled image)
                preset = presets.get_preset(key)
                preview = self._apply_preset_to_image(thumb_base, preset)
                thumb.set_image(preview)
                # Save to cache if we have a hash
                if self._current_image_hash and preview is not None:
                    storage.get_storage().save_preset_thumbnail(
                        self._current_image_hash, key, preview
                    )

    def _apply_preset_to_image(self, img: np.ndarray, preset: dict) -> np.ndarray:
        """Apply a preset's adjustments and curves to an image.

        This is a simplified version of the full adjustment pipeline,
        optimized for thumbnail generation.

        Handles both float32 (0-1) and uint8 (0-255) input.
        Always outputs uint8 for thumbnail storage.
        """
        if img is None:
            return None

        adj = preset.get('adjustments', {})
        curves = preset.get('curves', {})

        # Determine if input is float32 (0-1) or uint8 (0-255)
        is_float_input = img.dtype == np.float32

        # Work with float for precision
        result = img.astype(np.float32)
        if not is_float_input:
            # Convert 0-255 to 0-1 for unified processing
            result = result / 255.0

        # Now working in 0-1 range

        # 1. Exposure
        exposure = adj.get('exposure', 0)
        if exposure != 0:
            factor = 2.0 ** exposure
            result = result * factor

        # 1.5. White Balance Multipliers
        wb_r = adj.get('wb_r', 1.0)
        wb_g = adj.get('wb_g', 1.0)
        wb_b = adj.get('wb_b', 1.0)
        if wb_r != 1.0 or wb_g != 1.0 or wb_b != 1.0:
            result[:, :, 2] = result[:, :, 2] * wb_r  # Red (BGR order)
            result[:, :, 1] = result[:, :, 1] * wb_g  # Green
            result[:, :, 0] = result[:, :, 0] * wb_b  # Blue

        # 2. Temperature (scaled to 0-1 range)
        temperature = adj.get('temperature', 0)
        if temperature != 0:
            temp_adjust = temperature * (0.5 / 255.0)
            result[:, :, 0] = result[:, :, 0] - temp_adjust  # Blue
            result[:, :, 2] = result[:, :, 2] + temp_adjust  # Red

        # 3. Contrast (midpoint is 0.5)
        contrast = adj.get('contrast', 0)
        if contrast != 0:
            factor = (100 + contrast) / 100.0
            result = (result - 0.5) * factor + 0.5

        # 4. Highlights & Shadows
        highlights = adj.get('highlights', 0)
        shadows = adj.get('shadows', 0)
        if highlights != 0 or shadows != 0:
            lum = 0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]
            if highlights != 0:
                highlight_mask = np.clip((lum - 0.5) / 0.5, 0, 1)
                adjustment = -highlights * (0.5 / 255.0) * highlight_mask
                for c in range(3):
                    result[:, :, c] = result[:, :, c] + adjustment
            if shadows != 0:
                shadow_mask = np.clip((0.5 - lum) / 0.5, 0, 1)
                adjustment = shadows * (0.5 / 255.0) * shadow_mask
                for c in range(3):
                    result[:, :, c] = result[:, :, c] + adjustment

        # 5. Levels (blacks, whites, gamma) - scaled to 0-1
        blacks = adj.get('blacks', 0)
        whites = adj.get('whites', 0)
        gamma = adj.get('gamma', 1.0)
        if blacks != 0 or whites != 0 or gamma != 1.0:
            in_min = blacks / 100.0
            in_max = 1.0 - whites / 100.0
            if in_max > in_min:
                result = (result - in_min) / (in_max - in_min)
            if gamma != 1.0:
                result = np.clip(result, 0, 1)
                result = np.power(result, 1.0 / gamma)

        # 6. Vibrance
        vibrance = adj.get('vibrance', 0)
        if vibrance != 0:
            max_rgb = np.max(result, axis=2)
            min_rgb = np.min(result, axis=2)
            current_sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0)
            vibrance_mask = 1.0 - current_sat
            gray = 0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]
            amount = vibrance / 100.0
            for c in range(3):
                diff = result[:, :, c] - gray
                result[:, :, c] = result[:, :, c] + diff * amount * vibrance_mask

        # 7. Saturation
        saturation = adj.get('saturation', 0)
        if saturation != 0:
            gray = 0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]
            gray_stack = np.stack([gray, gray, gray], axis=2)
            factor = (100 + saturation) / 100.0
            result = gray_stack + factor * (result - gray_stack)

        # Convert back to uint8 (0-255) for thumbnail storage
        result = np.clip(result, 0, 1)
        result = (result * 255).astype(np.uint8)

        # 8. Apply curves (works on uint8)
        result = self._apply_curves(result, curves)

        return result

    def _apply_curves(self, img: np.ndarray, curves: dict) -> np.ndarray:
        """Apply curves to an image."""
        if not curves:
            return img

        result = img.copy()

        # Apply per-channel curves (BGR order in OpenCV)
        for i, channel in enumerate(['b', 'g', 'r']):
            if channel in curves:
                lut = self._build_lut(curves[channel])
                if not np.array_equal(lut, np.arange(256)):
                    result[:, :, i] = lut[result[:, :, i]]

        # Apply RGB (master) curve to all channels
        if 'rgb' in curves:
            lut = self._build_lut(curves['rgb'])
            if not np.array_equal(lut, np.arange(256)):
                result = lut[result]

        return result

    def _build_lut(self, points: list) -> np.ndarray:
        """Build 256-element LUT from control points using PCHIP interpolation."""
        points = sorted(points, key=lambda p: p[0])

        if len(points) < 2:
            return np.arange(256, dtype=np.uint8)

        # Remove duplicate x values
        unique_points = {}
        for p in points:
            unique_points[p[0]] = p[1]
        points = sorted(unique_points.items())

        if len(points) < 2:
            return np.arange(256, dtype=np.uint8)

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        interp = PchipInterpolator(xs, ys)
        lut = interp(np.arange(256))
        return np.clip(lut, 0, 255).astype(np.uint8)

    def set_modified(self, preset_key: str, modified: bool):
        """Mark a preset as modified (user tweaked it after loading)."""
        if preset_key in self.thumbnails:
            self.thumbnails[preset_key].set_modified(modified)

    def clear_all_modified(self):
        """Clear the modified flag from all presets."""
        for thumb in self.thumbnails.values():
            thumb.set_modified(False)
