"""
SUPER NEGATIVE PROCESSING SYSTEM - Thumbnail Bar Widget

Sidebar for displaying and selecting image thumbnails.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton
)
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QPixmap, QImage
import numpy as np
import cv2

import storage
from processing import load_image, RAW_EXTENSIONS


class ThumbnailLoaderWorker(QObject):
    """Background worker for loading thumbnails."""
    thumbnailLoaded = Signal(int, object)  # index, numpy array or None
    finished = Signal()

    def __init__(self, items: list):
        """items: list of (index, path, image_hash) tuples"""
        super().__init__()
        self._items = items
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from processing import RAW_EXTENSIONS, process_negative_image

        for index, path, image_hash in self._items:
            if self._cancelled:
                break

            img = None
            is_raw = Path(path).suffix.lower() in RAW_EXTENSIONS

            # Try cached thumbnail first for display (already processed from previous session)
            if image_hash:
                cached = storage.get_storage().load_thumbnail(image_hash)
                if cached is not None:
                    self.thumbnailLoaded.emit(index, cached)
                    # For RAW files, ensure demosaiced cache is warmed even if thumbnail exists
                    # This way navigation to this image will be instant
                    if is_raw and not storage.get_storage().has_raw_cache(image_hash):
                        try:
                            load_image(path)  # Triggers demosaic + cache
                        except (FileNotFoundError, ImportError):
                            pass
                    continue

            # Load full image (demosaics RAW if needed, caches result)
            try:
                img = load_image(path)
                # Run full processing pipeline: crop, rotate, invert
                processed = process_negative_image(img)
                self.thumbnailLoaded.emit(index, processed)
            except (FileNotFoundError, ImportError):
                self.thumbnailLoaded.emit(index, None)

        self.finished.emit()


class ThumbnailItem(QLabel):
    """A clickable thumbnail image."""

    clicked = Signal(int)

    def __init__(self, index: int, path: str, image_hash: str = None, defer_load: bool = False):
        super().__init__()
        self.index = index
        self.path = path
        self.image_hash = image_hash
        self._is_favorite = False
        self._has_processed_image = False  # Track if updated by processing
        self.setFixedSize(104, 78)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #2a2a2a; border: 2px solid #444; }")
        self.setCursor(Qt.PointingHandCursor)

        # Favorite star label (top-right corner overlay)
        self._star_label = QLabel("★", self)
        self._star_label.setFixedSize(18, 18)
        self._star_label.setAlignment(Qt.AlignCenter)
        self._star_label.move(84, 4)
        self._star_label.hide()
        self._update_star_style()

        if defer_load:
            # Show placeholder, load later via async worker
            self.setText("...")
            return

        # Synchronous load (for single file additions)
        # Try to load cached thumbnail first
        if image_hash:
            cached = storage.get_storage().load_thumbnail(image_hash)
            if cached is not None:
                self.set_image(cached)
                return

        # Fall back to loading original negative
        try:
            img = load_image(path)
            self.set_image(img)
        except (FileNotFoundError, ImportError):
            self.setText(Path(path).name[:10])

    def set_image(self, img: np.ndarray, from_processing: bool = False):
        """Update thumbnail with a BGR numpy array (e.g., inverted negative).

        Args:
            img: BGR numpy array (float32 0-1 or uint8 0-255) to display
            from_processing: If True, marks this as a processed image that should
                           not be overwritten by deferred loading
        """
        if img is None:
            self.setText("...")
            return

        if from_processing:
            self._has_processed_image = True

        # Convert float32 (0-1) to uint8 (0-255) for QImage
        if img.dtype == np.float32:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        # Scale to fit (leaving room for border)
        scale = min(100 / w, 74 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_scaled = cv2.resize(img_rgb, (new_w, new_h))
        qimg = QImage(img_scaled.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, event):
        self.clicked.emit(self.index)

    def set_selected(self, selected: bool):
        if selected:
            self.setStyleSheet("QLabel { background-color: #2a2a2a; border: 2px solid #e67e22; }")
        else:
            self.setStyleSheet("QLabel { background-color: #2a2a2a; border: 2px solid #444; }")

    def set_favorite(self, is_favorite: bool):
        """Set the favorite state and update the star display."""
        self._is_favorite = is_favorite
        if is_favorite:
            self._star_label.show()
        else:
            self._star_label.hide()

    def is_favorite(self) -> bool:
        """Return whether this thumbnail is marked as favorite."""
        return self._is_favorite

    def _update_star_style(self):
        """Update star label appearance."""
        self._star_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 0.6);
                color: #f1c40f;
                border: none;
                border-radius: 3px;
                font-size: 12px;
            }
        """)


class ThumbnailBar(QWidget):
    """Scrollable thumbnail bar with favorites filter."""

    imageSelected = Signal(int)
    favoriteToggled = Signal(int, bool)  # Emits (index, is_favorite)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(130)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header with favorites toggle button
        header = QWidget()
        header.setFixedHeight(32)
        header.setStyleSheet("background-color: #3a3a3a;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 4, 5, 4)

        header_layout.addStretch()
        self._favorites_btn = QPushButton("★")
        self._favorites_btn.setFixedSize(24, 24)
        self._favorites_btn.setCheckable(True)
        self._favorites_btn.setToolTip("Show favorites only (⇧⌘F to toggle filter)")
        self._favorites_btn.clicked.connect(self._on_favorites_filter_toggled)
        self._update_favorites_btn_style()
        header_layout.addWidget(self._favorites_btn)

        main_layout.addWidget(header)

        # Scroll area for thumbnails
        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background-color: #3a3a3a; }")

        self.container = QWidget()
        self.container.setFixedWidth(110)
        self._thumb_layout = QVBoxLayout(self.container)
        self._thumb_layout.setContentsMargins(3, 5, 3, 5)
        self._thumb_layout.setSpacing(5)
        self._thumb_layout.addStretch()

        self._scroll.setWidget(self.container)
        main_layout.addWidget(self._scroll)

        self.thumbnails = []
        self.current_index = -1
        self._show_favorites_only = False
        self._favorite_hashes = set()  # Set of favorite image hashes
        self._loader_thread = None
        self._loader_worker = None
        self._shutting_down = False

    def _cancel_pending_loads(self):
        """Cancel any pending thumbnail loading."""
        self._shutting_down = True
        if self._loader_worker:
            self._loader_worker.cancel()
        if self._loader_thread and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait(2000)  # Wait up to 2s for thread to finish
        self._loader_thread = None
        self._loader_worker = None

    def set_favorite_hashes(self, hashes: set):
        """Set the favorite hashes and update thumbnail display."""
        self._favorite_hashes = hashes
        for thumb in self.thumbnails:
            if thumb.image_hash:
                thumb.set_favorite(thumb.image_hash in hashes)
        self._apply_filter()

    def set_files(self, paths: list, hashes: list = None):
        """Set the file list and create thumbnails.

        Args:
            paths: List of image file paths
            hashes: Optional list of pre-computed SHA hashes for cache lookup
        """
        # Cancel any pending loads
        self._cancel_pending_loads()
        self._shutting_down = False  # Reset for new load

        # Clear existing
        for thumb in self.thumbnails:
            thumb.deleteLater()
        self.thumbnails = []

        # Remove stretch
        while self._thumb_layout.count():
            item = self._thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add thumbnails with deferred loading (show placeholder initially)
        items_to_load = []
        for i, path in enumerate(paths):
            image_hash = hashes[i] if hashes and i < len(hashes) else None
            thumb = ThumbnailItem(i, path, image_hash, defer_load=True)
            thumb.clicked.connect(self._on_thumb_clicked)
            # Set favorite state if hash is known
            if image_hash and image_hash in self._favorite_hashes:
                thumb.set_favorite(True)
            self._thumb_layout.addWidget(thumb)
            self.thumbnails.append(thumb)
            items_to_load.append((i, path, image_hash))

        self._thumb_layout.addStretch()
        self.current_index = -1
        self._apply_filter()

        # Start background loading
        if items_to_load:
            self._loader_thread = QThread()
            self._loader_worker = ThumbnailLoaderWorker(items_to_load)
            self._loader_worker.moveToThread(self._loader_thread)
            self._loader_thread.started.connect(self._loader_worker.run)
            self._loader_worker.thumbnailLoaded.connect(self._on_thumbnail_loaded)
            self._loader_worker.finished.connect(self._loader_thread.quit)
            self._loader_thread.start()

    def _on_thumbnail_loaded(self, index: int, img):
        """Handle thumbnail loaded from background thread."""
        # Ignore signals during shutdown to avoid accessing destroyed widgets
        if self._shutting_down:
            return
        if index < len(self.thumbnails):
            thumb = self.thumbnails[index]
            # Don't overwrite if processing already set the inverted image
            if thumb._has_processed_image:
                return
            if img is not None:
                thumb.set_image(img)
            else:
                thumb.setText(Path(thumb.path).name[:10])

    def _on_thumb_clicked(self, index: int):
        self.select_index(index)
        self.imageSelected.emit(index)

    def _on_favorites_filter_toggled(self, checked: bool):
        """Handle favorites filter toggle."""
        self._show_favorites_only = checked
        self._update_favorites_btn_style()
        self._apply_filter()

    def toggle_favorites_filter(self):
        """Programmatically toggle the favorites filter."""
        self._favorites_btn.setChecked(not self._favorites_btn.isChecked())
        self._on_favorites_filter_toggled(self._favorites_btn.isChecked())

    def _update_favorites_btn_style(self):
        """Update favorites button appearance."""
        if self._favorites_btn.isChecked():
            self._favorites_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f1c40f;
                    color: #1a1a1a;
                    border: none;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #f39c12;
                }
            """)
        else:
            self._favorites_btn.setStyleSheet("""
                QPushButton {
                    background-color: #333;
                    color: #888;
                    border: none;
                    border-radius: 4px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #444;
                    color: #f1c40f;
                }
            """)

    def _apply_filter(self):
        """Show/hide thumbnails based on favorites filter."""
        for thumb in self.thumbnails:
            if self._show_favorites_only:
                thumb.setVisible(thumb.is_favorite())
            else:
                thumb.setVisible(True)

    def toggle_favorite(self, index: int) -> bool:
        """Toggle favorite for thumbnail at index. Returns new favorite state."""
        if 0 <= index < len(self.thumbnails):
            thumb = self.thumbnails[index]
            new_state = not thumb.is_favorite()
            thumb.set_favorite(new_state)
            # Update hash set
            if thumb.image_hash:
                if new_state:
                    self._favorite_hashes.add(thumb.image_hash)
                else:
                    self._favorite_hashes.discard(thumb.image_hash)
            self._apply_filter()
            self.favoriteToggled.emit(index, new_state)
            return new_state
        return False

    def is_filtering_favorites(self) -> bool:
        """Return whether the favorites filter is active."""
        return self._show_favorites_only

    def get_prev_visible_index(self, current: int) -> int:
        """Get the previous visible thumbnail index, or -1 if none."""
        for i in range(current - 1, -1, -1):
            if i < len(self.thumbnails) and self.thumbnails[i].isVisible():
                return i
        return -1

    def get_next_visible_index(self, current: int) -> int:
        """Get the next visible thumbnail index, or -1 if none."""
        for i in range(current + 1, len(self.thumbnails)):
            if self.thumbnails[i].isVisible():
                return i
        return -1

    def select_index(self, index: int):
        """Highlight the selected thumbnail."""
        if self.current_index >= 0 and self.current_index < len(self.thumbnails):
            self.thumbnails[self.current_index].set_selected(False)

        self.current_index = index

        if index >= 0 and index < len(self.thumbnails):
            self.thumbnails[index].set_selected(True)
            # Scroll to make visible
            self._scroll.ensureWidgetVisible(self.thumbnails[index])

    def update_thumbnail(self, index: int, img: np.ndarray):
        """Update a specific thumbnail with a new image from processing."""
        if 0 <= index < len(self.thumbnails):
            self.thumbnails[index].set_image(img, from_processing=True)
