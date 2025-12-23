#!/usr/bin/env python3
"""
SUPER NEGATIVE PROCESSING SYSTEM - GUI Application

PySide6 application for interactive frame detection with live preview.
"""

import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

import sys
import math
import hashlib
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QPushButton, QFileDialog, QGroupBox,
    QSpinBox, QDoubleSpinBox, QSizePolicy, QScrollArea, QFrame, QCheckBox,
    QStackedWidget, QTabBar, QButtonGroup, QDialog, QComboBox, QDialogButtonBox
)
from scipy.interpolate import PchipInterpolator
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, QPropertyAnimation, QEasingCurve, QMimeData, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QAction, QKeySequence, QPalette, QColor, QPainter, QPen, QFont, QDrag, QShortcut

import presets
import storage
from processing import load_image, RAW_EXTENSIONS
from auto_rotate import detect_auto_rotation, RotationConfidence


class KeybindingsDialog(QDialog):
    """Dialog showing all keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(750)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Define keybindings by category (organized for two columns)
        left_column = [
            ("Navigation", [
                ("Up / Down", "Previous / Next image"),
                ("Left / Right", "Rotate 90° CCW / CW"),
                ("Tab", "Switch between Detection/Development"),
            ]),
            ("Presets", [
                ("1-9", "Apply preset by number"),
            ]),
            ("View", [
                ("F", "Toggle fullscreen (hide panels)"),
            ]),
        ]

        right_column = [
            ("Panels", [
                ("` / \u00a7", "Toggle presets panel"),
                ("A / ~ / \u00b1", "Toggle adjustments panel"),
            ]),
            ("Crop Mode", [
                ("C", "Toggle crop mode"),
                ("Cmd+Arrow", "Top / Right edge (1px)"),
                ("Option+Arrow", "Bottom / Left edge (1px)"),
                ("+Shift", "Move 5px instead of 1px"),
            ]),
            ("File", [
                ("Cmd+O", "Open file"),
                ("Cmd+S", "Export frame"),
                (",", "Settings"),
                ("?", "Show this help"),
            ]),
            ("Zoom & Pan", [
                ("Scroll", "Zoom in/out"),
                ("Drag", "Pan image"),
                ("Double-click", "Reset view"),
            ]),
        ]

        # Two-column layout
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(20)

        for column_data in [left_column, right_column]:
            column_widget = QWidget()
            column_layout = QVBoxLayout(column_widget)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(15)

            for category, bindings in column_data:
                group = QGroupBox(category)
                group_layout = QVBoxLayout(group)
                group_layout.setSpacing(4)

                for shortcut, description in bindings:
                    row = QHBoxLayout()
                    key_label = QLabel(shortcut)
                    key_label.setFixedWidth(120)
                    key_label.setStyleSheet("""
                        QLabel {
                            background: #3a3a3a;
                            padding: 4px 8px;
                            border-radius: 3px;
                            font-family: monospace;
                        }
                    """)
                    desc_label = QLabel(description)
                    desc_label.setStyleSheet("color: #ccc;")
                    row.addWidget(key_label)
                    row.addWidget(desc_label, 1)
                    group_layout.addLayout(row)

                column_layout.addWidget(group)

            column_layout.addStretch()
            columns_layout.addWidget(column_widget)

        layout.addLayout(columns_layout)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class SettingsDialog(QDialog):
    """Settings dialog for application preferences."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Display settings group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self._show_preset_name_checkbox = QCheckBox("Show preset name on change")
        self._show_preset_name_checkbox.setToolTip(
            "Display an overlay with the preset name when switching presets"
        )
        display_layout.addWidget(self._show_preset_name_checkbox)

        layout.addWidget(display_group)

        # Panel behavior group
        panels_group = QGroupBox("Panel Behavior on Startup")
        panels_layout = QVBoxLayout(panels_group)

        # Presets panel behavior
        presets_layout = QHBoxLayout()
        presets_label = QLabel("Presets panel:")
        presets_label.setFixedWidth(120)
        self._presets_combo = QComboBox()
        self._presets_combo.addItem("Remember last position", "last")
        self._presets_combo.addItem("Always expanded", "expanded")
        self._presets_combo.addItem("Always collapsed", "collapsed")
        presets_layout.addWidget(presets_label)
        presets_layout.addWidget(self._presets_combo, 1)
        panels_layout.addLayout(presets_layout)

        # Adjustments panel behavior
        adjustments_layout = QHBoxLayout()
        adjustments_label = QLabel("Adjustments panel:")
        adjustments_label.setFixedWidth(120)
        self._adjustments_combo = QComboBox()
        self._adjustments_combo.addItem("Remember last position", "last")
        self._adjustments_combo.addItem("Always expanded", "expanded")
        self._adjustments_combo.addItem("Always collapsed", "collapsed")
        adjustments_layout.addWidget(adjustments_label)
        adjustments_layout.addWidget(self._adjustments_combo, 1)
        panels_layout.addLayout(adjustments_layout)

        # Controls panel behavior
        controls_layout = QHBoxLayout()
        controls_label = QLabel("Controls panel:")
        controls_label.setFixedWidth(120)
        self._controls_combo = QComboBox()
        self._controls_combo.addItem("Remember last position", "last")
        self._controls_combo.addItem("Always expanded", "expanded")
        self._controls_combo.addItem("Always collapsed", "collapsed")
        controls_layout.addWidget(controls_label)
        controls_layout.addWidget(self._controls_combo, 1)
        panels_layout.addLayout(controls_layout)

        # Debug panels behavior
        debug_layout = QHBoxLayout()
        debug_label = QLabel("Debug panels:")
        debug_label.setFixedWidth(120)
        self._debug_combo = QComboBox()
        self._debug_combo.addItem("Remember last position", "last")
        self._debug_combo.addItem("Always expanded", "expanded")
        self._debug_combo.addItem("Always collapsed", "collapsed")
        debug_layout.addWidget(debug_label)
        debug_layout.addWidget(self._debug_combo, 1)
        panels_layout.addLayout(debug_layout)

        # Default view on startup
        view_layout = QHBoxLayout()
        view_label = QLabel("Default view:")
        view_label.setFixedWidth(120)
        self._startup_view_combo = QComboBox()
        self._startup_view_combo.addItem("Detection tab", "detection")
        self._startup_view_combo.addItem("Development tab", "development")
        self._startup_view_combo.addItem("Last opened tab", "last")
        view_layout.addWidget(view_label)
        view_layout.addWidget(self._startup_view_combo, 1)
        panels_layout.addLayout(view_layout)

        layout.addWidget(panels_group)

        # Crop settings group
        crop_group = QGroupBox("Crop Settings")
        crop_layout = QVBoxLayout(crop_group)

        # Default aspect ratio
        aspect_layout = QHBoxLayout()
        aspect_label = QLabel("Default aspect ratio:")
        aspect_label.setFixedWidth(120)
        self._default_aspect_combo = QComboBox()
        for key in TransformState.ASPECT_RATIOS:
            display_name = TransformState.ASPECT_RATIOS[key][2]
            self._default_aspect_combo.addItem(display_name, key)
        aspect_layout.addWidget(aspect_label)
        aspect_layout.addWidget(self._default_aspect_combo, 1)
        crop_layout.addLayout(aspect_layout)

        # Crop invert mode behavior
        invert_layout = QHBoxLayout()
        invert_label = QLabel("Invert mode:")
        invert_label.setFixedWidth(120)
        self._crop_invert_combo = QComboBox()
        self._crop_invert_combo.addItem("Remember last state", "last")
        self._crop_invert_combo.addItem("Always on", "on")
        self._crop_invert_combo.addItem("Always off", "off")
        invert_layout.addWidget(invert_label)
        invert_layout.addWidget(self._crop_invert_combo, 1)
        crop_layout.addLayout(invert_layout)

        layout.addWidget(crop_group)

        # Spacer
        layout.addStretch()

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._save_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_settings(self):
        """Load current settings into the dialog."""
        store = storage.get_storage()

        # Show preset name
        self._show_preset_name_checkbox.setChecked(
            store.get_show_preset_name_on_change()
        )

        # Presets panel behavior
        presets_behavior = store.get_preset_panel_startup_behavior()
        index = self._presets_combo.findData(presets_behavior)
        if index >= 0:
            self._presets_combo.setCurrentIndex(index)

        # Adjustments panel behavior
        adjustments_behavior = store.get_adjustments_panel_startup_behavior()
        index = self._adjustments_combo.findData(adjustments_behavior)
        if index >= 0:
            self._adjustments_combo.setCurrentIndex(index)

        # Controls panel behavior
        controls_behavior = store.get_controls_panel_startup_behavior()
        index = self._controls_combo.findData(controls_behavior)
        if index >= 0:
            self._controls_combo.setCurrentIndex(index)

        # Debug panels behavior
        debug_behavior = store.get_debug_panel_startup_behavior()
        index = self._debug_combo.findData(debug_behavior)
        if index >= 0:
            self._debug_combo.setCurrentIndex(index)

        # Startup view
        startup_tab = store.get_startup_tab()
        index = self._startup_view_combo.findData(startup_tab)
        if index >= 0:
            self._startup_view_combo.setCurrentIndex(index)

        # Default aspect ratio
        default_aspect = store.get_default_aspect_ratio()
        index = self._default_aspect_combo.findData(default_aspect)
        if index >= 0:
            self._default_aspect_combo.setCurrentIndex(index)

        # Crop invert mode behavior
        crop_invert_behavior = store.get_crop_invert_startup_behavior()
        index = self._crop_invert_combo.findData(crop_invert_behavior)
        if index >= 0:
            self._crop_invert_combo.setCurrentIndex(index)

    def _save_and_accept(self):
        """Save settings and close the dialog."""
        store = storage.get_storage()

        # Save show preset name
        store.set_show_preset_name_on_change(
            self._show_preset_name_checkbox.isChecked()
        )

        # Save presets panel behavior
        store.set_preset_panel_startup_behavior(
            self._presets_combo.currentData()
        )

        # Save adjustments panel behavior
        store.set_adjustments_panel_startup_behavior(
            self._adjustments_combo.currentData()
        )

        # Save controls panel behavior
        store.set_controls_panel_startup_behavior(
            self._controls_combo.currentData()
        )

        # Save debug panels behavior
        store.set_debug_panel_startup_behavior(
            self._debug_combo.currentData()
        )

        # Save startup view
        store.set_startup_tab(
            self._startup_view_combo.currentData()
        )

        # Save default aspect ratio
        store.set_default_aspect_ratio(
            self._default_aspect_combo.currentData()
        )

        # Save crop invert mode behavior
        store.set_crop_invert_startup_behavior(
            self._crop_invert_combo.currentData()
        )

        self.accept()


class SliderWithButtons(QWidget):
    """A slider with +/- buttons for fine adjustment and reset.

    Reset button states when a preset is active:
    - Hidden: Value matches both absolute default and preset default
    - Orange: Value equals preset default but differs from absolute default
    - Red: Value differs from preset default (user tweaked)
    - Blue: Value equals absolute default but differs from preset default
    """

    valueChanged = Signal(float)

    def __init__(self, label: str, min_val: float, max_val: float, default: float,
                 step: float = 0.1, decimals: int = 1, coarse_step: float = None,
                 info_text: str = None):
        super().__init__()
        self.step = step
        self.coarse_step = coarse_step
        self.decimals = decimals
        self.min_val = min_val
        self.max_val = max_val
        self.default = default  # Absolute default (usually 0 or neutral)
        self._preset_default = None  # Preset's default value (None = no preset)
        self._info_text = info_text
        self._label_text = label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)

        # Label, value, and reset button
        header = QHBoxLayout()
        header.addWidget(QLabel(label))

        # Info button (optional)
        if info_text:
            self.info_btn = QPushButton("ⓘ")
            self.info_btn.setFixedSize(20, 20)
            self.info_btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    color: #888;
                    font-size: 14px;
                }
                QPushButton:hover {
                    color: #e67e22;
                }
            """)
            self.info_btn.setCursor(Qt.PointingHandCursor)
            self.info_btn.clicked.connect(self._show_info)
            header.addWidget(self.info_btn)

        header.addStretch()
        self.value_label = QLabel(f"{default:.{decimals}f}")
        self.value_label.setStyleSheet("font-weight: bold;")
        header.addWidget(self.value_label)

        self.reset_btn = QPushButton("↺")
        self.reset_btn.setFixedSize(24, 24)
        self.reset_btn.clicked.connect(self._reset)
        self.reset_btn.setToolTip(f"Reset to {default:.{decimals}f}")
        header.addWidget(self.reset_btn)

        layout.addLayout(header)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val / step), int(max_val / step))
        self.slider.setValue(int(default / step))
        self.slider.valueChanged.connect(self._on_slider_change)
        layout.addWidget(self.slider)

        # +/- buttons
        btn_layout = QHBoxLayout()

        # Coarse decrement (optional)
        if coarse_step is not None:
            self.coarse_minus_btn = QPushButton(f"-{coarse_step:g}")
            self.coarse_minus_btn.setFixedWidth(40)
            self.coarse_minus_btn.clicked.connect(self._coarse_decrement)
            btn_layout.addWidget(self.coarse_minus_btn)

        self.minus_btn = QPushButton(f"-{step}")
        self.minus_btn.setFixedWidth(50)
        self.minus_btn.clicked.connect(self._decrement)
        btn_layout.addWidget(self.minus_btn)

        btn_layout.addStretch()

        self.plus_btn = QPushButton(f"+{step}")
        self.plus_btn.setFixedWidth(50)
        self.plus_btn.clicked.connect(self._increment)
        btn_layout.addWidget(self.plus_btn)

        # Coarse increment (optional)
        if coarse_step is not None:
            self.coarse_plus_btn = QPushButton(f"+{coarse_step:g}")
            self.coarse_plus_btn.setFixedWidth(40)
            self.coarse_plus_btn.clicked.connect(self._coarse_increment)
            btn_layout.addWidget(self.coarse_plus_btn)

        layout.addLayout(btn_layout)

        # Initial style (at default, so not highlighted)
        self._update_reset_style()

    def value(self) -> float:
        return self.slider.value() * self.step

    def setValue(self, val: float):
        self.slider.blockSignals(True)
        self.slider.setValue(round(val / self.step))
        self.value_label.setText(f"{val:.{self.decimals}f}")
        self.slider.blockSignals(False)
        self._update_reset_style()

    def set_preset_default(self, val):
        """Set the preset's default value. None means no preset active."""
        self._preset_default = val
        self._update_reset_style()

    def set_gradient(self, left_color: str, right_color: str):
        """Set a gradient background on the slider track.

        Args:
            left_color: CSS color for minimum end (e.g., '#00ffff' for cyan)
            right_color: CSS color for maximum end (e.g., '#ff0000' for red)
        """
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid #444;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {left_color}, stop:1 {right_color});
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: #ccc;
                border: 1px solid #888;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #fff;
            }}
        """)

    def _on_slider_change(self, val):
        real_val = val * self.step
        self.value_label.setText(f"{real_val:.{self.decimals}f}")
        self._update_reset_style()
        self.valueChanged.emit(real_val)

    def _increment(self):
        new_val = round(self.value() + self.step, self.decimals)
        new_val = min(new_val, self.max_val)
        self.setValue(new_val)
        self.valueChanged.emit(new_val)

    def _decrement(self):
        new_val = round(self.value() - self.step, self.decimals)
        new_val = max(new_val, self.min_val)
        self.setValue(new_val)
        self.valueChanged.emit(new_val)

    def _coarse_increment(self):
        if self.coarse_step is None:
            return
        new_val = round(self.value() + self.coarse_step, self.decimals)
        new_val = min(new_val, self.max_val)
        self.setValue(new_val)
        self.valueChanged.emit(new_val)

    def _coarse_decrement(self):
        if self.coarse_step is None:
            return
        new_val = round(self.value() - self.coarse_step, self.decimals)
        new_val = max(new_val, self.min_val)
        self.setValue(new_val)
        self.valueChanged.emit(new_val)

    def _values_equal(self, a: float, b: float) -> bool:
        """Check if two values are equal within tolerance."""
        return abs(a - b) < self.step / 2

    def _reset(self):
        """Reset to appropriate value based on current state."""
        current = self.value()
        at_absolute = self._values_equal(current, self.default)
        preset_active = self._preset_default is not None
        at_preset = preset_active and self._values_equal(current, self._preset_default)

        if not preset_active:
            # No preset: just reset to absolute default
            target = self.default
        elif at_absolute and not at_preset:
            # Blue state: at absolute, go to preset
            target = self._preset_default
        elif at_preset and not at_absolute:
            # Orange state: at preset, go to absolute
            target = self.default
        else:
            # Red state: tweaked, go to preset default
            target = self._preset_default

        self.setValue(target)
        self.valueChanged.emit(target)

    def _update_reset_style(self):
        """Update reset button appearance based on current state."""
        current = self.value()
        at_absolute = self._values_equal(current, self.default)
        preset_active = self._preset_default is not None
        preset_equals_absolute = preset_active and self._values_equal(self._preset_default, self.default)
        at_preset = preset_active and self._values_equal(current, self._preset_default)

        if not preset_active:
            # No preset mode: simple orange when modified
            if at_absolute:
                self.reset_btn.setStyleSheet("")
                self.reset_btn.setToolTip(f"At default: {self.default:.{self.decimals}f}")
            else:
                self.reset_btn.setStyleSheet(
                    "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
                self.reset_btn.setToolTip(f"Reset → {self.default:.{self.decimals}f}")
        elif at_absolute and at_preset:
            # At both defaults (they're equal) - no reset needed
            self.reset_btn.setStyleSheet("")
            self.reset_btn.setToolTip(f"At default: {self.default:.{self.decimals}f}")
        elif at_preset and not at_absolute:
            # Orange: at preset default, can go to absolute
            self.reset_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
            self.reset_btn.setToolTip(f"Reset to absolute → {self.default:.{self.decimals}f}")
        elif at_absolute and not at_preset:
            # Blue: at absolute default, can go to preset
            self.reset_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
            self.reset_btn.setToolTip(f"Reset to preset → {self._preset_default:.{self.decimals}f}")
        else:
            # Red: tweaked away from preset, can go to preset
            self.reset_btn.setStyleSheet(
                "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; }")
            self.reset_btn.setToolTip(f"Reset to preset → {self._preset_default:.{self.decimals}f}")

    def _show_info(self):
        """Show info popup for this adjustment."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self._label_text} - Info")
        dialog.setMinimumWidth(350)
        layout = QVBoxLayout(dialog)

        # Title
        title = QLabel(f"<h3>{self._label_text}</h3>")
        layout.addWidget(title)

        # Info text
        info = QLabel(self._info_text)
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background: #2a2a2a; border-radius: 4px;")
        layout.addWidget(info)

        # Close button
        close_btn = QPushButton("Got it")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.open()  # Non-blocking modal

    def blockSignals(self, block: bool):
        """Block/unblock signals from this widget."""
        super().blockSignals(block)
        self.slider.blockSignals(block)


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


class VerticalToggleButton(QWidget):
    """A clickable vertical button with rotated text."""

    clicked = Signal()

    def __init__(self, text: str = "PRESETS", side: str = "right"):
        """
        Args:
            text: Label text to display vertically
            side: Which side of the panel ("left" or "right") - affects arrow direction
        """
        super().__init__()
        self._text = text
        self._side = side  # "left" = button on left of panel, "right" = button on right
        self._hovered = False
        self._collapsed = False  # When panel is collapsed, show arrow hint
        self.setFixedWidth(28)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

    def set_collapsed(self, collapsed: bool):
        """Update visual state based on panel collapsed state."""
        self._collapsed = collapsed
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        if self._hovered:
            painter.fillRect(self.rect(), QColor("#3a3a3a"))
        else:
            painter.fillRect(self.rect(), QColor("#2a2a2a"))

        # Border (left border for right-side button, right border for left-side button)
        painter.setPen(QPen(QColor("#3a3a3a"), 1))
        if self._side == "right":
            painter.drawLine(0, 0, 0, self.height())
        else:
            painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())

        # Draw rotated text
        painter.save()
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(-90)  # Rotate so text reads bottom-to-top

        # Text styling
        font = painter.font()
        font.setPixelSize(11)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 120)
        painter.setFont(font)

        if self._hovered:
            painter.setPen(QColor("#ffffff"))
        else:
            painter.setPen(QColor("#888888"))

        # Draw the text centered, with arrow indicator
        # Arrow direction depends on side and collapsed state:
        # - Right side (presets): collapsed=▲ (expand left), expanded=▼ (collapse right)
        # - Left side (adjustments): collapsed=▼ (expand right), expanded=▲ (collapse left)
        if self._side == "right":
            arrow = '▲ ' if self._collapsed else '▼ '
        else:
            arrow = '▼ ' if self._collapsed else '▲ '
        display_text = f"{arrow}{self._text}"
        text_rect = painter.fontMetrics().boundingRect(display_text)
        painter.drawText(-text_rect.width() // 2, text_rect.height() // 4, display_text)

        painter.restore()

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class SplitVerticalToggleButton(QWidget):
    """A vertical toggle button with two clickable zones for different expansion levels.

    Top zone: Toggle between collapsed and normal expanded
    Bottom zone: Toggle between collapsed/normal and full expanded (grid view)
    """

    topClicked = Signal()  # Normal expand/collapse
    bottomClicked = Signal()  # Full expand/collapse

    # Three states: 'collapsed', 'expanded', 'full'
    STATE_COLLAPSED = 'collapsed'
    STATE_EXPANDED = 'expanded'
    STATE_FULL = 'full'

    def __init__(self, text: str = "PRESETS"):
        super().__init__()
        self._text = text
        self._hovered_zone = None  # 'top', 'bottom', or None
        self._state = self.STATE_COLLAPSED
        self.setFixedWidth(28)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

    def set_state(self, state: str):
        """Update visual state: 'collapsed', 'expanded', or 'full'."""
        self._state = state
        self.update()

    def _get_zone(self, pos) -> str:
        """Determine which zone a position is in."""
        if pos.y() < self.height() / 2:
            return 'top'
        return 'bottom'

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        mid_y = self.height() // 2

        # Draw top zone background
        top_rect = QRect(0, 0, self.width(), mid_y)
        if self._hovered_zone == 'top':
            painter.fillRect(top_rect, QColor("#3a3a3a"))
        else:
            painter.fillRect(top_rect, QColor("#2a2a2a"))

        # Draw bottom zone background
        bottom_rect = QRect(0, mid_y, self.width(), self.height() - mid_y)
        if self._hovered_zone == 'bottom':
            painter.fillRect(bottom_rect, QColor("#3a3a3a"))
        else:
            painter.fillRect(bottom_rect, QColor("#2a2a2a"))

        # Draw divider line between zones
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.drawLine(4, mid_y, self.width() - 4, mid_y)

        # Border on left side (panel is on right)
        painter.setPen(QPen(QColor("#3a3a3a"), 1))
        painter.drawLine(0, 0, 0, self.height())

        # Font setup
        font = painter.font()
        font.setPixelSize(11)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 120)
        painter.setFont(font)

        # Draw top zone text (rotated) - "PRESETS" with arrow
        painter.save()
        painter.translate(self.width() / 2, mid_y / 2)
        painter.rotate(-90)

        if self._hovered_zone == 'top':
            painter.setPen(QColor("#ffffff"))
        else:
            painter.setPen(QColor("#888888"))

        # Arrow indicates action: collapsed shows expand arrow, expanded shows collapse
        if self._state == self.STATE_COLLAPSED:
            arrow = '▲ '  # Will expand
        else:
            arrow = '▼ '  # Will collapse (or switch from full to collapsed)

        display_text = f"{arrow}{self._text}"
        text_rect = painter.fontMetrics().boundingRect(display_text)
        painter.drawText(-text_rect.width() // 2, text_rect.height() // 4, display_text)
        painter.restore()

        # Draw bottom zone text (rotated) - "GRID" indicator
        painter.save()
        painter.translate(self.width() / 2, mid_y + (self.height() - mid_y) / 2)
        painter.rotate(-90)

        if self._hovered_zone == 'bottom':
            painter.setPen(QColor("#ffffff"))
        elif self._state == self.STATE_FULL:
            painter.setPen(QColor("#e67e22"))  # Orange when full
        else:
            painter.setPen(QColor("#888888"))

        # Show grid icon/text - use same arrow style as top zone (rotated -90°)
        if self._state == self.STATE_FULL:
            grid_text = "▼ GRID"  # Will collapse (arrow points right after rotation)
        else:
            grid_text = "▲ GRID"  # Will expand to grid (arrow points left after rotation)

        text_rect = painter.fontMetrics().boundingRect(grid_text)
        painter.drawText(-text_rect.width() // 2, text_rect.height() // 4, grid_text)
        painter.restore()

    def enterEvent(self, event):
        self._hovered_zone = self._get_zone(event.position().toPoint())
        self.update()

    def leaveEvent(self, event):
        self._hovered_zone = None
        self.update()

    def mouseMoveEvent(self, event):
        new_zone = self._get_zone(event.position().toPoint())
        if new_zone != self._hovered_zone:
            self._hovered_zone = new_zone
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            zone = self._get_zone(event.position().toPoint())
            if zone == 'top':
                self.topClicked.emit()
            else:
                self.bottomClicked.emit()


class HorizontalToggleButton(QWidget):
    """A clickable horizontal button for bottom panels."""

    clicked = Signal()

    def __init__(self, text: str = "TRANSFORM"):
        super().__init__()
        self._text = text
        self._hovered = False
        self._collapsed = False
        self.setFixedHeight(24)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

    def set_collapsed(self, collapsed: bool):
        """Update visual state based on panel collapsed state."""
        self._collapsed = collapsed
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        if self._hovered:
            painter.fillRect(self.rect(), QColor("#3a3a3a"))
        else:
            painter.fillRect(self.rect(), QColor("#2a2a2a"))

        # Top border
        painter.setPen(QPen(QColor("#3a3a3a"), 1))
        painter.drawLine(0, 0, self.width(), 0)

        # Text styling
        font = painter.font()
        font.setPixelSize(11)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 120)
        painter.setFont(font)

        if self._hovered:
            painter.setPen(QColor("#ffffff"))
        else:
            painter.setPen(QColor("#888888"))

        # Arrow: collapsed=▲ (expand up), expanded=▼ (collapse down)
        arrow = '▲' if self._collapsed else '▼'
        display_text = f"{arrow} {self._text}"
        text_rect = painter.fontMetrics().boundingRect(display_text)
        x = (self.width() - text_rect.width()) // 2
        y = (self.height() + text_rect.height()) // 2 - 2
        painter.drawText(x, y, display_text)

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class TransformState(QObject):
    """Centralized state for transform controls, shared between views.

    This allows both Detection and Development tabs to share the same
    transform state (rotation, grid, crop) with automatic synchronization.
    """

    # Signals for state changes
    rotationChanged = Signal(int)  # 0, 90, 180, 270
    fineRotationChanged = Signal(float)  # -10.0 to +10.0
    gridEnabledChanged = Signal(bool)
    gridDivisionsChanged = Signal(int)  # 2-20
    cropModeChanged = Signal(bool)
    cropInvertChanged = Signal(bool)
    cropAspectRatioChanged = Signal(str)  # Aspect ratio key
    cropResetRequested = Signal()
    autoRotateRequested = Signal()
    resetRotationRequested = Signal()
    resetFineRotationRequested = Signal()

    # Common film aspect ratios: key -> (width_ratio, height_ratio, display_name)
    ASPECT_RATIOS = {
        'free': (None, None, 'Free'),
        'half_frame': (4, 3, 'Half Frame (4:3)'),
        '35mm': (3, 2, '35mm (3:2)'),
        '6x45': (4, 3, '6×4.5 (4:3)'),
        '6x6': (1, 1, '6×6 (1:1)'),
        '6x7': (7, 6, '6×7 (7:6)'),
        '6x9': (3, 2, '6×9 (3:2)'),
        '4x5': (5, 4, '4×5 (5:4)'),
    }

    def __init__(self):
        super().__init__()
        self._rotation = 0  # 0, 90, 180, 270
        self._fine_rotation = 0.0  # -10.0 to +10.0
        self._grid_enabled = False
        self._grid_divisions = 3
        self._crop_mode = False
        # Load crop invert based on startup behavior setting
        store = storage.get_storage()
        invert_behavior = store.get_crop_invert_startup_behavior()
        if invert_behavior == 'on':
            self._crop_invert = True
        elif invert_behavior == 'off':
            self._crop_invert = False
        else:  # 'last'
            self._crop_invert = store.get_crop_invert_state()
        # Load default aspect ratio from settings
        default_ratio = store.get_default_aspect_ratio()
        self._crop_aspect_ratio = default_ratio if default_ratio in self.ASPECT_RATIOS else '35mm'

    @property
    def rotation(self) -> int:
        return self._rotation

    @rotation.setter
    def rotation(self, value: int):
        value = value % 360
        if self._rotation != value:
            self._rotation = value
            self.rotationChanged.emit(value)

    @property
    def fine_rotation(self) -> float:
        return self._fine_rotation

    @fine_rotation.setter
    def fine_rotation(self, value: float):
        value = max(-10.0, min(10.0, value))
        if self._fine_rotation != value:
            self._fine_rotation = value
            self.fineRotationChanged.emit(value)

    @property
    def grid_enabled(self) -> bool:
        return self._grid_enabled

    @grid_enabled.setter
    def grid_enabled(self, value: bool):
        if self._grid_enabled != value:
            self._grid_enabled = value
            self.gridEnabledChanged.emit(value)

    @property
    def grid_divisions(self) -> int:
        return self._grid_divisions

    @grid_divisions.setter
    def grid_divisions(self, value: int):
        value = max(2, min(20, value))
        if self._grid_divisions != value:
            self._grid_divisions = value
            self.gridDivisionsChanged.emit(value)

    @property
    def crop_mode(self) -> bool:
        return self._crop_mode

    @crop_mode.setter
    def crop_mode(self, value: bool):
        if self._crop_mode != value:
            self._crop_mode = value
            self.cropModeChanged.emit(value)

    @property
    def crop_invert(self) -> bool:
        return self._crop_invert

    @crop_invert.setter
    def crop_invert(self, value: bool):
        if self._crop_invert != value:
            self._crop_invert = value
            # Save state for 'remember last' behavior
            storage.get_storage().set_crop_invert_state(value)
            self.cropInvertChanged.emit(value)

    @property
    def crop_aspect_ratio(self) -> str:
        return self._crop_aspect_ratio

    @crop_aspect_ratio.setter
    def crop_aspect_ratio(self, value: str):
        if value in self.ASPECT_RATIOS and self._crop_aspect_ratio != value:
            self._crop_aspect_ratio = value
            self.cropAspectRatioChanged.emit(value)

    def get_aspect_ratio_value(self):
        """Get the current aspect ratio as a float (width/height), or None if free."""
        ratio_data = self.ASPECT_RATIOS.get(self._crop_aspect_ratio)
        if ratio_data and ratio_data[0] is not None:
            return ratio_data[0] / ratio_data[1]
        return None

    def rotate_cw(self):
        """Rotate 90 degrees clockwise."""
        self.rotation = (self._rotation + 90) % 360

    def rotate_ccw(self):
        """Rotate 90 degrees counter-clockwise."""
        self.rotation = (self._rotation - 90) % 360

    def rotate_180(self):
        """Rotate 180 degrees."""
        self.rotation = (self._rotation + 180) % 360

    def reset_rotation(self):
        """Reset rotation to 0."""
        self._rotation = 0
        self.resetRotationRequested.emit()

    def reset_fine_rotation(self):
        """Reset fine rotation to 0."""
        self._fine_rotation = 0.0
        self.resetFineRotationRequested.emit()

    def request_auto_rotate(self):
        """Request auto-rotation detection."""
        self.autoRotateRequested.emit()

    def request_crop_reset(self):
        """Request crop bounds reset."""
        self.cropResetRequested.emit()


class TransformControlsWidget(QWidget):
    """Transform controls UI - rotation, grid, crop.

    This widget contains the actual controls and connects to a TransformState
    object for shared state management between views.
    """

    def __init__(self, transform_state: TransformState):
        super().__init__()
        self._state = transform_state
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Rotation section
        rotation_group = QGroupBox("Rotation")
        rotation_layout = QVBoxLayout(rotation_group)

        # 90° rotation buttons row
        rotate_row = QHBoxLayout()
        rotate_row.setSpacing(8)

        self.rotate_ccw_btn = QPushButton("↺ 90°")
        self.rotate_ccw_btn.setToolTip("Rotate counter-clockwise")
        rotate_row.addWidget(self.rotate_ccw_btn)

        self.rotate_180_btn = QPushButton("180°")
        self.rotate_180_btn.setToolTip("Rotate 180 degrees")
        rotate_row.addWidget(self.rotate_180_btn)

        self.rotate_cw_btn = QPushButton("↻ 90°")
        self.rotate_cw_btn.setToolTip("Rotate clockwise")
        rotate_row.addWidget(self.rotate_cw_btn)

        self.auto_rotate_btn = QPushButton("Auto")
        self.auto_rotate_btn.setToolTip("Auto-detect rotation from image content")
        rotate_row.addWidget(self.auto_rotate_btn)

        self.reset_rotation_btn = QPushButton("⟲")
        self.reset_rotation_btn.setFixedSize(24, 24)
        self.reset_rotation_btn.setToolTip("Reset rotation to 0°")
        rotate_row.addWidget(self.reset_rotation_btn)

        rotation_layout.addLayout(rotate_row)

        # Fine rotation row
        fine_row = QHBoxLayout()
        fine_row.setSpacing(8)

        fine_label = QLabel("Fine:")
        fine_row.addWidget(fine_label)

        self.fine_rotation_slider = QSlider(Qt.Horizontal)
        self.fine_rotation_slider.setRange(-100, 100)  # -10.0 to +10.0 degrees
        self.fine_rotation_slider.setValue(0)
        self.fine_rotation_slider.setToolTip("Fine rotation for horizon straightening")
        fine_row.addWidget(self.fine_rotation_slider, 1)

        self.fine_rotation_label = QLabel("0.0°")
        self.fine_rotation_label.setFixedWidth(40)
        fine_row.addWidget(self.fine_rotation_label)

        self.reset_fine_rotation_btn = QPushButton("⟲")
        self.reset_fine_rotation_btn.setFixedSize(24, 24)
        self.reset_fine_rotation_btn.setToolTip("Reset fine rotation to 0°")
        fine_row.addWidget(self.reset_fine_rotation_btn)

        rotation_layout.addLayout(fine_row)
        layout.addWidget(rotation_group)

        # Grid section
        grid_group = QGroupBox("Grid")
        grid_layout = QVBoxLayout(grid_group)

        grid_row = QHBoxLayout()
        grid_row.setSpacing(8)

        self.grid_checkbox = QCheckBox("Show Grid")
        grid_row.addWidget(self.grid_checkbox)
        grid_row.addStretch()
        grid_layout.addLayout(grid_row)

        divisions_row = QHBoxLayout()
        divisions_row.setSpacing(8)

        divisions_label = QLabel("Divisions:")
        divisions_row.addWidget(divisions_label)

        self.grid_slider = QSlider(Qt.Horizontal)
        self.grid_slider.setRange(2, 20)
        self.grid_slider.setValue(3)
        divisions_row.addWidget(self.grid_slider, 1)

        self.grid_label = QLabel("3")
        self.grid_label.setFixedWidth(24)
        divisions_row.addWidget(self.grid_label)

        grid_layout.addLayout(divisions_row)
        layout.addWidget(grid_group)

        # Crop section
        crop_group = QGroupBox("Crop")
        crop_layout = QVBoxLayout(crop_group)

        crop_row = QHBoxLayout()
        crop_row.setSpacing(8)

        self.crop_mode_btn = QPushButton("Enter Crop Mode")
        self.crop_mode_btn.setCheckable(True)
        self.crop_mode_btn.setToolTip("Adjust detected frame bounds (C)")
        crop_row.addWidget(self.crop_mode_btn)

        self.invert_crop_btn = QPushButton("Invert")
        self.invert_crop_btn.setCheckable(True)
        self.invert_crop_btn.setToolTip("Toggle inverted preview in crop mode (V)")
        crop_row.addWidget(self.invert_crop_btn)

        self.reset_crop_btn = QPushButton("⟲")
        self.reset_crop_btn.setToolTip("Reset crop to auto-detected bounds")
        self.reset_crop_btn.setFixedSize(24, 24)
        crop_row.addWidget(self.reset_crop_btn)

        crop_layout.addLayout(crop_row)

        # Aspect ratio row
        aspect_row = QHBoxLayout()
        aspect_row.setSpacing(8)

        aspect_label = QLabel("Aspect Ratio:")
        aspect_row.addWidget(aspect_label)

        self.aspect_ratio_combo = QComboBox()
        for key in TransformState.ASPECT_RATIOS:
            display_name = TransformState.ASPECT_RATIOS[key][2]
            self.aspect_ratio_combo.addItem(display_name, key)
        # Set initial selection to match state default
        default_index = self.aspect_ratio_combo.findData(self._state.crop_aspect_ratio)
        if default_index >= 0:
            self.aspect_ratio_combo.setCurrentIndex(default_index)
        self.aspect_ratio_combo.setToolTip("Lock crop to film aspect ratio")
        aspect_row.addWidget(self.aspect_ratio_combo, 1)

        crop_layout.addLayout(aspect_row)
        layout.addWidget(crop_group)

        layout.addStretch()

    def _connect_signals(self):
        """Connect UI controls to state and vice versa."""
        # Button clicks -> state
        self.rotate_ccw_btn.clicked.connect(self._state.rotate_ccw)
        self.rotate_180_btn.clicked.connect(self._state.rotate_180)
        self.rotate_cw_btn.clicked.connect(self._state.rotate_cw)
        self.reset_rotation_btn.clicked.connect(self._state.reset_rotation)
        self.auto_rotate_btn.clicked.connect(self._state.request_auto_rotate)
        self.reset_fine_rotation_btn.clicked.connect(self._state.reset_fine_rotation)
        self.reset_crop_btn.clicked.connect(self._state.request_crop_reset)

        # Slider/checkbox changes -> state
        self.fine_rotation_slider.valueChanged.connect(self._on_fine_rotation_slider_changed)
        self.grid_checkbox.toggled.connect(self._on_grid_checkbox_toggled)
        self.grid_slider.valueChanged.connect(self._on_grid_slider_changed)
        self.crop_mode_btn.toggled.connect(self._on_crop_mode_toggled)
        self.invert_crop_btn.toggled.connect(self._on_invert_crop_toggled)
        self.aspect_ratio_combo.currentIndexChanged.connect(self._on_aspect_ratio_changed)

        # State changes -> UI updates
        self._state.fineRotationChanged.connect(self._update_fine_rotation_ui)
        self._state.gridEnabledChanged.connect(self._update_grid_enabled_ui)
        self._state.gridDivisionsChanged.connect(self._update_grid_divisions_ui)
        self._state.cropModeChanged.connect(self._update_crop_mode_ui)
        self._state.cropInvertChanged.connect(self._update_crop_invert_ui)
        self._state.cropAspectRatioChanged.connect(self._update_aspect_ratio_ui)
        self._state.resetFineRotationRequested.connect(self._on_fine_rotation_reset)

    def _on_fine_rotation_slider_changed(self, value: int):
        self._state.fine_rotation = value / 10.0
        self.fine_rotation_label.setText(f"{value / 10.0:.1f}°")

    def _on_grid_checkbox_toggled(self, checked: bool):
        self._state.grid_enabled = checked

    def _on_grid_slider_changed(self, value: int):
        self._state.grid_divisions = value
        self.grid_label.setText(str(value))

    def _on_crop_mode_toggled(self, checked: bool):
        self._state.crop_mode = checked

    def _on_invert_crop_toggled(self, checked: bool):
        self._state.crop_invert = checked

    def _on_aspect_ratio_changed(self, index: int):
        key = self.aspect_ratio_combo.itemData(index)
        if key:
            self._state.crop_aspect_ratio = key

    def _update_aspect_ratio_ui(self, key: str):
        self.aspect_ratio_combo.blockSignals(True)
        index = self.aspect_ratio_combo.findData(key)
        if index >= 0:
            self.aspect_ratio_combo.setCurrentIndex(index)
        self.aspect_ratio_combo.blockSignals(False)

    def _update_fine_rotation_ui(self, value: float):
        self.fine_rotation_slider.blockSignals(True)
        self.fine_rotation_slider.setValue(int(value * 10))
        self.fine_rotation_slider.blockSignals(False)
        self.fine_rotation_label.setText(f"{value:.1f}°")
        self._update_fine_rotation_reset_style(value)

    def _update_fine_rotation_reset_style(self, value: float = None):
        """Update fine rotation reset button style based on current value."""
        if value is None:
            value = self.fine_rotation_slider.value() / 10.0
        if abs(value) > 0.05:  # Not at default (0)
            self.reset_fine_rotation_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
            self.reset_fine_rotation_btn.setToolTip(f"Reset fine rotation to 0° (currently {value:.1f}°)")
        else:
            self.reset_fine_rotation_btn.setStyleSheet("")
            self.reset_fine_rotation_btn.setToolTip("Fine rotation at default: 0°")

    def _update_grid_enabled_ui(self, enabled: bool):
        self.grid_checkbox.blockSignals(True)
        self.grid_checkbox.setChecked(enabled)
        self.grid_checkbox.blockSignals(False)

    def _update_grid_divisions_ui(self, divisions: int):
        self.grid_slider.blockSignals(True)
        self.grid_slider.setValue(divisions)
        self.grid_slider.blockSignals(False)
        self.grid_label.setText(str(divisions))

    def _update_crop_mode_ui(self, active: bool):
        self.crop_mode_btn.blockSignals(True)
        self.crop_mode_btn.setChecked(active)
        self.crop_mode_btn.setText("Exit Crop Mode" if active else "Enter Crop Mode")
        self.crop_mode_btn.blockSignals(False)

    def _update_crop_invert_ui(self, inverted: bool):
        self.invert_crop_btn.blockSignals(True)
        self.invert_crop_btn.setChecked(inverted)
        self.invert_crop_btn.blockSignals(False)

    def _on_fine_rotation_reset(self):
        self.fine_rotation_slider.blockSignals(True)
        self.fine_rotation_slider.setValue(0)
        self.fine_rotation_slider.blockSignals(False)
        self.fine_rotation_label.setText("0.0°")
        self._update_fine_rotation_reset_style(0.0)

    def set_crop_adjustment(self, adjustment: dict):
        """Update crop reset button style based on adjustment values."""
        has_adjustment = any(v != 0 for v in adjustment.values())
        if has_adjustment:
            self.reset_crop_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
            self.reset_crop_btn.setToolTip(f"Reset crop (L:{adjustment.get('left', 0)} T:{adjustment.get('top', 0)} R:{adjustment.get('right', 0)} B:{adjustment.get('bottom', 0)})")
        else:
            self.reset_crop_btn.setStyleSheet("")
            self.reset_crop_btn.setToolTip("Crop at auto-detected bounds")


class CollapsibleTransformPanel(QWidget):
    """A collapsible bottom panel for transform controls (rotation, grid)."""

    visibilityChanged = Signal(bool)

    EXPANDED_HEIGHT = 115  # Three rows stacked with spacing
    COLLAPSED_HEIGHT = 24  # Height of toggle button
    ANIMATION_DURATION = 200

    def __init__(self):
        super().__init__()
        self._expanded = True
        self._setup_ui()
        self._setup_animation()
        self._apply_state_immediate()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toggle button at top
        self._toggle_btn = HorizontalToggleButton("TRANSFORM")
        self._toggle_btn.setToolTip("Toggle transform panel (T)")
        self._toggle_btn.clicked.connect(self.toggle)
        layout.addWidget(self._toggle_btn)

        # Content area - stacked rows
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(10, 6, 10, 6)
        content_layout.setSpacing(8)

        # Row 1: Rotation controls
        rotation_row = QHBoxLayout()
        rotation_row.setSpacing(8)

        rotation_label = QLabel("Rotation")
        rotation_label.setStyleSheet("font-weight: bold;")
        rotation_label.setFixedWidth(55)
        rotation_row.addWidget(rotation_label)

        self.rotate_ccw_btn = QPushButton("↺ 90°")
        self.rotate_ccw_btn.setToolTip("Rotate counter-clockwise")
        self.rotate_ccw_btn.setFixedWidth(50)
        rotation_row.addWidget(self.rotate_ccw_btn)

        self.rotate_cw_btn = QPushButton("↻ 90°")
        self.rotate_cw_btn.setToolTip("Rotate clockwise")
        self.rotate_cw_btn.setFixedWidth(50)
        rotation_row.addWidget(self.rotate_cw_btn)

        self.reset_rotation_btn = QPushButton("⟲")
        self.reset_rotation_btn.setFixedSize(24, 24)
        self.reset_rotation_btn.setToolTip("Reset rotation to 0°")
        rotation_row.addWidget(self.reset_rotation_btn)

        self.auto_rotate_btn = QPushButton("Auto")
        self.auto_rotate_btn.setToolTip("Auto-detect rotation from image content")
        self.auto_rotate_btn.setFixedWidth(40)
        rotation_row.addWidget(self.auto_rotate_btn)

        rotation_row.addSpacing(10)

        fine_label = QLabel("Fine:")
        rotation_row.addWidget(fine_label)

        self.fine_rotation_slider = QSlider(Qt.Horizontal)
        self.fine_rotation_slider.setRange(-100, 100)  # -10.0 to +10.0 degrees
        self.fine_rotation_slider.setValue(0)
        self.fine_rotation_slider.setFixedWidth(100)
        self.fine_rotation_slider.setToolTip("Fine rotation for horizon straightening")
        rotation_row.addWidget(self.fine_rotation_slider)

        self.fine_rotation_label = QLabel("0.0°")
        self.fine_rotation_label.setFixedWidth(35)
        rotation_row.addWidget(self.fine_rotation_label)

        self.reset_fine_rotation_btn = QPushButton("⟲")
        self.reset_fine_rotation_btn.setFixedSize(24, 24)
        self.reset_fine_rotation_btn.setToolTip("Reset fine rotation to 0°")
        self.reset_fine_rotation_btn.clicked.connect(self._reset_fine_rotation)
        rotation_row.addWidget(self.reset_fine_rotation_btn)

        rotation_row.addStretch()
        content_layout.addLayout(rotation_row)

        # Row 2: Grid controls
        grid_row = QHBoxLayout()
        grid_row.setSpacing(8)

        grid_label = QLabel("Grid")
        grid_label.setStyleSheet("font-weight: bold;")
        grid_label.setFixedWidth(55)
        grid_row.addWidget(grid_label)

        self.grid_checkbox = QCheckBox("Show")
        grid_row.addWidget(self.grid_checkbox)

        grid_row.addSpacing(10)

        divisions_label = QLabel("Divisions:")
        grid_row.addWidget(divisions_label)

        self.grid_slider = QSlider(Qt.Horizontal)
        self.grid_slider.setRange(2, 20)
        self.grid_slider.setValue(3)
        self.grid_slider.setFixedWidth(80)
        grid_row.addWidget(self.grid_slider)

        self.grid_label = QLabel("3")
        self.grid_label.setFixedWidth(20)
        grid_row.addWidget(self.grid_label)

        grid_row.addStretch()
        content_layout.addLayout(grid_row)

        # Row 3: Crop controls
        crop_row = QHBoxLayout()
        crop_row.setSpacing(8)

        crop_label = QLabel("Crop")
        crop_label.setStyleSheet("font-weight: bold;")
        crop_label.setFixedWidth(55)
        crop_row.addWidget(crop_label)

        self.crop_mode_btn = QPushButton("Enter Crop Mode")
        self.crop_mode_btn.setCheckable(True)
        self.crop_mode_btn.setToolTip("Adjust detected frame bounds (C)")
        self.crop_mode_btn.setFixedWidth(120)
        crop_row.addWidget(self.crop_mode_btn)

        self.invert_crop_btn = QPushButton("Invert")
        self.invert_crop_btn.setCheckable(True)
        self.invert_crop_btn.setToolTip("Toggle inverted preview in crop mode (V)")
        self.invert_crop_btn.setFixedWidth(60)
        crop_row.addWidget(self.invert_crop_btn)

        self.reset_crop_btn = QPushButton("⟲")
        self.reset_crop_btn.setToolTip("Reset crop to auto-detected bounds")
        self.reset_crop_btn.setFixedSize(24, 24)
        crop_row.addWidget(self.reset_crop_btn)

        crop_row.addSpacing(10)

        aspect_label = QLabel("Aspect:")
        crop_row.addWidget(aspect_label)

        self.aspect_ratio_combo = QComboBox()
        for key in TransformState.ASPECT_RATIOS:
            display_name = TransformState.ASPECT_RATIOS[key][2]
            self.aspect_ratio_combo.addItem(display_name, key)
        self.aspect_ratio_combo.setToolTip("Lock crop to film aspect ratio")
        self.aspect_ratio_combo.setFixedWidth(120)
        crop_row.addWidget(self.aspect_ratio_combo)

        crop_row.addStretch()
        content_layout.addLayout(crop_row)

        layout.addWidget(self._content)

    def _setup_animation(self):
        self._animation = QPropertyAnimation(self, b"maximumHeight")
        self._animation.setDuration(self.ANIMATION_DURATION)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)

    def toggle(self):
        """Toggle the panel open/closed."""
        if self._animation.state() == QPropertyAnimation.Running:
            return
        if self._expanded:
            self._collapse()
        else:
            self._expand()

    def _expand(self):
        self._expanded = True
        target_height = self.EXPANDED_HEIGHT
        self._animation.setStartValue(self.height())
        self._animation.setEndValue(target_height)
        self._animation.start()
        self._toggle_btn.set_collapsed(False)
        self._toggle_btn.setToolTip("Hide transform panel (T)")
        self.visibilityChanged.emit(True)

    def _collapse(self):
        self._expanded = False
        self._animation.setStartValue(self.height())
        self._animation.setEndValue(self.COLLAPSED_HEIGHT)
        self._animation.start()
        self._toggle_btn.set_collapsed(True)
        self._toggle_btn.setToolTip("Show transform panel (T)")
        self.visibilityChanged.emit(False)

    def _apply_state_immediate(self):
        if self._expanded:
            self.setFixedHeight(self.EXPANDED_HEIGHT)
            self._content.show()
            self._toggle_btn.set_collapsed(False)
        else:
            self.setFixedHeight(self.COLLAPSED_HEIGHT)
            self._content.hide()
            self._toggle_btn.set_collapsed(True)

    def _on_animation_finished(self):
        if self._expanded:
            self.setFixedHeight(self.EXPANDED_HEIGHT)
            self._content.show()
        else:
            self.setFixedHeight(self.COLLAPSED_HEIGHT)
            self._content.hide()

    def is_expanded(self) -> bool:
        return self._expanded

    def set_fine_rotation(self, value: float):
        """Set fine rotation slider value."""
        self.fine_rotation_slider.blockSignals(True)
        self.fine_rotation_slider.setValue(int(value * 10))
        self.fine_rotation_label.setText(f"{value:.1f}°")
        self.fine_rotation_slider.blockSignals(False)
        self._update_fine_rotation_reset_style(value)

    def get_fine_rotation(self) -> float:
        """Get fine rotation value in degrees."""
        return self.fine_rotation_slider.value() / 10.0

    def _reset_fine_rotation(self):
        """Reset fine rotation to 0."""
        self.fine_rotation_slider.setValue(0)
        self.fine_rotation_label.setText("0.0°")
        self._update_fine_rotation_reset_style(0.0)

    def _update_fine_rotation_reset_style(self, value: float = None):
        """Update fine rotation reset button style based on current value."""
        if value is None:
            value = self.fine_rotation_slider.value() / 10.0
        if abs(value) > 0.05:  # Not at default (0)
            self.reset_fine_rotation_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
            self.reset_fine_rotation_btn.setToolTip(f"Reset fine rotation to 0° (currently {value:.1f}°)")
        else:
            self.reset_fine_rotation_btn.setStyleSheet("")
            self.reset_fine_rotation_btn.setToolTip("Fine rotation at default: 0°")

    def set_crop_adjustment(self, adjustment: dict):
        """Update crop reset button style based on adjustment values."""
        has_adjustment = any(v != 0 for v in adjustment.values())
        if has_adjustment:
            self.reset_crop_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }")
            self.reset_crop_btn.setToolTip(f"Reset crop (L:{adjustment.get('left', 0)} T:{adjustment.get('top', 0)} R:{adjustment.get('right', 0)} B:{adjustment.get('bottom', 0)})")
        else:
            self.reset_crop_btn.setStyleSheet("")
            self.reset_crop_btn.setToolTip("Crop at auto-detected bounds")


class CollapsiblePresetPanel(QWidget):
    """A collapsible container for the PresetBar with slide animation.

    Supports three states:
    - Collapsed: Only toggle button visible
    - Expanded: Normal width (280px), vertical list of presets
    - Full: Takes available width, hides preview, grid layout
    """

    presetSelected = Signal(str)  # Pass-through from PresetBar
    visibilityChanged = Signal(bool)  # Emitted when panel is shown/hidden
    fullModeChanged = Signal(bool)  # Emitted when entering/exiting full mode

    # States
    STATE_COLLAPSED = 'collapsed'
    STATE_EXPANDED = 'expanded'
    STATE_FULL = 'full'

    EXPANDED_WIDTH = 280  # Normal expanded width
    COLLAPSED_WIDTH = 28  # Width of toggle button strip
    ANIMATION_DURATION = 250  # milliseconds

    def __init__(self):
        super().__init__()
        self._state = self.STATE_COLLAPSED
        self._full_width = 800  # Will be updated dynamically
        # Determine initial state based on startup behavior setting
        store = storage.get_storage()
        behavior = store.get_preset_panel_startup_behavior()
        if behavior == 'expanded':
            self._state = self.STATE_EXPANDED
        elif behavior == 'collapsed':
            self._state = self.STATE_COLLAPSED
        else:  # 'last' - use saved state
            if store.get_preset_panel_expanded():
                self._state = self.STATE_EXPANDED
        self._setup_ui()
        self._setup_animation()
        # Apply initial state without animation
        self._apply_state_immediate()

    def _setup_ui(self):
        # Main horizontal layout: preset bar + toggle button
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # The preset bar (main content)
        self._preset_bar = PresetBar()
        self._preset_bar.presetSelected.connect(self.presetSelected.emit)
        layout.addWidget(self._preset_bar)

        # Split vertical toggle button on the right edge
        self._toggle_btn = SplitVerticalToggleButton("PRESETS")
        self._toggle_btn.topClicked.connect(self._on_top_clicked)
        self._toggle_btn.bottomClicked.connect(self._on_bottom_clicked)
        layout.addWidget(self._toggle_btn)

        # Set initial size
        self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)

    def _setup_animation(self):
        self._animation = QPropertyAnimation(self, b"maximumWidth")
        self._animation.setDuration(self.ANIMATION_DURATION)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)

    def set_full_width(self, width: int):
        """Set the width to use when in full expanded mode."""
        self._full_width = width
        # If currently in full mode, update size
        if self._state == self.STATE_FULL:
            self.setFixedWidth(width)
            self._preset_bar.setFixedWidth(width - self.COLLAPSED_WIDTH)

    def toggle(self):
        """Toggle between collapsed and normal expanded (for keyboard shortcut)."""
        if self._animation.state() == QPropertyAnimation.Running:
            return

        if self._state == self.STATE_COLLAPSED:
            self._expand_normal()
        else:
            self._collapse()

    def _on_top_clicked(self):
        """Handle top zone click - toggle collapsed/normal expanded."""
        if self._animation.state() == QPropertyAnimation.Running:
            return

        if self._state == self.STATE_COLLAPSED:
            self._expand_normal()
        else:
            self._collapse()

    def _on_bottom_clicked(self):
        """Handle bottom zone click - toggle to/from full expanded."""
        if self._animation.state() == QPropertyAnimation.Running:
            return

        if self._state == self.STATE_FULL:
            self._collapse()
        else:
            self._expand_full()

    def _expand_normal(self):
        """Expand to normal width with list view."""
        self._state = self.STATE_EXPANDED
        storage.get_storage().set_preset_panel_expanded(True)
        target_width = self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH

        # Switch to list layout
        self._preset_bar.set_grid_mode(False)
        self._preset_bar.setFixedWidth(self.EXPANDED_WIDTH)

        self._animation.setStartValue(self.width())
        self._animation.setEndValue(target_width)
        self._animation.start()

        self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_EXPANDED)
        self.visibilityChanged.emit(True)

    def _expand_full(self):
        """Expand to full width with grid view."""
        was_full = self._state == self.STATE_FULL
        self._state = self.STATE_FULL
        storage.get_storage().set_preset_panel_expanded(True)

        # Emit signal first so parent can hide preview and update full_width
        if not was_full:
            self.fullModeChanged.emit(True)

        # Now use the updated full_width
        target_width = self._full_width

        # Switch to grid layout
        self._preset_bar.set_grid_mode(True)
        self._preset_bar.setFixedWidth(target_width - self.COLLAPSED_WIDTH)

        self._animation.setStartValue(self.width())
        self._animation.setEndValue(target_width)
        self._animation.start()

        self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_FULL)
        self.visibilityChanged.emit(True)

    def _collapse(self):
        """Collapse the panel."""
        was_full = self._state == self.STATE_FULL
        self._state = self.STATE_COLLAPSED
        storage.get_storage().set_preset_panel_expanded(False)

        # If coming from full mode, first fix the width so animation works
        if was_full:
            current_width = self.width()
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            self.setFixedWidth(current_width)
            self.fullModeChanged.emit(False)

        # Switch back to list layout
        self._preset_bar.set_grid_mode(False)

        self._animation.setStartValue(self.width())
        self._animation.setEndValue(self.COLLAPSED_WIDTH)
        self._animation.start()

        self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_COLLAPSED)
        self.visibilityChanged.emit(False)

    def _apply_state_immediate(self):
        """Apply current state without animation."""
        if self._state == self.STATE_FULL:
            # In full mode, allow expansion
            self.setMinimumWidth(self.COLLAPSED_WIDTH + 100)
            self.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self._preset_bar.set_grid_mode(True)
            self._preset_bar.show()
            self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_FULL)
            # Width will be set by layout, then resizeEvent updates preset bar
        elif self._state == self.STATE_EXPANDED:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._preset_bar.setFixedWidth(self.EXPANDED_WIDTH)
            self._preset_bar.set_grid_mode(False)
            self._preset_bar.show()
            self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_EXPANDED)
        else:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._preset_bar.hide()
            self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_COLLAPSED)

    def _on_animation_finished(self):
        """Handle animation completion."""
        if self._state == self.STATE_FULL:
            # In full mode, remove fixed width and let layout expand the panel
            self.setMinimumWidth(self.COLLAPSED_WIDTH + 100)
            self.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX - allow expansion
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self._preset_bar.show()
            # Update preset bar to match current width
            self._update_preset_bar_width()
        elif self._state == self.STATE_EXPANDED:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._preset_bar.show()
        else:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._preset_bar.hide()

    def resizeEvent(self, event):
        """Handle resize - update preset bar width in full mode."""
        super().resizeEvent(event)
        if self._state == self.STATE_FULL:
            self._update_preset_bar_width()

    def _update_preset_bar_width(self):
        """Update preset bar width to match panel width."""
        bar_width = self.width() - self.COLLAPSED_WIDTH
        if bar_width > 0:
            self._preset_bar.setFixedWidth(bar_width)

    def is_expanded(self) -> bool:
        """Return whether the panel is currently expanded (normal or full)."""
        return self._state != self.STATE_COLLAPSED

    def is_full(self) -> bool:
        """Return whether the panel is in full grid mode."""
        return self._state == self.STATE_FULL

    def set_expanded(self, expanded: bool):
        """Set the panel state without animation (for compatibility)."""
        if expanded and self._state == self.STATE_COLLAPSED:
            self._state = self.STATE_EXPANDED
            self._apply_state_immediate()
            self.visibilityChanged.emit(True)
        elif not expanded and self._state != self.STATE_COLLAPSED:
            self._state = self.STATE_COLLAPSED
            self._apply_state_immediate()
            self.visibilityChanged.emit(False)

    # Pass-through methods to PresetBar
    def select_preset(self, preset_key: str):
        self._preset_bar.select_preset(preset_key)

    def set_base_image(self, img, image_hash: str = None):
        self._preset_bar.set_base_image(img, image_hash)

    def set_modified(self, preset_key: str, modified: bool):
        self._preset_bar.set_modified(preset_key, modified)

    def clear_all_modified(self):
        self._preset_bar.clear_all_modified()

    def get_preset_key_by_number(self, number: int) -> str:
        """Get preset key for a shortcut number (1-9)."""
        return self._preset_bar.get_preset_key_by_number(number)

    def move_current_preset_up(self):
        """Move current preset up in the list."""
        self._preset_bar.move_current_preset_up()

    def move_current_preset_down(self):
        """Move current preset down in the list."""
        self._preset_bar.move_current_preset_down()

    def select_previous_preset(self) -> str:
        """Select previous preset."""
        return self._preset_bar.select_previous_preset()

    def select_next_preset(self) -> str:
        """Select next preset."""
        return self._preset_bar.select_next_preset()

    @property
    def preset_bar(self) -> PresetBar:
        """Access the underlying PresetBar widget."""
        return self._preset_bar


class CollapsibleAdjustmentsPanel(QWidget):
    """A collapsible container for the adjustments controls with slide animation."""

    visibilityChanged = Signal(bool)  # Emitted when panel is shown/hidden

    EXPANDED_WIDTH = 320
    COLLAPSED_WIDTH = 28  # Width of toggle button strip
    ANIMATION_DURATION = 250  # milliseconds

    def __init__(self, content_widget: QWidget):
        """
        Args:
            content_widget: The widget to show/hide (scroll area with controls)
        """
        super().__init__()
        self._content = content_widget
        # Determine initial state based on startup behavior setting
        store = storage.get_storage()
        behavior = store.get_adjustments_panel_startup_behavior()
        if behavior == 'expanded':
            self._expanded = True
        elif behavior == 'collapsed':
            self._expanded = False
        else:  # 'last' - use saved state
            self._expanded = store.get_adjustments_panel_expanded()
        self._setup_ui()
        self._setup_animation()
        # Apply initial state without animation
        self._apply_state_immediate()

    def _setup_ui(self):
        # Main horizontal layout: toggle button + content
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Vertical toggle button on the left edge (since panel is on right side)
        self._toggle_btn = VerticalToggleButton("ADJUSTMENTS", side="left")
        self._toggle_btn.setToolTip("Toggle adjustments panel (A)")
        self._toggle_btn.clicked.connect(self.toggle)
        layout.addWidget(self._toggle_btn)

        # The content (scroll area with controls)
        layout.addWidget(self._content)

        # Set initial size
        self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)

    def _setup_animation(self):
        self._animation = QPropertyAnimation(self, b"maximumWidth")
        self._animation.setDuration(self.ANIMATION_DURATION)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)

    def toggle(self):
        """Toggle the panel open/closed."""
        if self._animation.state() == QPropertyAnimation.Running:
            return  # Don't interrupt ongoing animation

        if self._expanded:
            self._collapse()
        else:
            self._expand()

    def _expand(self):
        """Slide the panel open."""
        self._expanded = True
        storage.get_storage().set_adjustments_panel_expanded(True)
        target_width = self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH

        self._animation.setStartValue(self.width())
        self._animation.setEndValue(target_width)
        self._animation.start()

        self._toggle_btn.set_collapsed(False)
        self._toggle_btn.setToolTip("Hide adjustments panel (A)")
        self.visibilityChanged.emit(True)

    def _collapse(self):
        """Slide the panel closed."""
        self._expanded = False
        storage.get_storage().set_adjustments_panel_expanded(False)

        self._animation.setStartValue(self.width())
        self._animation.setEndValue(self.COLLAPSED_WIDTH)
        self._animation.start()

        self._toggle_btn.set_collapsed(True)
        self._toggle_btn.setToolTip("Show adjustments panel (A)")
        self.visibilityChanged.emit(False)

    def _apply_state_immediate(self):
        """Apply current expanded state without animation."""
        if self._expanded:
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._content.show()
            self._toggle_btn.set_collapsed(False)
            self._toggle_btn.setToolTip("Hide adjustments panel (A)")
        else:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._content.hide()
            self._toggle_btn.set_collapsed(True)
            self._toggle_btn.setToolTip("Show adjustments panel (A)")

    def _on_animation_finished(self):
        """Handle animation completion."""
        if self._expanded:
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._content.show()
        else:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._content.hide()

    def is_expanded(self) -> bool:
        """Return whether the panel is currently expanded."""
        return self._expanded

    def set_expanded(self, expanded: bool):
        """Set the panel state without animation."""
        if expanded == self._expanded:
            return

        self._expanded = expanded
        if expanded:
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._content.show()
            self._toggle_btn.set_collapsed(False)
            self._toggle_btn.setToolTip("Hide adjustments panel (A)")
        else:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._content.hide()
            self._toggle_btn.set_collapsed(True)
            self._toggle_btn.setToolTip("Show adjustments panel (A)")
        self.visibilityChanged.emit(expanded)


class CollapsibleDebugPanel(QWidget):
    """A collapsible container for detection debug panels (right sidebar)."""

    visibilityChanged = Signal(bool)

    EXPANDED_WIDTH = 280
    COLLAPSED_WIDTH = 28
    ANIMATION_DURATION = 250

    def __init__(self):
        super().__init__()
        # Determine initial state based on startup behavior setting
        store = storage.get_storage()
        behavior = store.get_debug_panel_startup_behavior()
        if behavior == 'expanded':
            self._expanded = True
        elif behavior == 'collapsed':
            self._expanded = False
        else:  # 'last' - use saved state
            self._expanded = store.get_debug_panel_expanded()
        self._setup_ui()
        self._setup_animation()
        self._apply_state_immediate()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Content area with stacked panels
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(5)

        # Create the debug panels in processing order
        self.panel_negative = ImagePanel("Negative Mask")
        self.panel_base = BaseSelectionWidget()
        self.panel_edges = ImagePanel("Edge Detection")
        self.panel_extracted = ImagePanel("Extracted Frame")

        # Add panels with labels
        content_layout.addWidget(self._wrap_panel("1. Negative Mask", self.panel_negative))
        content_layout.addWidget(self._wrap_panel("2. Base Selection", self.panel_base))
        content_layout.addWidget(self._wrap_panel("3. Edge Detection", self.panel_edges))
        content_layout.addWidget(self._wrap_panel("4. Extracted Frame", self.panel_extracted))

        layout.addWidget(self._content)

        # Toggle button on right edge (panel is on left side of main layout)
        self._toggle_btn = VerticalToggleButton("DEBUG")
        self._toggle_btn.setToolTip("Toggle debug panels (` or §)")
        self._toggle_btn.clicked.connect(self.toggle)
        layout.addWidget(self._toggle_btn)

        self.setFixedWidth(self.COLLAPSED_WIDTH)

    def _wrap_panel(self, title: str, panel: QWidget) -> QWidget:
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(3, 3, 3, 3)
        group_layout.addWidget(panel)
        return group

    def _setup_animation(self):
        self._animation = QPropertyAnimation(self, b"maximumWidth")
        self._animation.setDuration(self.ANIMATION_DURATION)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)

    def toggle(self):
        if self._animation.state() == QPropertyAnimation.Running:
            return
        if self._expanded:
            self._collapse()
        else:
            self._expand()

    def _expand(self):
        self._expanded = True
        storage.get_storage().set_debug_panel_expanded(True)
        target_width = self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH
        self._animation.setStartValue(self.width())
        self._animation.setEndValue(target_width)
        self._animation.start()
        self._toggle_btn.set_collapsed(False)
        self._toggle_btn.setToolTip("Hide debug panels (` or §)")
        self.visibilityChanged.emit(True)

    def _collapse(self):
        self._expanded = False
        storage.get_storage().set_debug_panel_expanded(False)
        self._animation.setStartValue(self.width())
        self._animation.setEndValue(self.COLLAPSED_WIDTH)
        self._animation.start()
        self._toggle_btn.set_collapsed(True)
        self._toggle_btn.setToolTip("Show debug panels (` or §)")
        self.visibilityChanged.emit(False)

    def _apply_state_immediate(self):
        if self._expanded:
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._content.show()
            self._toggle_btn.set_collapsed(False)
            self._toggle_btn.setToolTip("Hide debug panels (` or §)")
        else:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._content.hide()
            self._toggle_btn.set_collapsed(True)
            self._toggle_btn.setToolTip("Show debug panels (` or §)")

    def _on_animation_finished(self):
        if self._expanded:
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._content.show()
        else:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._content.hide()

    def is_expanded(self) -> bool:
        return self._expanded


class CollapsibleControlsPanel(QWidget):
    """A collapsible container for controls sidebar (right side)."""

    visibilityChanged = Signal(bool)

    EXPANDED_WIDTH = 320  # Match adjustments panel width
    COLLAPSED_WIDTH = 28
    ANIMATION_DURATION = 250

    def __init__(self):
        super().__init__()
        # Determine initial state based on startup behavior setting
        store = storage.get_storage()
        behavior = store.get_controls_panel_startup_behavior()
        if behavior == 'expanded':
            self._expanded = True
        elif behavior == 'collapsed':
            self._expanded = False
        else:  # 'last' - use saved state
            self._expanded = store.get_controls_panel_expanded()
        self._content = None  # Will be set via set_content()
        self._setup_ui()
        self._setup_animation()

    def _setup_ui(self):
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Toggle button on left edge (panel is on right side of main layout)
        self._toggle_btn = VerticalToggleButton("CONTROLS", side="left")
        self._toggle_btn.setToolTip("Toggle controls panel (~ or ±)")
        self._toggle_btn.clicked.connect(self.toggle)
        self._layout.addWidget(self._toggle_btn)

        # Content will be added via set_content()
        self._content_container = QWidget()
        self._content_layout = QVBoxLayout(self._content_container)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._content_container)

        self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)

    def set_content(self, widget: QWidget):
        """Set the content widget for this panel."""
        self._content = widget
        self._content_layout.addWidget(widget)
        self._apply_state_immediate()

    def _setup_animation(self):
        self._animation = QPropertyAnimation(self, b"maximumWidth")
        self._animation.setDuration(self.ANIMATION_DURATION)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)

    def toggle(self):
        if self._animation.state() == QPropertyAnimation.Running:
            return
        if self._expanded:
            self._collapse()
        else:
            self._expand()

    def _expand(self):
        self._expanded = True
        storage.get_storage().set_controls_panel_expanded(True)
        target_width = self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH
        self._animation.setStartValue(self.width())
        self._animation.setEndValue(target_width)
        self._animation.start()
        self._toggle_btn.set_collapsed(False)
        self._toggle_btn.setToolTip("Hide controls panel (~ or ±)")
        self.visibilityChanged.emit(True)

    def _collapse(self):
        self._expanded = False
        storage.get_storage().set_controls_panel_expanded(False)
        self._animation.setStartValue(self.width())
        self._animation.setEndValue(self.COLLAPSED_WIDTH)
        self._animation.start()
        self._toggle_btn.set_collapsed(True)
        self._toggle_btn.setToolTip("Show controls panel (~ or ±)")
        self.visibilityChanged.emit(False)

    def _apply_state_immediate(self):
        if self._expanded:
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._content_container.show()
            self._toggle_btn.set_collapsed(False)
            self._toggle_btn.setToolTip("Hide controls panel (~ or ±)")
        else:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._content_container.hide()
            self._toggle_btn.set_collapsed(True)
            self._toggle_btn.setToolTip("Show controls panel (~ or ±)")

    def _on_animation_finished(self):
        if self._expanded:
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._content_container.show()
        else:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._content_container.hide()

    def is_expanded(self) -> bool:
        return self._expanded


class TabbedRightPanel(QWidget):
    """Collapsible right panel with tabbed content (Controls/Transform).

    This panel contains two tabs:
    - Tab 0: Controls (detection or adjustment controls)
    - Tab 1: Transform (rotation, grid, crop tools)
    """

    visibilityChanged = Signal(bool)
    TAB_CONTROLS = 0
    TAB_TRANSFORM = 1

    EXPANDED_WIDTH = 320
    COLLAPSED_WIDTH = 28
    ANIMATION_DURATION = 250

    def __init__(self, panel_name: str, controls_widget: QWidget,
                 transform_widget: TransformControlsWidget,
                 storage_key: str = "controls"):
        """
        Args:
            panel_name: Name for toggle button (e.g., "CONTROLS" or "ADJUSTMENTS")
            controls_widget: Widget for the first tab (controls/adjustments)
            transform_widget: Widget for the second tab (transform tools)
            storage_key: Key prefix for storage ('controls' or 'adjustments')
        """
        super().__init__()
        self._panel_name = panel_name
        self._controls_widget = controls_widget
        self._transform_widget = transform_widget
        self._storage_key = storage_key

        # Determine initial state based on startup behavior setting
        store = storage.get_storage()
        if storage_key == "adjustments":
            behavior = store.get_adjustments_panel_startup_behavior()
            if behavior == 'expanded':
                self._expanded = True
            elif behavior == 'collapsed':
                self._expanded = False
            else:
                self._expanded = store.get_adjustments_panel_expanded()
        else:  # controls
            behavior = store.get_controls_panel_startup_behavior()
            if behavior == 'expanded':
                self._expanded = True
            elif behavior == 'collapsed':
                self._expanded = False
            else:
                self._expanded = store.get_controls_panel_expanded()

        self._setup_ui()
        self._setup_animation()
        self._apply_state_immediate()

    def _setup_ui(self):
        # Main horizontal layout: toggle button + content
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toggle button on left edge (panel is on right side)
        self._toggle_btn = VerticalToggleButton(self._panel_name, side="left")
        shortcut = "A" if self._storage_key == "adjustments" else "~ or ±"
        self._toggle_btn.setToolTip(f"Toggle {self._panel_name.lower()} panel ({shortcut})")
        self._toggle_btn.clicked.connect(self.toggle)
        layout.addWidget(self._toggle_btn)

        # Content area with tabs
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Tab bar at top
        self._tab_bar = QTabBar()
        self._tab_bar.addTab("Controls")
        self._tab_bar.addTab("Transform")
        self._tab_bar.setExpanding(False)
        self._tab_bar.currentChanged.connect(self._on_tab_changed)
        self._tab_bar.setStyleSheet("""
            QTabBar {
                background: transparent;
            }
            QTabBar::tab {
                padding: 6px 16px;
                margin: 0;
                border: none;
                background: #2a2a2a;
                color: #888;
            }
            QTabBar::tab:selected {
                background: #3a3a3a;
                color: #fff;
                border-bottom: 2px solid #e67e22;
            }
            QTabBar::tab:hover:!selected {
                background: #333;
            }
        """)
        content_layout.addWidget(self._tab_bar)

        # Stacked widget for tab content
        self._stack = QStackedWidget()
        self._stack.addWidget(self._controls_widget)
        self._stack.addWidget(self._transform_widget)
        content_layout.addWidget(self._stack, 1)

        layout.addWidget(self._content)

        # Set initial size
        self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)

    def _setup_animation(self):
        self._animation = QPropertyAnimation(self, b"maximumWidth")
        self._animation.setDuration(self.ANIMATION_DURATION)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)

    def _on_tab_changed(self, index: int):
        """Handle tab switch."""
        self._stack.setCurrentIndex(index)

    def toggle(self):
        """Toggle the panel open/closed."""
        if self._animation.state() == QPropertyAnimation.Running:
            return
        if self._expanded:
            self._collapse()
        else:
            self._expand()

    def _expand(self):
        self._expanded = True
        store = storage.get_storage()
        if self._storage_key == "adjustments":
            store.set_adjustments_panel_expanded(True)
        else:
            store.set_controls_panel_expanded(True)

        target_width = self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH
        self._animation.setStartValue(self.width())
        self._animation.setEndValue(target_width)
        self._animation.start()

        shortcut = "A" if self._storage_key == "adjustments" else "~ or ±"
        self._toggle_btn.set_collapsed(False)
        self._toggle_btn.setToolTip(f"Hide {self._panel_name.lower()} panel ({shortcut})")
        self.visibilityChanged.emit(True)

    def _collapse(self):
        self._expanded = False
        store = storage.get_storage()
        if self._storage_key == "adjustments":
            store.set_adjustments_panel_expanded(False)
        else:
            store.set_controls_panel_expanded(False)

        self._animation.setStartValue(self.width())
        self._animation.setEndValue(self.COLLAPSED_WIDTH)
        self._animation.start()

        shortcut = "A" if self._storage_key == "adjustments" else "~ or ±"
        self._toggle_btn.set_collapsed(True)
        self._toggle_btn.setToolTip(f"Show {self._panel_name.lower()} panel ({shortcut})")
        self.visibilityChanged.emit(False)

    def _apply_state_immediate(self):
        """Apply current state without animation."""
        shortcut = "A" if self._storage_key == "adjustments" else "~ or ±"
        if self._expanded:
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._content.show()
            self._toggle_btn.set_collapsed(False)
            self._toggle_btn.setToolTip(f"Hide {self._panel_name.lower()} panel ({shortcut})")
        else:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._content.hide()
            self._toggle_btn.set_collapsed(True)
            self._toggle_btn.setToolTip(f"Show {self._panel_name.lower()} panel ({shortcut})")

    def _on_animation_finished(self):
        if self._expanded:
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._content.show()
        else:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._content.hide()

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool):
        """Set the panel state without animation."""
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._apply_state_immediate()
        self.visibilityChanged.emit(expanded)

    def current_tab(self) -> int:
        """Return the current tab index."""
        return self._tab_bar.currentIndex()

    def set_current_tab(self, index: int):
        """Set the current tab."""
        self._tab_bar.setCurrentIndex(index)

    def show_transform_tab(self):
        """Switch to the transform tab."""
        self._tab_bar.setCurrentIndex(self.TAB_TRANSFORM)

    def show_controls_tab(self):
        """Switch to the controls tab."""
        self._tab_bar.setCurrentIndex(self.TAB_CONTROLS)

    @property
    def transform_widget(self) -> TransformControlsWidget:
        """Access the transform controls widget."""
        return self._transform_widget

    @property
    def controls_widget(self) -> QWidget:
        """Access the controls widget."""
        return self._controls_widget


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
        # This prevents stale positions from previous images persisting
        # if auto_detect_base() fails to find contours
        self._pos = [0.5, 0.5]
        # Note: Don't clear _debug_sample_boxes here - they'll be cleared
        # when auto_detect_base runs. This preserves boxes through rotations.

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
        """Auto-detect base color position from the film border region.

        Uses morphological erosion to find the border (rebate) area of the film,
        then selects the point with lowest color variance (most uniform = film base).

        Args:
            negative_mask: Binary mask of the film negative area
            brightness_threshold: Minimum mean brightness to consider (rejects dark areas)
                                  Given in uint8 scale (0-255), auto-scaled for float32 images
        """
        if self._image is None or negative_mask is None:
            return

        h, w = self._image.shape[:2]

        # Scale brightness threshold for float32 images (0-1 range)
        if self._image.dtype == np.float32:
            brightness_threshold = brightness_threshold / 255.0

        # 1. Find border region via double erosion
        # First erosion: get safely inside the film (away from black background)
        # Second erosion: define the inner content area
        # The difference is a ring that's definitely on the orange film base
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        outer_mask = cv2.erode(negative_mask, kernel, iterations=3)   # ~15px inside film edge
        inner_mask = cv2.erode(negative_mask, kernel, iterations=15)  # ~75px inside (content area)
        border_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

        # 2. Get all candidate points within the border region
        border_points = np.where(border_mask > 0)
        if len(border_points[0]) == 0:
            # No border found - still update to clear old debug boxes
            self.update()
            return

        # 3. Sample ~30 random points from the border region
        num_candidates = min(30, len(border_points[0]))
        indices = np.linspace(0, len(border_points[0]) - 1, num_candidates, dtype=int)

        best_pos = None
        best_variance = float('inf')
        best_idx = -1
        sample_radius = 15  # pixels to sample around each candidate

        # Clear previous debug boxes and collect new ones
        self._debug_sample_boxes = []
        sample_boxes_temp = []

        for i, idx in enumerate(indices):
            py, px = border_points[0][idx], border_points[1][idx]

            # Extract patch around this point
            x1, y1 = max(0, px - sample_radius), max(0, py - sample_radius)
            x2, y2 = min(w, px + sample_radius), min(h, py + sample_radius)
            patch = self._image[y1:y2, x1:x2]

            if patch.size == 0:
                continue

            # Calculate mean brightness - reject dark patches (likely black background)
            mean_brightness = np.mean(patch)
            if mean_brightness < brightness_threshold:
                # Store as rejected (won't be considered for best)
                sample_boxes_temp.append((px / w, py / h, sample_radius / w, sample_radius / h, float('inf')))
                continue

            # Calculate color variance (lower = more uniform = better base candidate)
            variance = np.var(patch)

            # Store for debug visualization (x%, y%, radius%, variance)
            sample_boxes_temp.append((px / w, py / h, sample_radius / w, sample_radius / h, variance))

            if variance < best_variance:
                best_variance = variance
                best_pos = (px / w, py / h)
                best_idx = len(sample_boxes_temp) - 1

        # Mark the best one
        for i, box in enumerate(sample_boxes_temp):
            x_pct, y_pct, r_w, r_h, var = box
            is_best = (i == best_idx)
            self._debug_sample_boxes.append((x_pct, y_pct, r_w, r_h, var, is_best))

        # 4. Update position if we found a valid candidate
        if best_pos is not None:
            self._pos = list(best_pos)
            self._sample_color()
            self.positionChanged.emit(tuple(self._pos))

        # Always repaint to show debug boxes (even if no best candidate found)
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

        # Sample a small region and average (3x3)
        x1, x2 = max(0, x - 1), min(w, x + 2)
        y1, y2 = max(0, y - 1), min(h, y + 2)
        region = self._image[y1:y2, x1:x2]

        # Keep the same dtype as the image (float32 or uint8)
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
            # Scale radius to widget coordinates
            box_w = r_w * iw * 2
            box_h = r_h * ih * 2

            if is_best:
                # Green for the selected/best box
                painter.setPen(QPen(QColor(0, 255, 0), 2))
            else:
                # Red for other candidates
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
            # Convert float32 (0-1) to uint8 (0-255) for display
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
            # Update cursor when over image
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
        self._crop_full_image = None  # Full rotated image for crop visualization
        self._crop_full_pixmap = None
        self._crop_full_pixmap_inverted = None  # Inverted version for toggle
        self._crop_bounds = None  # (x, y, w, h) in image pixels
        self._crop_adjustment = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
        self._flash_edge = None  # Edge currently flashing red
        self._crop_inverted = False  # Show inverted colors in crop preview
        self._border_flash_edge = None  # Edge to flash on border in normal view

        # Crop edge dragging state
        self._dragging_edge = None  # 'left', 'right', 'top', 'bottom', or None
        self._dragging_box = False  # True when dragging entire box
        self._drag_start_pos = None
        self._drag_start_value = 0
        self._drag_last_pos = None  # For box dragging incremental updates
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
        """Set crop mode state.

        Args:
            active: Whether crop mode is active
            full_image: Full rotated image to show in crop mode (negative)
            inverted_image: Properly inverted image with orange mask removed
            bounds: Auto-detected crop bounds (x, y, w, h) in image pixels
            adjustment: Crop adjustment dict {'left', 'top', 'right', 'bottom'}
            flash_edge: Edge to flash red ('left', 'right', 'top', 'bottom', or None)
        """
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

        # Convert float32 (0-1) to uint8 (0-255) for display
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

        # Draw border
        painter.setPen(QPen(QColor(68, 68, 68), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        # Crop mode: draw full image with crop overlay
        if self._crop_mode_active and self._crop_full_pixmap is not None:
            self._draw_crop_mode(painter)
            return

        if self._pixmap is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, self.title)
            return

        # Draw scaled image
        self._image_rect = self._get_image_rect()
        ix, iy, iw, ih = self._image_rect
        scaled = self._pixmap.scaled(iw, ih, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(ix, iy, scaled)

        # Draw grid overlay
        if self._grid_enabled:
            self._draw_grid(painter, ix, iy, iw, ih)

        # Draw border flash for crop edge feedback in normal view
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
        # Use inverted pixmap if toggled
        pixmap = self._crop_full_pixmap_inverted if self._crop_inverted else self._crop_full_pixmap
        # Calculate image position (same as normal)
        pw, ph = pixmap.width(), pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        iw, ih = int(pw * scale), int(ph * scale)
        ix = (ww - iw) // 2
        iy = (wh - ih) // 2

        # Draw the full rotated image (normal or inverted)
        scaled = pixmap.scaled(iw, ih, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(ix, iy, scaled)

        # Calculate adjusted crop bounds in widget coordinates
        if self._crop_bounds is not None:
            cx, cy, cw, ch = self._crop_bounds
            adj = self._crop_adjustment

            # Apply adjustments (positive = expand outward)
            ax = cx - adj['left']
            ay = cy - adj['top']
            aw = cw + adj['left'] + adj['right']
            ah = ch + adj['top'] + adj['bottom']

            # Scale to widget coordinates
            crop_x = ix + int(ax * scale)
            crop_y = iy + int(ay * scale)
            crop_w = int(aw * scale)
            crop_h = int(ah * scale)

            # Draw darkened overlay outside crop area
            overlay = QColor(0, 0, 0, 160)
            # Top region
            painter.fillRect(ix, iy, iw, crop_y - iy, overlay)
            # Bottom region
            painter.fillRect(ix, crop_y + crop_h, iw, iy + ih - crop_y - crop_h, overlay)
            # Left region (between top and bottom)
            painter.fillRect(ix, crop_y, crop_x - ix, crop_h, overlay)
            # Right region (between top and bottom)
            painter.fillRect(crop_x + crop_w, crop_y, ix + iw - crop_x - crop_w, crop_h, overlay)

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

            # Draw edge handles (small squares at midpoints)
            handle_size = 10
            painter.setPen(QPen(QColor(0, 0, 0), 1))

            # Left handle
            handle_color = flash_color if self._flash_edge == 'left' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x - handle_size // 2, crop_y + crop_h // 2 - handle_size // 2,
                           handle_size, handle_size)
            # Right handle
            handle_color = flash_color if self._flash_edge == 'right' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x + crop_w - handle_size // 2, crop_y + crop_h // 2 - handle_size // 2,
                           handle_size, handle_size)
            # Top handle
            handle_color = flash_color if self._flash_edge == 'top' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x + crop_w // 2 - handle_size // 2, crop_y - handle_size // 2,
                           handle_size, handle_size)
            # Bottom handle
            handle_color = flash_color if self._flash_edge == 'bottom' else normal_color
            painter.setBrush(handle_color)
            painter.drawRect(crop_x + crop_w // 2 - handle_size // 2, crop_y + crop_h - handle_size // 2,
                           handle_size, handle_size)

            # Draw corner handles
            # Top-left corner
            painter.setBrush(normal_color)
            painter.drawRect(crop_x - handle_size // 2, crop_y - handle_size // 2,
                           handle_size, handle_size)
            # Top-right corner
            painter.drawRect(crop_x + crop_w - handle_size // 2, crop_y - handle_size // 2,
                           handle_size, handle_size)
            # Bottom-left corner
            painter.drawRect(crop_x - handle_size // 2, crop_y + crop_h - handle_size // 2,
                           handle_size, handle_size)
            # Bottom-right corner
            painter.drawRect(crop_x + crop_w - handle_size // 2, crop_y + crop_h - handle_size // 2,
                           handle_size, handle_size)

    def _draw_grid(self, painter, ix, iy, iw, ih):
        """Draw grid overlay on the image."""
        for i in range(1, self._grid_divisions):
            # Calculate line positions
            x = ix + (iw * i / self._grid_divisions)
            y = iy + (ih * i / self._grid_divisions)

            # Draw dark outline first (3px black)
            painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
            painter.drawLine(int(x), iy, int(x), iy + ih)
            painter.drawLine(ix, int(y), ix + iw, int(y))

            # Draw white center line (1px white)
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

        # Calculate image position and scale
        pw, ph = self._crop_full_pixmap.width(), self._crop_full_pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        iw, ih = int(pw * scale), int(ph * scale)
        ix = (ww - iw) // 2
        iy = (wh - ih) // 2

        # Calculate crop rectangle in widget coordinates
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
        # Left handle
        if abs(x - crop_x) < handle_radius and abs(y - (crop_y + crop_h // 2)) < handle_radius:
            return 'left'
        # Right handle
        if abs(x - (crop_x + crop_w)) < handle_radius and abs(y - (crop_y + crop_h // 2)) < handle_radius:
            return 'right'
        # Top handle
        if abs(x - (crop_x + crop_w // 2)) < handle_radius and abs(y - crop_y) < handle_radius:
            return 'top'
        # Bottom handle
        if abs(x - (crop_x + crop_w // 2)) < handle_radius and abs(y - (crop_y + crop_h)) < handle_radius:
            return 'bottom'

        return None

    def _is_inside_crop_box(self, pos):
        """Check if position is inside the crop box (but not on an edge handle)."""
        if not self._crop_mode_active or self._crop_full_pixmap is None or self._crop_bounds is None:
            return False

        # Calculate image position and scale
        pw, ph = self._crop_full_pixmap.width(), self._crop_full_pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        iw, ih = int(pw * scale), int(ph * scale)
        ix = (ww - iw) // 2
        iy = (wh - ih) // 2

        # Calculate crop rectangle in widget coordinates
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
            # First check for edge/corner handles
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
            # Then check if inside the box for box dragging
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
            # Calculate delta in pixels (in image coordinates)
            pw, ph = self._crop_full_pixmap.width(), self._crop_full_pixmap.height()
            ww, wh = self.width(), self.height()
            scale = min(ww / pw, wh / ph)

            # Handle corner dragging - emit raw image-space deltas from drag start
            # Main window will track start values and apply constrained values absolutely
            if self._dragging_edge in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
                delta_x_widget = event.position().x() - self._drag_start_pos.x()
                delta_y_widget = event.position().y() - self._drag_start_pos.y()
                # Convert to image space - these are CUMULATIVE deltas from drag start
                delta_x_image = round(delta_x_widget / scale)
                delta_y_image = round(delta_y_widget / scale)
                self.cropCornerDragged.emit(self._dragging_edge, delta_x_image, delta_y_image)
                event.accept()
                return

            # Handle single edge dragging
            if self._dragging_edge in ('left', 'right'):
                delta_widget = event.position().x() - self._drag_start_pos.x()
                delta_image = int(delta_widget / scale)
                # For left edge, moving right = contract (negative delta for adjustment)
                # For right edge, moving right = expand (positive delta for adjustment)
                if self._dragging_edge == 'left':
                    new_value = self._drag_start_value - delta_image
                else:
                    new_value = self._drag_start_value + delta_image
            else:  # top or bottom
                delta_widget = event.position().y() - self._drag_start_pos.y()
                delta_image = int(delta_widget / scale)
                # For top edge, moving down = contract (negative delta for adjustment)
                # For bottom edge, moving down = expand (positive delta for adjustment)
                if self._dragging_edge == 'top':
                    new_value = self._drag_start_value - delta_image
                else:
                    new_value = self._drag_start_value + delta_image

            # Emit signal with the delta from the original value
            delta = new_value - self._crop_adjustment[self._dragging_edge]
            if delta != 0:
                self.cropEdgeDragged.emit(self._dragging_edge, delta)
            event.accept()
            return

        if self._dragging_box:
            # Calculate delta in image coordinates since last position
            pw, ph = self._crop_full_pixmap.width(), self._crop_full_pixmap.height()
            ww, wh = self.width(), self.height()
            scale = min(ww / pw, wh / ph)

            delta_x_widget = event.position().x() - self._drag_last_pos.x()
            delta_y_widget = event.position().y() - self._drag_last_pos.y()
            delta_x_image = int(delta_x_widget / scale)
            delta_y_image = int(delta_y_widget / scale)

            if delta_x_image != 0 or delta_y_image != 0:
                self.cropBoxMoved.emit(delta_x_image, delta_y_image)
                # Update last position based on what we actually moved
                self._drag_last_pos = event.position()
            event.accept()
            return

        # Update cursor based on what's under the mouse
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()


class ShortcutsOverlay(QWidget):
    """Overlay showing keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0.7);")
        self.hide()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # Container box
        box = QFrame()
        box.setFixedWidth(220)
        box.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 6px;
            }
        """)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(16, 12, 16, 16)
        box_layout.setSpacing(10)

        # Title inside box
        title = QLabel("Keyboard Shortcuts")
        title.setStyleSheet("color: #888; font-size: 12px; border: none;")
        title.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(title)

        shortcut_groups = [
            # Navigation
            [
                (["←", "↑"], "Previous"),
                (["→", "↓"], "Next"),
                (["f"], "Favorite"),
                (["F"], "Filter ★"),
            ],
            # Presets
            [
                (["`"], "Presets"),
                (["1-9"], "Apply preset"),
                (["⌘↑"], "Move up"),
                (["⌘↓"], "Move down"),
            ],
            # View
            [
                (["\\"], "Adjustments"),
                (["t"], "Transform"),
                (["x"], "Fit to screen"),
            ],
            # Crop Mode
            [
                (["c"], "Crop mode"),
                (["⌥←→↑↓"], "Crop in"),
                (["⇧⌥←→↑↓"], "Expand"),
            ],
            # File
            [
                (["⌘O"], "Open"),
                (["⌘S"], "Export"),
                (["⌘Q"], "Quit"),
            ],
        ]

        for i, shortcuts in enumerate(shortcut_groups):
            if i > 0:
                box_layout.addSpacing(6)

            for keys, desc in shortcuts:
                row = QHBoxLayout()
                row.setSpacing(6)

                # Fixed-width container for keys
                keys_widget = QWidget()
                keys_widget.setFixedWidth(60)
                keys_widget.setStyleSheet("border: none; background-color: #1a1a1a;")
                keys_layout = QHBoxLayout(keys_widget)
                keys_layout.setContentsMargins(0, 0, 0, 0)
                keys_layout.setSpacing(4)

                keys_layout.addStretch()
                for key in keys:
                    key_label = QLabel(key)
                    key_label.setStyleSheet("""
                        background-color: #2a2a2a;
                        color: #ccc;
                        padding: 3px 6px;
                        border: 1px solid #444;
                        border-radius: 3px;
                        font-size: 12px;
                    """)
                    keys_layout.addWidget(key_label)

                row.addWidget(keys_widget)
                row.addSpacing(8)

                desc_label = QLabel(desc)
                desc_label.setStyleSheet("color: #888; font-size: 12px; border: none;")
                row.addWidget(desc_label)
                row.addStretch()
                box_layout.addLayout(row)

        layout.addWidget(box)

    def showEvent(self, event):
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().showEvent(event)


class NegativeDetectorGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SUPER NEGATIVE PROCESSING SYSTEM")
        self.setMinimumSize(1400, 900)

        self.current_image = None
        self.current_image_original = None  # Unrotated original
        self.current_path = None
        self.current_rotation = 0  # 0, 90, 180, 270
        self.negative_mask = None
        self._current_inverted = None  # For adjustments view
        self._pending_base_pos = None
        self._pending_rotation = None
        self._is_new_image = False
        self._needs_auto_rotate = False  # Auto-rotate on first load when no saved rotation
        self._needs_full_update = True

        # Crop mode state
        self._crop_mode_active = False
        self._crop_adjustment = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
        self._rotated_full_image = None  # Full rotated negative for crop visualization
        self._rotated_full_image_inverted = None  # Properly inverted version (with orange mask removed)
        self._auto_detected_bounds = None  # (x, y, w, h) from auto-detection
        self._flash_edge = None  # Edge currently flashing red
        self._flash_timer = QTimer()
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash_edge)
        # Corner drag state (for absolute positioning)
        self._corner_drag_start = None  # {'corner': str, 'h_edge': str, 'v_edge': str, 'h_val': int, 'v_val': int}

        # File list for CLI multi-file support
        self.file_list = []
        self.file_index = 0

        # Per-image settings storage: {path: {bg, threshold, border, inset}}
        self.image_settings = {}
        self._current_hash = None
        self._storage = storage.get_storage()
        self._file_hashes = {}  # Cache of path -> hash

        # Shared transform state between Detection and Development views
        self._transform_state = TransformState()

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._do_process)

        self._setup_ui()
        self._setup_menu()


    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Outer vertical layout: tab bar on top, content below
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Tab bar row: tabs on left, settings cog on right
        tab_row = QWidget()
        tab_row.setStyleSheet("background: #2a2a2a; border: none;")
        tab_row_layout = QHBoxLayout(tab_row)
        tab_row_layout.setContentsMargins(0, 0, 8, 0)
        tab_row_layout.setSpacing(0)

        self.tab_bar = QTabBar()
        self.tab_bar.addTab("Detection")
        self.tab_bar.addTab("Development")
        self.tab_bar.currentChanged.connect(self._on_tab_changed)
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setStyleSheet("""
            QTabBar {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                padding: 8px 24px;
                margin-right: 2px;
                border: none;
                background: #2a2a2a;
            }
            QTabBar::tab:selected {
                background: #3a3a3a;
                border-bottom: 2px solid #e67e22;
            }
            QTabBar::tab:hover:!selected {
                background: #333;
            }
        """)
        tab_row_layout.addWidget(self.tab_bar)
        tab_row_layout.addStretch()

        # Common style for toolbar buttons
        toolbar_btn_style = """
            QPushButton {
                background: transparent;
                border: none;
                font-size: 16px;
                color: #888;
            }
            QPushButton:hover {
                color: #fff;
            }
        """

        # Help/keybindings button
        help_btn = QPushButton("\u2328")  # Keyboard unicode character
        help_btn.setFixedSize(28, 28)
        help_btn.setToolTip("Keyboard Shortcuts (?)")
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.clicked.connect(self._show_keybindings)
        help_btn.setStyleSheet(toolbar_btn_style + """
            QPushButton { font-size: 18px; }
        """)
        tab_row_layout.addWidget(help_btn)

        # Settings cog button
        settings_btn = QPushButton("\u2699")  # Gear/cog unicode character
        settings_btn.setFixedSize(28, 28)
        settings_btn.setToolTip("Settings (,)")
        settings_btn.setCursor(Qt.PointingHandCursor)
        settings_btn.clicked.connect(self._show_settings)
        settings_btn.setStyleSheet(toolbar_btn_style + """
            QPushButton { font-size: 18px; }
        """)
        tab_row_layout.addWidget(settings_btn)

        outer_layout.addWidget(tab_row)

        # Content area
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Left: Thumbnail bar (shared between both views)
        self.thumbnail_bar = ThumbnailBar()
        self.thumbnail_bar.imageSelected.connect(self._on_thumbnail_selected)
        self.thumbnail_bar.favoriteToggled.connect(self._on_favorite_toggled)
        # Load saved favorite image hashes
        self.thumbnail_bar.set_favorite_hashes(self._storage.get_favorite_images())
        content_layout.addWidget(self.thumbnail_bar)

        # Stacked widget for switching between views
        self.view_stack = QStackedWidget()

        # Detection view: debug panels + main panel + controls sidebar
        detection_view = QWidget()
        detection_layout = QHBoxLayout(detection_view)
        detection_layout.setContentsMargins(10, 10, 10, 10)

        # Left: collapsible debug panels (pipeline visualization)
        debug_panel = self._create_debug_panel()
        detection_layout.addWidget(debug_panel)

        # Center: main panel (Inverted Negative)
        self.panel_inverted = ImagePanel("Inverted Negative")
        self.panel_inverted.cropEdgeDragged.connect(self._on_crop_edge_dragged)
        self.panel_inverted.cropCornerDragStarted.connect(self._on_crop_corner_drag_started)
        self.panel_inverted.cropCornerDragged.connect(self._on_crop_corner_dragged)
        self.panel_inverted.cropBoxMoved.connect(self._on_crop_box_moved)
        detection_layout.addWidget(self.panel_inverted, stretch=1)

        # Right: collapsible controls panel (includes transform at bottom)
        sidebar_content = self._create_sidebar_content()
        self._controls_panel = CollapsibleControlsPanel()
        self._controls_panel.set_content(sidebar_content)
        detection_layout.addWidget(self._controls_panel)

        self.view_stack.addWidget(detection_view)

        # Adjustments view: full-screen preview + curves (shares transform state)
        self.adjustments_view = AdjustmentsView(self._transform_state)
        self.adjustments_view._preview.cropEdgeDragged.connect(self._on_crop_edge_dragged)
        self.adjustments_view._preview.cropCornerDragStarted.connect(self._on_crop_corner_drag_started)
        self.adjustments_view._preview.cropCornerDragged.connect(self._on_crop_corner_dragged)
        self.adjustments_view._preview.cropBoxMoved.connect(self._on_crop_box_moved)
        self.view_stack.addWidget(self.adjustments_view)

        # Connect shared transform state signals to handlers
        self._transform_state.rotationChanged.connect(self._on_transform_rotation_changed)
        self._transform_state.resetRotationRequested.connect(self._reset_rotation)
        self._transform_state.autoRotateRequested.connect(self._auto_rotate_from_content)
        self._transform_state.fineRotationChanged.connect(self._on_fine_rotation_changed)
        self._transform_state.gridEnabledChanged.connect(self._toggle_grid)
        self._transform_state.gridDivisionsChanged.connect(self._update_grid_divisions)
        self._transform_state.cropModeChanged.connect(self._on_crop_mode_changed)
        self._transform_state.cropInvertChanged.connect(self._on_crop_invert_changed)
        self._transform_state.cropResetRequested.connect(self._reset_crop_adjustment)

        content_layout.addWidget(self.view_stack, stretch=1)
        outer_layout.addLayout(content_layout)

        # Determine startup tab based on user preference
        startup_pref = self._storage.get_startup_tab()
        if startup_pref == 'detection':
            startup_index = 0
        elif startup_pref == 'last':
            startup_index = self._storage.get_last_opened_tab()
        else:  # 'development' or default
            startup_index = 1
        self.tab_bar.setCurrentIndex(startup_index)
        self.view_stack.setCurrentIndex(startup_index)

        # Shortcuts overlay (shown when holding ?)
        self._shortcuts_overlay = ShortcutsOverlay(central)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        # V to toggle invert in crop mode
        if event.key() == Qt.Key_V and not event.modifiers():
            if self._crop_mode_active:
                self._toggle_crop_invert()
                event.accept()
                return

        # Tab to switch between Detection/Development
        if event.key() == Qt.Key_Tab and not event.modifiers():
            self._toggle_tab()
            event.accept()
            return

        # C to toggle crop mode
        if event.key() == Qt.Key_C and not event.modifiers():
            self._transform_state.crop_mode = not self._transform_state.crop_mode
            event.accept()
            return

        # T to toggle controls/adjustments panel
        if event.key() == Qt.Key_T and not event.modifiers():
            current_view = self.view_stack.currentIndex()
            if current_view == 1:  # Development view
                self.adjustments_view._adjustments_panel.toggle()
            else:  # Detection view
                self._controls_panel.toggle()
            event.accept()
            return

        # Q/A W/S E/D for R/G/B white balance adjustment, Shift to reset
        wb_step = 0.05
        if event.key() == Qt.Key_Q and not event.modifiers():
            new_val = self.adjustments_view.wb_r_slider.value() + wb_step
            self.adjustments_view.wb_r_slider.setValue(new_val)
            self.adjustments_view.wb_r_slider.valueChanged.emit(new_val)
            event.accept()
            return
        if event.key() == Qt.Key_A and event.modifiers() == Qt.ShiftModifier:
            self.adjustments_view.wb_r_slider.setValue(1.0)
            self.adjustments_view.wb_r_slider.valueChanged.emit(1.0)
            event.accept()
            return
        if event.key() == Qt.Key_A and not event.modifiers():
            new_val = self.adjustments_view.wb_r_slider.value() - wb_step
            self.adjustments_view.wb_r_slider.setValue(new_val)
            self.adjustments_view.wb_r_slider.valueChanged.emit(new_val)
            event.accept()
            return
        if event.key() == Qt.Key_W and not event.modifiers():
            new_val = self.adjustments_view.wb_g_slider.value() + wb_step
            self.adjustments_view.wb_g_slider.setValue(new_val)
            self.adjustments_view.wb_g_slider.valueChanged.emit(new_val)
            event.accept()
            return
        if event.key() == Qt.Key_S and event.modifiers() == Qt.ShiftModifier:
            self.adjustments_view.wb_g_slider.setValue(1.0)
            self.adjustments_view.wb_g_slider.valueChanged.emit(1.0)
            event.accept()
            return
        if event.key() == Qt.Key_S and not event.modifiers():
            new_val = self.adjustments_view.wb_g_slider.value() - wb_step
            self.adjustments_view.wb_g_slider.setValue(new_val)
            self.adjustments_view.wb_g_slider.valueChanged.emit(new_val)
            event.accept()
            return
        if event.key() == Qt.Key_E and not event.modifiers():
            new_val = self.adjustments_view.wb_b_slider.value() + wb_step
            self.adjustments_view.wb_b_slider.setValue(new_val)
            self.adjustments_view.wb_b_slider.valueChanged.emit(new_val)
            event.accept()
            return
        if event.key() == Qt.Key_D and event.modifiers() == Qt.ShiftModifier:
            self.adjustments_view.wb_b_slider.setValue(1.0)
            self.adjustments_view.wb_b_slider.valueChanged.emit(1.0)
            event.accept()
            return
        if event.key() == Qt.Key_D and not event.modifiers():
            new_val = self.adjustments_view.wb_b_slider.value() - wb_step
            self.adjustments_view.wb_b_slider.setValue(new_val)
            self.adjustments_view.wb_b_slider.valueChanged.emit(new_val)
            event.accept()
            return

        # Fit image to screen with X key
        if event.key() == Qt.Key_X and not event.modifiers():
            self.adjustments_view._preview.reset_view()
            event.accept()
            return

        # Show shortcuts overlay when ? is pressed
        if event.key() == Qt.Key_Question and not event.isAutoRepeat():
            self._shortcuts_overlay.setGeometry(self.centralWidget().rect())
            self._shortcuts_overlay.show()
            self._shortcuts_overlay.raise_()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Hide shortcuts overlay when ? is released."""
        if event.key() == Qt.Key_Question and not event.isAutoRepeat():
            self._shortcuts_overlay.hide()
        else:
            super().keyReleaseEvent(event)

    def _on_tab_changed(self, index: int):
        """Handle tab switching between Detection and Adjustments views."""
        self.view_stack.setCurrentIndex(index)
        # Save last opened tab for "remember last" setting
        self._storage.set_last_opened_tab(index)
        if index == 1:  # Adjustments view
            self._update_adjustments_preview()

    def _on_fine_rotation_changed(self, value: float):
        """Handle fine rotation change from shared transform state."""
        # Exit crop mode when rotating (bounds become invalid until reprocessed)
        if self._crop_mode_active:
            self._transform_state.crop_mode = False
        self._schedule_content_update()

    def _on_transform_rotation_changed(self, new_rotation: int):
        """Handle rotation change from shared transform state."""
        if self.current_image_original is None:
            return
        # Calculate delta from current to new rotation
        delta = (new_rotation - self.current_rotation) % 360
        if delta == 0:
            return
        if delta > 180:
            delta = delta - 360  # Use negative rotation for shorter path
        self._rotate_image(delta)

    def _on_crop_mode_changed(self, enabled: bool):
        """Handle crop mode change from shared transform state."""
        self._toggle_crop_mode(enabled)

    def _on_crop_invert_changed(self, checked: bool):
        """Handle crop invert toggle from shared transform state."""
        if self._crop_mode_active:
            # Sync invert state to both panels
            self.panel_inverted._invert_btn.blockSignals(True)
            self.panel_inverted._invert_btn.setChecked(checked)
            self.panel_inverted._crop_inverted = checked
            self.panel_inverted._invert_btn.blockSignals(False)
            self.panel_inverted.update()

            self.adjustments_view._preview._invert_btn.blockSignals(True)
            self.adjustments_view._preview._invert_btn.setChecked(checked)
            self.adjustments_view._preview._crop_inverted = checked
            self.adjustments_view._preview._invert_btn.blockSignals(False)
            self.adjustments_view._preview.update()

    def _create_sidebar_content(self) -> QWidget:
        """Create the sidebar content for Detection view controls tab."""
        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)

        # Parameters section
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        # Background threshold
        self.bg_slider = SliderWithButtons("Background Threshold", 1, 100, 15, step=1, decimals=0)
        self.bg_slider.valueChanged.connect(self._schedule_full_update)
        params_layout.addWidget(self.bg_slider)

        # Detection sensitivity (controls edge detection and margin)
        self.sensitivity_slider = SliderWithButtons("Detection Sensitivity", 1, 100, 100, step=1, decimals=0)
        self.sensitivity_slider.setToolTip("Higher = more aggressive edge detection and cropping")
        self.sensitivity_slider.valueChanged.connect(self._schedule_content_update)
        params_layout.addWidget(self.sensitivity_slider)

        layout.addWidget(params_group)

        # Base Color section
        base_group = QGroupBox("Base Color")
        base_layout = QVBoxLayout(base_group)

        # Color swatch and values
        color_row = QHBoxLayout()
        self.base_color_swatch = QLabel()
        self.base_color_swatch.setFixedSize(40, 40)
        self.base_color_swatch.setStyleSheet("background-color: #808080; border: 2px solid #666;")
        color_row.addWidget(self.base_color_swatch)

        self.base_color_label = QLabel("R: --  G: --  B: --")
        self.base_color_label.setStyleSheet("font-family: 'Courier New', Monaco, monospace;")
        color_row.addWidget(self.base_color_label)
        color_row.addStretch()

        base_layout.addLayout(color_row)

        # Brightness threshold slider for auto-detect
        threshold_header = QHBoxLayout()
        threshold_header.addWidget(QLabel("Min Brightness:"))
        self._brightness_threshold_label = QLabel("80")
        threshold_header.addStretch()
        threshold_header.addWidget(self._brightness_threshold_label)
        base_layout.addLayout(threshold_header)

        self._brightness_threshold_slider = QSlider(Qt.Horizontal)
        self._brightness_threshold_slider.setRange(10, 200)
        self._brightness_threshold_slider.setValue(80)
        self._brightness_threshold_slider.setToolTip("Minimum brightness to consider as film base (rejects dark areas)")
        self._brightness_threshold_slider.valueChanged.connect(
            lambda v: self._brightness_threshold_label.setText(str(v))
        )
        base_layout.addWidget(self._brightness_threshold_slider)

        # Re-detect button
        self.auto_detect_btn = QPushButton("Re-detect Base")
        self.auto_detect_btn.clicked.connect(self._auto_detect_base)
        base_layout.addWidget(self.auto_detect_btn)

        layout.addWidget(base_group)

        # Crop adjustment shortcuts (QShortcut works regardless of widget focus)
        # Command + arrows = top/right crop edges
        QShortcut(QKeySequence("Ctrl+Up"), self).activated.connect(lambda: self._adjust_crop_edge('top', 1))
        QShortcut(QKeySequence("Ctrl+Down"), self).activated.connect(lambda: self._adjust_crop_edge('top', -1))
        QShortcut(QKeySequence("Ctrl+Left"), self).activated.connect(lambda: self._adjust_crop_edge('right', -1))
        QShortcut(QKeySequence("Ctrl+Right"), self).activated.connect(lambda: self._adjust_crop_edge('right', 1))

        # Option + arrows = bottom/left crop edges
        QShortcut(QKeySequence("Alt+Up"), self).activated.connect(lambda: self._adjust_crop_edge('bottom', -1))
        QShortcut(QKeySequence("Alt+Down"), self).activated.connect(lambda: self._adjust_crop_edge('bottom', 1))
        QShortcut(QKeySequence("Alt+Left"), self).activated.connect(lambda: self._adjust_crop_edge('left', 1))
        QShortcut(QKeySequence("Alt+Right"), self).activated.connect(lambda: self._adjust_crop_edge('left', -1))

        # Shift+Command + arrows = top/right crop edges (5px)
        QShortcut(QKeySequence("Ctrl+Shift+Up"), self).activated.connect(lambda: self._adjust_crop_edge('top', 5))
        QShortcut(QKeySequence("Ctrl+Shift+Down"), self).activated.connect(lambda: self._adjust_crop_edge('top', -5))
        QShortcut(QKeySequence("Ctrl+Shift+Left"), self).activated.connect(lambda: self._adjust_crop_edge('right', -5))
        QShortcut(QKeySequence("Ctrl+Shift+Right"), self).activated.connect(lambda: self._adjust_crop_edge('right', 5))

        # Shift+Option + arrows = bottom/left crop edges (5px)
        QShortcut(QKeySequence("Alt+Shift+Up"), self).activated.connect(lambda: self._adjust_crop_edge('bottom', -5))
        QShortcut(QKeySequence("Alt+Shift+Down"), self).activated.connect(lambda: self._adjust_crop_edge('bottom', 5))
        QShortcut(QKeySequence("Alt+Shift+Left"), self).activated.connect(lambda: self._adjust_crop_edge('left', 5))
        QShortcut(QKeySequence("Alt+Shift+Right"), self).activated.connect(lambda: self._adjust_crop_edge('left', -5))

        # Transform controls at the bottom
        self._detection_transform_widget = TransformControlsWidget(self._transform_state)
        layout.addWidget(self._detection_transform_widget)

        layout.addStretch()

        return sidebar

    def _create_debug_panel(self) -> CollapsibleDebugPanel:
        """Create the collapsible debug sidebar with processing pipeline panels."""
        self._debug_panel = CollapsibleDebugPanel()

        # Wire up signals from panels in the debug sidebar
        self._debug_panel.panel_base.positionChanged.connect(self._on_base_position_changed)

        # Create convenience references
        self.panel_negative = self._debug_panel.panel_negative
        self.panel_base = self._debug_panel.panel_base
        self.panel_edges = self._debug_panel.panel_edges
        self.panel_extracted = self._debug_panel.panel_extracted

        return self._debug_panel

    def _wrap_panel(self, title: str, panel: QWidget) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(panel)
        return group

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._load_image)
        file_menu.addAction(open_action)

        export_action = QAction("Export Frame...", self)
        export_action.setShortcut(QKeySequence.Save)
        export_action.triggered.connect(self._export_frame)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        prev_action = QAction("Previous Image", self)
        prev_action.setShortcut(QKeySequence("Up"))
        prev_action.triggered.connect(self._prev_image)
        file_menu.addAction(prev_action)

        next_action = QAction("Next Image", self)
        next_action.setShortcut(QKeySequence("Down"))
        next_action.triggered.connect(self._next_image)
        file_menu.addAction(next_action)

        file_menu.addSeparator()

        favorite_action = QAction("Toggle Favorite", self)
        favorite_action.setShortcut(QKeySequence("Ctrl+F"))
        favorite_action.triggered.connect(self._toggle_current_favorite)
        file_menu.addAction(favorite_action)

        filter_favorites_action = QAction("Show Favorites Only", self)
        filter_favorites_action.setShortcut(QKeySequence("Ctrl+Shift+F"))
        filter_favorites_action.triggered.connect(self._toggle_favorites_filter)
        file_menu.addAction(filter_favorites_action)

        file_menu.addSeparator()

        settings_action = QAction("Settings...", self)
        settings_action.setShortcut(QKeySequence(","))
        settings_action.triggered.connect(self._show_settings)
        file_menu.addAction(settings_action)

        keybindings_action = QAction("Keyboard Shortcuts...", self)
        keybindings_action.setShortcut(QKeySequence("?"))
        keybindings_action.triggered.connect(self._show_keybindings)
        file_menu.addAction(keybindings_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        # Fullscreen toggle (F): hide/show both panels
        fullscreen_action = QAction("Toggle Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("f"))
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        view_menu.addSeparator()

        # Left panel toggle (` and §): Debug on Detection, Presets on Development
        toggle_left_action = QAction("Toggle Left Panel", self)
        toggle_left_action.setShortcut(QKeySequence("`"))
        toggle_left_action.triggered.connect(self._toggle_left_panel)
        view_menu.addAction(toggle_left_action)

        toggle_left_action2 = QAction("Toggle Left Panel", self)
        toggle_left_action2.setShortcut(QKeySequence("§"))
        toggle_left_action2.triggered.connect(self._toggle_left_panel)
        self.addAction(toggle_left_action2)

        # Right panel toggle (~ and ±): Controls on Detection, Adjustments on Development
        toggle_right_action = QAction("Toggle Right Panel", self)
        toggle_right_action.setShortcut(QKeySequence("~"))
        toggle_right_action.triggered.connect(self._toggle_right_panel)
        view_menu.addAction(toggle_right_action)

        toggle_right_action2 = QAction("Toggle Right Panel", self)
        toggle_right_action2.setShortcut(QKeySequence("±"))
        toggle_right_action2.triggered.connect(self._toggle_right_panel)
        self.addAction(toggle_right_action2)

        # Preset shortcuts (1-9)
        view_menu.addSeparator()
        for i in range(1, 10):
            preset_action = QAction(f"Apply Preset {i}", self)
            preset_action.setShortcut(QKeySequence(str(i)))
            # Use lambda with default argument to capture current value of i
            preset_action.triggered.connect(lambda checked, num=i: self.adjustments_view.apply_preset_by_number(num))
            view_menu.addAction(preset_action)

        # Preset navigation shortcuts
        view_menu.addSeparator()
        prev_preset_action = QAction("Previous Preset", self)
        prev_preset_action.setShortcut(QKeySequence("Shift+Up"))
        prev_preset_action.triggered.connect(self.adjustments_view.select_previous_preset)
        view_menu.addAction(prev_preset_action)

        next_preset_action = QAction("Next Preset", self)
        next_preset_action.setShortcut(QKeySequence("Shift+Down"))
        next_preset_action.triggered.connect(self.adjustments_view.select_next_preset)
        view_menu.addAction(next_preset_action)

        # Preset reordering shortcuts (Ctrl+Alt to avoid conflict with crop Ctrl+Shift)
        move_up_action = QAction("Move Preset Up", self)
        move_up_action.setShortcut(QKeySequence("Ctrl+Alt+Up"))
        move_up_action.triggered.connect(self.adjustments_view.move_current_preset_up)
        view_menu.addAction(move_up_action)

        move_down_action = QAction("Move Preset Down", self)
        move_down_action.setShortcut(QKeySequence("Ctrl+Alt+Down"))
        move_down_action.triggered.connect(self.adjustments_view.move_current_preset_down)
        view_menu.addAction(move_down_action)

        view_menu.addSeparator()

        # Rotation menu items (shortcuts handled via QShortcut in _create_detection_sidebar)
        rotate_ccw_action = QAction("Rotate Counter-Clockwise", self)
        rotate_ccw_action.setShortcut(QKeySequence(Qt.Key_Left))
        rotate_ccw_action.triggered.connect(lambda: self._rotate_image(-90))
        view_menu.addAction(rotate_ccw_action)

        rotate_cw_action = QAction("Rotate Clockwise", self)
        rotate_cw_action.setShortcut(QKeySequence(Qt.Key_Right))
        rotate_cw_action.triggered.connect(lambda: self._rotate_image(90))
        view_menu.addAction(rotate_cw_action)

        rotate_180_action = QAction("Rotate 180°", self)
        rotate_180_action.triggered.connect(lambda: self._rotate_image(180))
        view_menu.addAction(rotate_180_action)

        view_menu.addSeparator()

        # Tab switching (handled in keyPressEvent for reliability)
        switch_tab_action = QAction("Switch Tab", self)
        switch_tab_action.setShortcut(QKeySequence(Qt.Key_Tab))
        switch_tab_action.triggered.connect(self._toggle_tab)
        view_menu.addAction(switch_tab_action)

    def _toggle_tab(self):
        """Toggle between Detection and Development tabs."""
        current = self.tab_bar.currentIndex()
        self.tab_bar.setCurrentIndex(1 - current)

    def _toggle_left_panel(self):
        """Toggle the left panel based on current tab."""
        if self.tab_bar.currentIndex() == 0:  # Detection tab
            self._debug_panel.toggle()
        else:  # Development tab
            self.adjustments_view.toggle_preset_panel()

    def _toggle_right_panel(self):
        """Toggle the right panel based on current tab."""
        if self.tab_bar.currentIndex() == 0:  # Detection tab
            self._controls_panel.toggle()
        else:  # Development tab
            self.adjustments_view.toggle_adjustments_panel()

    def _toggle_fullscreen(self):
        """Toggle both panels for fullscreen view.

        If either panel is open, close both.
        If both panels are closed, open both.
        """
        if self.tab_bar.currentIndex() == 0:  # Detection tab
            left_expanded = self._debug_panel.is_expanded()
            right_expanded = self._controls_panel.is_expanded()

            if left_expanded or right_expanded:
                # Close both
                if left_expanded:
                    self._debug_panel.toggle()
                if right_expanded:
                    self._controls_panel.toggle()
            else:
                # Open both
                self._debug_panel.toggle()
                self._controls_panel.toggle()
        else:  # Development tab
            left_expanded = self.adjustments_view._preset_panel.is_expanded()
            right_expanded = self.adjustments_view._adjustments_panel.is_expanded()

            if left_expanded or right_expanded:
                # Close both
                if left_expanded:
                    self.adjustments_view.toggle_preset_panel()
                if right_expanded:
                    self.adjustments_view.toggle_adjustments_panel()
            else:
                # Open both
                self.adjustments_view.toggle_preset_panel()
                self.adjustments_view.toggle_adjustments_panel()

    def _load_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Image(s)", str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.nef *.cr2 *.cr3 *.arw *.dng *.orf *.rw2 *.raf *.pef *.srw);;All Files (*)"
        )
        if paths:
            self.file_list = sorted(paths)
            self.file_index = 0
            self._compute_file_hashes()
            hashes = [self._file_hashes.get(p) for p in self.file_list]
            self.thumbnail_bar.set_files(self.file_list, hashes)
            self._load_current_file()

    def _load_image_from_path(self, path: str):
        """Load a single image (adds to file list if not already there)."""
        if path not in self.file_list:
            self.file_list = [path]
            self.file_index = 0
            self._compute_file_hashes()
            hashes = [self._file_hashes.get(p) for p in self.file_list]
            self.thumbnail_bar.set_files(self.file_list, hashes)
        self._load_current_file()

    def set_file_list(self, paths: list):
        """Set multiple files from CLI."""
        valid_paths = [p for p in paths if Path(p).exists()]
        if valid_paths:
            self.file_list = valid_paths
            self.file_index = 0
            self._compute_file_hashes()
            hashes = [self._file_hashes.get(p) for p in self.file_list]
            self.thumbnail_bar.set_files(valid_paths, hashes)
            self._load_current_file()

    def _compute_file_hashes(self):
        """Compute and cache hashes for all files in file_list."""
        for path in self.file_list:
            if path not in self._file_hashes:
                self._file_hashes[path] = self._compute_image_hash(path)

    def _load_current_file(self):
        """Load the current file from the file list."""
        import time
        _t_start = time.time()

        if not self.file_list or self.file_index >= len(self.file_list):
            return

        # Exit crop mode when switching images (crop adjustments are per-image)
        if self._crop_mode_active:
            self._transform_state.crop_mode = False

        # Clear debug sample boxes from previous image
        self.panel_base._debug_sample_boxes = []

        path = self.file_list[self.file_index]
        _t0 = time.time()
        try:
            img = load_image(path)
        except (FileNotFoundError, ImportError):
            return
        _t_load = time.time() - _t0

        self.current_image_original = img
        self.current_path = path

        # Use cached hash or compute if not available
        _t0 = time.time()
        self._current_hash = self._file_hashes.get(path) or self._compute_image_hash(path)
        _t_hash = time.time() - _t0

        # Load per-image settings (block signals to avoid multiple reprocessing)
        self.bg_slider.blockSignals(True)
        self.sensitivity_slider.blockSignals(True)

        self._load_settings_for_image(path)

        self.bg_slider.blockSignals(False)
        self.sensitivity_slider.blockSignals(False)

        # Apply rotation (from settings or default 0)
        _t0 = time.time()
        self.current_rotation = self._pending_rotation if self._pending_rotation is not None else 0
        self._pending_rotation = None
        self.current_image = self._apply_rotation(self.current_image_original, self.current_rotation)
        self._update_rotation_reset_button_styles()
        # Sync TransformState with loaded rotation (without triggering signal)
        self._transform_state._rotation = self.current_rotation
        _t_rotate = time.time() - _t0

        self._is_new_image = True
        self._update_nav_state()

        _t0 = time.time()
        self._process_full()
        _t_process = time.time() - _t0

        _t_total = time.time() - _t_start
        print(f"[PERF] load={_t_load*1000:.0f}ms hash={_t_hash*1000:.0f}ms rotate={_t_rotate*1000:.0f}ms process={_t_process*1000:.0f}ms | TOTAL={_t_total*1000:.0f}ms")

    def _update_nav_state(self):
        """Update navigation state."""
        count = len(self.file_list)
        self.thumbnail_bar.select_index(self.file_index)

        if count > 0:
            name = Path(self.file_list[self.file_index]).name
            self.setWindowTitle(f"SUPER NEGATIVE PROCESSING SYSTEM - {name} ({self.file_index + 1}/{count})")
        else:
            self.setWindowTitle("SUPER NEGATIVE PROCESSING SYSTEM")

    def _compute_image_hash(self, path: str) -> str:
        """Compute SHA-256 hash of image file for identification."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _save_current_settings(self):
        """Save settings and thumbnail for current image to SQLite."""
        if self.current_path and self._current_hash:
            preset_state = self.adjustments_view.get_preset_state()
            settings = {
                'bg': self.bg_slider.value(),
                'sensitivity': self.sensitivity_slider.value(),
                'fine_rotation': self._transform_state.fine_rotation,
                'crop_adjustment': self._crop_adjustment.copy(),
                'base_pos': self.panel_base.get_position(),
                'rotation': self.current_rotation,
                'preset_state': preset_state,
            }
            self.image_settings[self.current_path] = settings

            # Save to SQLite with thumbnail
            thumbnail = getattr(self, '_current_inverted', None)
            self._storage.save_all(self._current_hash, settings, thumbnail)

    def _load_settings_for_image(self, path: str):
        """Load settings for an image from SQLite, or use defaults."""
        self._pending_base_pos = None
        self._pending_rotation = None

        # Check in-memory settings first, then SQLite by hash
        settings = None
        if path in self.image_settings:
            settings = self.image_settings[path]
        elif self._current_hash:
            settings = self._storage.load_settings(self._current_hash)
            if settings:
                self.image_settings[path] = settings

        if settings:
            self.bg_slider.setValue(settings.get('bg', self.bg_slider.default))

            # Handle sensitivity - migrate from old inset if needed
            if 'sensitivity' in settings:
                self.sensitivity_slider.setValue(settings['sensitivity'])
            elif 'inset' in settings:
                # Migrate old inset (0-100) to sensitivity (centered at 50)
                old_inset = settings.get('inset', 0)
                migrated_sensitivity = 50 + int(old_inset / 2)
                self.sensitivity_slider.setValue(migrated_sensitivity)
            else:
                self.sensitivity_slider.setValue(self.sensitivity_slider.default)

            fine_rot = settings.get('fine_rotation', 0)
            self._transform_state.fine_rotation = fine_rot

            # Load preset state
            preset_state = settings.get('preset_state', {'active_preset': 'none', 'preset_states': {}})
            self.adjustments_view.set_preset_state(preset_state)

            # Load adjustments/curves for the active preset
            active = preset_state.get('active_preset', 'none')
            preset_states = preset_state.get('preset_states', {})
            if active in preset_states:
                state = preset_states[active]
                self.adjustments_view.set_adjustments(state.get('adjustments', {}))
                curves = state.get('curves', {})
                if curves:
                    self.adjustments_view.curves_widget.set_curves(curves)
                else:
                    self.adjustments_view.curves_widget.reset()
            else:
                # Active preset not in saved states, load from preset defaults
                preset_defaults = presets.get_preset(active)
                self.adjustments_view.set_adjustments(preset_defaults.get('adjustments', {}))
                curves = preset_defaults.get('curves', {})
                if curves:
                    self.adjustments_view.curves_widget.set_curves(curves)
                else:
                    self.adjustments_view.curves_widget.reset()

            # Load crop adjustment (new format)
            if 'crop_adjustment' in settings:
                self._crop_adjustment = settings['crop_adjustment'].copy()
            else:
                self._crop_adjustment = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
            self._update_crop_reset_button_styles()

            # Store base_pos, rotation to apply after image is loaded
            if 'base_pos' in settings:
                self._pending_base_pos = settings['base_pos']
            if 'rotation' in settings:
                self._pending_rotation = settings['rotation']
                self._needs_auto_rotate = False  # Has saved rotation, don't auto-rotate
            else:
                self._needs_auto_rotate = True  # No saved rotation, try auto-detect
        else:
            self._needs_auto_rotate = True  # No settings at all, try auto-detect
            # Reset to defaults
            self.bg_slider.setValue(self.bg_slider.default)
            self.sensitivity_slider.setValue(self.sensitivity_slider.default)
            self._transform_state.fine_rotation = 0
            self.adjustments_view.curves_widget.reset()
            self.adjustments_view.reset_adjustments()
            self.adjustments_view.set_preset_state({'active_preset': 'none', 'preset_states': {}})
            self._crop_adjustment = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
            self._update_crop_reset_button_styles()

    def _on_thumbnail_selected(self, index: int):
        """Handle thumbnail click."""
        if index != self.file_index:
            self._save_current_settings()
            self.file_index = index
            self._load_current_file()

    def _prev_image(self):
        if self.thumbnail_bar.is_filtering_favorites():
            prev_idx = self.thumbnail_bar.get_prev_visible_index(self.file_index)
            if prev_idx >= 0:
                self._save_current_settings()
                self.file_index = prev_idx
                self._load_current_file()
        elif self.file_index > 0:
            self._save_current_settings()
            self.file_index -= 1
            self._load_current_file()

    def _next_image(self):
        if self.thumbnail_bar.is_filtering_favorites():
            next_idx = self.thumbnail_bar.get_next_visible_index(self.file_index)
            if next_idx >= 0:
                self._save_current_settings()
                self.file_index = next_idx
                self._load_current_file()
        elif self.file_index < len(self.file_list) - 1:
            self._save_current_settings()
            self.file_index += 1
            self._load_current_file()

    def _toggle_current_favorite(self):
        """Toggle favorite status for the current image."""
        if self.file_index >= 0 and self.file_index < len(self.file_list):
            self.thumbnail_bar.toggle_favorite(self.file_index)

    def _toggle_favorites_filter(self):
        """Toggle the favorites-only filter."""
        self.thumbnail_bar.toggle_favorites_filter()

    def _show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        dialog.open()  # Opens as modal via open() instead of exec()
        dialog.finished.connect(dialog.deleteLater)

    def _show_keybindings(self):
        """Show the keyboard shortcuts dialog."""
        dialog = KeybindingsDialog(self)
        dialog.open()
        dialog.finished.connect(dialog.deleteLater)

    def _on_favorite_toggled(self, index: int, is_favorite: bool):
        """Handle favorite toggle - persist to storage."""
        if 0 <= index < len(self.file_list):
            path = self.file_list[index]
            image_hash = self._file_hashes.get(path)
            if image_hash:
                favorites = self._storage.get_favorite_images()
                if is_favorite:
                    favorites.add(image_hash)
                else:
                    favorites.discard(image_hash)
                self._storage.set_favorite_images(favorites)

    def _schedule_full_update(self):
        self._needs_full_update = True
        self.update_timer.start(100)

    def _schedule_content_update(self):
        self._needs_full_update = False
        self.update_timer.start(50)

    def _do_process(self):
        if self.current_image is None:
            return
        if self._needs_full_update:
            self._process_full()
        else:
            self._process_content_only()

    def _process_full(self):
        import time
        if self.current_image is None:
            return

        img = self.current_image

        _t0 = time.time()
        self.negative_mask = self._isolate_negative(img, int(self.bg_slider.value()))
        self.panel_negative.set_image(self.negative_mask, is_mask=True)
        _t_isolate = time.time() - _t0

        # Set base selection panel image (still used for inversion color)
        self.panel_base.set_image(img)

        _t0 = time.time()
        self._process_content_only()
        _t_content = time.time() - _t0

        print(f"  [PERF process_full] isolate={_t_isolate*1000:.0f}ms content={_t_content*1000:.0f}ms")

    def _process_content_only(self):
        import time
        if self.current_image is None or self.negative_mask is None:
            return

        img = self.current_image

        # Map sensitivity to detection parameters
        sensitivity = int(self.sensitivity_slider.value())
        canny_low, canny_high, margin = self._map_sensitivity(sensitivity)
        fine_rotation = self._transform_state.fine_rotation

        # Run new edge-based detection
        _t0 = time.time()
        extracted, detected_angle, edge_vis = self._auto_crop_rotate(
            img,
            negative_mask=self.negative_mask,
            canny_low=canny_low,
            canny_high=canny_high,
            margin=margin,
            fine_rotation=fine_rotation
        )
        _t_crop = time.time() - _t0

        # Show edge detection visualization
        self.panel_edges.set_image(edge_vis)

        # Update extracted frame display (crop adjustment is already applied in _auto_crop_rotate)
        self.panel_extracted.set_image(extracted)

        # Handle base position for new images
        if self._is_new_image:
            if self._pending_base_pos:
                self.panel_base.set_position(self._pending_base_pos)
            else:
                # Auto-detect base position for new images
                self.panel_base.auto_detect_base(self.negative_mask)

            self._pending_base_pos = None
            self._is_new_image = False

        # Update base color display and inverted panel
        self._update_base_color_display()

        _t0 = time.time()
        self._update_inverted()
        _t_invert = time.time() - _t0

        print(f"    [PERF content] crop={_t_crop*1000:.0f}ms invert={_t_invert*1000:.0f}ms")

    def _isolate_negative(self, img: np.ndarray, threshold: int = 15) -> np.ndarray:
        # Convert float32 (0-1) to uint8 for thresholding
        if img.dtype == np.float32:
            img_8bit = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img_8bit = img

        gray = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _map_sensitivity(self, sensitivity: int) -> tuple:
        """Map sensitivity slider (1-100) to detection parameters.

        Returns: (canny_low, canny_high, margin)
        """
        # Higher sensitivity = lower thresholds (more edges detected)
        canny_low = max(20, int(150 - sensitivity * 1.2))
        canny_high = max(50, int(300 - sensitivity * 2.5))
        # Higher sensitivity = larger margin (more aggressive crop)
        margin = 0.01 + (sensitivity / 100) * 0.09  # 1%-10%
        return canny_low, canny_high, margin

    def _auto_crop_rotate(self, image: np.ndarray, negative_mask: np.ndarray = None,
                          canny_low: int = 50, canny_high: int = 150,
                          margin: float = 0.05, fine_rotation: float = 0.0) -> tuple:
        """Detect frame angle using Hough Lines and extract straightened crop.

        Args:
            image: Input BGR image
            negative_mask: Optional mask to constrain detection
            canny_low: Lower Canny threshold
            canny_high: Upper Canny threshold
            margin: Fractional margin to crop inward (0.05 = 5%)
            fine_rotation: Additional manual rotation adjustment

        Returns:
            (extracted_image, detected_angle, edge_visualization)
        """
        h, w = image.shape[:2]

        # 1. Work on grayscale small copy for speed
        scale = 800 / max(h, w)
        if scale >= 1:
            scale = 1

        small = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # Convert float32 (0-1) to uint8 for Canny edge detection
        if small.dtype == np.float32:
            small = (np.clip(small, 0, 1) * 255).astype(np.uint8)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, canny_low, canny_high)

        # Apply negative mask if provided (scaled down)
        if negative_mask is not None:
            small_mask = cv2.resize(negative_mask, (small.shape[1], small.shape[0]))
            edges = cv2.bitwise_and(edges, small_mask)

        # 2. Detect Angle using Hough Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        rotation_angle = 0.0

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:  # Vertical line
                    angle = 90
                else:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                # Normalize angle to deviation from horizontal (0) or vertical (90)
                if -45 <= angle <= 45:
                    angles.append(angle)
                elif 45 < angle <= 135:
                    angles.append(angle - 90)
                elif -135 <= angle < -45:
                    angles.append(angle + 90)

            if angles:
                rotation_angle = np.median(angles)

        # Add fine rotation adjustment
        total_rotation = rotation_angle + fine_rotation

        # 3. Create edge visualization
        edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edge_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(edge_vis, f"Angle: {rotation_angle:.1f} + {fine_rotation:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 4. Straighten the small image to find precise crop box
        h_s, w_s = small.shape[:2]
        center_s = (w_s // 2, h_s // 2)
        M_s = cv2.getRotationMatrix2D(center_s, total_rotation, 1.0)
        small_straight = cv2.warpAffine(small, M_s, (w_s, h_s),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)

        # 5. Find Contour on Straightened Image
        gray_s = cv2.cvtColor(small_straight, cv2.COLOR_BGR2GRAY)
        blur_s = cv2.GaussianBlur(gray_s, (7, 7), 0)
        edges_s = cv2.Canny(blur_s, canny_low, canny_high)
        dilated_s = cv2.dilate(edges_s, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(dilated_s, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: Just rotate full image
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, total_rotation, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            self._rotated_full_image = rotated.copy()
            base_color = self.panel_base.get_sampled_color_bgr()
            if base_color is not None:
                self._rotated_full_image_inverted = self._invert_negative(rotated, base_color)
            else:
                # Simple inversion fallback (handle both float32 and uint8)
                if rotated.dtype == np.float32:
                    self._rotated_full_image_inverted = 1.0 - rotated
                else:
                    self._rotated_full_image_inverted = 255 - rotated
            self._auto_detected_bounds = (0, 0, w, h)
            return rotated, rotation_angle, edge_vis

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < (small.shape[0] * small.shape[1] * 0.1):
            # Too small, just rotate
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, total_rotation, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            self._rotated_full_image = rotated.copy()
            base_color = self.panel_base.get_sampled_color_bgr()
            if base_color is not None:
                self._rotated_full_image_inverted = self._invert_negative(rotated, base_color)
            else:
                # Simple inversion fallback (handle both float32 and uint8)
                if rotated.dtype == np.float32:
                    self._rotated_full_image_inverted = 1.0 - rotated
                else:
                    self._rotated_full_image_inverted = 255 - rotated
            self._auto_detected_bounds = (0, 0, w, h)
            return rotated, rotation_angle, edge_vis

        # Get Axis-Aligned Bounding Box (since it's now straight)
        x_s, y_s, w_crop_s, h_crop_s = cv2.boundingRect(c)

        # 6. Apply to Full Resolution Image
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, total_rotation, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        # Store full rotated image for crop mode visualization
        self._rotated_full_image = rotated.copy()
        # Also create properly inverted version (with orange mask removal)
        base_color = self.panel_base.get_sampled_color_bgr()
        if base_color is not None:
            self._rotated_full_image_inverted = self._invert_negative(rotated, base_color)
        else:
            # Simple inversion fallback (handle both float32 and uint8)
            if rotated.dtype == np.float32:
                self._rotated_full_image_inverted = 1.0 - rotated
            else:
                self._rotated_full_image_inverted = 255 - rotated

        # Scale coordinates back to full size
        crop_x = int(x_s / scale)
        crop_y = int(y_s / scale)
        crop_w = int(w_crop_s / scale)
        crop_h = int(h_crop_s / scale)

        # Apply Margin
        crop_w_m = int(crop_w * (1 - margin))
        crop_h_m = int(crop_h * (1 - margin))

        diff_w = crop_w - crop_w_m
        diff_h = crop_h - crop_h_m

        crop_x += diff_w // 2
        crop_y += diff_h // 2
        crop_w = crop_w_m
        crop_h = crop_h_m

        # Store auto-detected bounds (before user adjustment)
        self._auto_detected_bounds = (crop_x, crop_y, crop_w, crop_h)

        # Apply user's crop adjustment (positive = expand, negative = contract)
        adj = self._crop_adjustment
        crop_x -= adj['left']
        crop_y -= adj['top']
        crop_w += adj['left'] + adj['right']
        crop_h += adj['top'] + adj['bottom']

        # Bounds check
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        crop_w = min(w - crop_x, crop_w)
        crop_h = min(h - crop_y, crop_h)

        if crop_w > 100 and crop_h > 100:
            extracted = rotated[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            return extracted, rotation_angle, edge_vis

        return rotated, rotation_angle, edge_vis

    def _apply_rotation(self, img: np.ndarray, degrees: int) -> np.ndarray:
        """Apply rotation to image. Degrees should be 0, 90, 180, or 270."""
        if img is None:
            return None
        degrees = degrees % 360
        if degrees == 0:
            return img.copy()
        elif degrees == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif degrees == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif degrees == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img.copy()

    def _rotate_image(self, delta: int):
        """Rotate the current image by delta degrees (90 or -90)."""
        if self.current_image_original is None:
            return

        # Exit crop mode when rotating (bounds become invalid until reprocessed)
        if self._crop_mode_active:
            self._transform_state.crop_mode = False

        # Get current base position before rotation
        old_base_pos = self.panel_base.get_position()  # (x, y) as percentages 0-1

        # Transform crop adjustment for the rotation
        # When rotating, edges move to different positions
        adj = self._crop_adjustment
        if delta == 90:  # Clockwise: left→top, top→right, right→bottom, bottom→left
            self._crop_adjustment = {
                'left': adj['bottom'],
                'top': adj['left'],
                'right': adj['top'],
                'bottom': adj['right']
            }
        elif delta == -90:  # Counter-clockwise: left→bottom, top→left, right→top, bottom→right
            self._crop_adjustment = {
                'left': adj['top'],
                'top': adj['right'],
                'right': adj['bottom'],
                'bottom': adj['left']
            }

        # Update rotation (normalize to 0, 90, 180, 270)
        self.current_rotation = (self.current_rotation + delta) % 360
        # Sync TransformState (without triggering signal)
        self._transform_state._rotation = self.current_rotation

        # Apply rotation to get new current image
        self.current_image = self._apply_rotation(self.current_image_original, self.current_rotation)

        # Save rotation to settings
        if self.current_path:
            if self.current_path not in self.image_settings:
                self.image_settings[self.current_path] = {}
            self.image_settings[self.current_path]['rotation'] = self.current_rotation

        # Update reset button styling
        self._update_rotation_reset_button_styles()

        # Transform base position for the rotation
        # Base position is (x, y) as percentages 0-1
        if old_base_pos:
            x, y = old_base_pos
            if delta == 90:  # Clockwise: (x, y) -> (1-y, x)
                new_base_pos = (1 - y, x)
            elif delta == -90:  # Counter-clockwise: (x, y) -> (y, 1-x)
                new_base_pos = (y, 1 - x)
            else:
                new_base_pos = None
            self._pending_base_pos = new_base_pos
        else:
            self._pending_base_pos = None

        # Reprocess
        self._is_new_image = True
        self._process_full()

    def _reset_rotation(self):
        """Reset the 90° rotation to 0°."""
        if self.current_image_original is None:
            return

        if self.current_rotation == 0:
            return  # Already at 0

        # Reset rotation
        self.current_rotation = 0
        # Sync TransformState (without triggering signal)
        self._transform_state._rotation = 0
        self.current_image = self.current_image_original.copy()

        # Save rotation to settings
        if self.current_path:
            if self.current_path not in self.image_settings:
                self.image_settings[self.current_path] = {}
            self.image_settings[self.current_path]['rotation'] = self.current_rotation

        # Update reset button styling
        self._update_rotation_reset_button_styles()

        # Reset crop adjustment since orientation changed significantly
        self._crop_adjustment = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}

        # Reprocess
        self._is_new_image = True
        self._process_full()

    def _auto_rotate_from_content(self):
        """Auto-detect rotation based on image content (faces, sky, gradients)."""
        # We need to detect on the image at 0° rotation to get absolute orientation
        # First, compute what the inverted image would look like at 0° rotation
        if self.current_image_original is None:
            return

        # Get the inverted image at 0° rotation for consistent detection
        # This avoids the issue of detecting on an already-rotated image
        test_image = self._get_inverted_at_zero_rotation()
        if test_image is None:
            return

        result = detect_auto_rotation(test_image)

        # Calculate what rotation we need to SET (not add)
        target_rotation = result.rotation

        if target_rotation == self.current_rotation:
            confidence_str = result.confidence.value
            print(f"[Auto-rotate] Already at correct rotation {self.current_rotation}° (confidence: {confidence_str}, method: {result.method})")
            return

        confidence_str = result.confidence.value
        print(f"[Auto-rotate] Setting rotation to {target_rotation}° (was {self.current_rotation}°, confidence: {confidence_str}, method: {result.method})")

        # Set absolute rotation (not delta)
        self._set_rotation_absolute(target_rotation)

    def _get_inverted_at_zero_rotation(self):
        """Get the inverted image as it would appear at 0° rotation.

        This ensures auto-rotation detection is always consistent regardless
        of current rotation state.
        """
        if self.current_image_original is None:
            return None

        # Quick processing pipeline at 0° rotation
        # 1. Isolate negative
        img = self.current_image_original
        threshold = int(self.bg_slider.value())
        negative_mask = self._isolate_negative(img, threshold)

        # 2. Get a quick crop (use simpler method for speed)
        h, w = img.shape[:2]
        margin = 0.05
        x1, y1 = int(w * margin), int(h * margin)
        x2, y2 = int(w * (1 - margin)), int(h * (1 - margin))
        extracted = img[y1:y2, x1:x2]

        # 3. Invert
        base_color = self.panel_base.get_sampled_color_bgr()
        if base_color is not None:
            inverted = self._invert_negative(extracted, base_color)
        else:
            # Simple inversion fallback (handle both float32 and uint8)
            if extracted.dtype == np.float32:
                inverted = 1.0 - extracted
            else:
                inverted = 255 - extracted

        return inverted

    def _set_rotation_absolute(self, degrees: int):
        """Set rotation to an absolute value (not delta)."""
        degrees = degrees % 360
        if degrees == self.current_rotation:
            return

        # Calculate delta from current
        delta = (degrees - self.current_rotation) % 360
        if delta > 180:
            delta -= 360  # Use shorter rotation path

        self._rotate_image(delta)

    def _try_auto_rotate_first_load(self):
        """Try auto-rotation on first image load (only if confident enough)."""
        # On first load, _current_inverted is at 0° rotation, so we can use it directly
        if self._current_inverted is None:
            return

        result = detect_auto_rotation(self._current_inverted)

        # Apply if we have any detection (HIGH, MEDIUM, or LOW confidence)
        if result.confidence != RotationConfidence.NONE and result.rotation != 0:
            confidence_str = result.confidence.value
            print(f"[Auto-rotate] First load: setting to {result.rotation}° (confidence: {confidence_str}, method: {result.method})")
            # Use QTimer to defer rotation to avoid recursion during _update_inverted
            QTimer.singleShot(0, lambda: self._set_rotation_absolute(result.rotation))
        else:
            confidence_str = result.confidence.value
            print(f"[Auto-rotate] First load: no rotation needed (confidence: {confidence_str}, method: {result.method})")

    def _update_rotation_reset_button_styles(self):
        """Highlight rotation reset button when rotation is non-zero."""
        if self.current_rotation != 0:
            style = "QPushButton { background-color: #e67e22; color: white; font-weight: bold; }"
        else:
            style = ""
        # Update transform widget reset button (Detection view only now)
        self._detection_transform_widget.reset_rotation_btn.setStyleSheet(style)

    def _update_crop_reset_button_styles(self):
        """Update crop reset button style based on crop adjustment values."""
        self._detection_transform_widget.set_crop_adjustment(self._crop_adjustment)

    def _toggle_grid(self, enabled: bool):
        """Toggle grid overlay on extracted frame and inverted panels."""
        self.panel_extracted.set_grid_enabled(enabled)
        self.panel_inverted.set_grid_enabled(enabled)

    def _update_grid_divisions(self, divisions: float):
        """Update grid divisions on both panels."""
        div = int(divisions)
        self.panel_extracted.set_grid_divisions(div)
        self.panel_inverted.set_grid_divisions(div)

    def _on_base_position_changed(self, pos: tuple):
        """Handle base position changes from the crosshair widget."""
        if self.current_path:
            if self.current_path not in self.image_settings:
                self.image_settings[self.current_path] = {}
            self.image_settings[self.current_path]['base_pos'] = pos
        self._update_base_color_display()
        self._update_inverted()

    def _auto_detect_base(self):
        """Auto-detect the base color position."""
        if self.current_image is None or self.negative_mask is None:
            return
        threshold = self._brightness_threshold_slider.value()
        self.panel_base.auto_detect_base(self.negative_mask, threshold)

    def _toggle_crop_mode(self, enabled: bool):
        """Toggle crop mode on/off."""
        self._crop_mode_active = enabled

        # Update both views with crop mode data
        self._update_crop_overlay()

        # Update adjustments view crop mode too
        if hasattr(self, 'adjustments_view'):
            self.adjustments_view.set_crop_mode(
                enabled,
                full_image=self._rotated_full_image,
                inverted_image=self._rotated_full_image_inverted,
                bounds=self._auto_detected_bounds,
                adjustment=self._crop_adjustment
            )

        # Restore invert state when entering crop mode
        if enabled:
            invert_state = self._transform_state.crop_invert
            self._on_crop_invert_changed(invert_state)

    def _reset_crop_adjustment(self):
        """Reset crop adjustment to auto-detected bounds."""
        self._crop_adjustment = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
        self._update_crop_reset_button_styles()
        self._update_all_crop_overlays()
        self._schedule_content_update()

    def _toggle_crop_invert(self):
        """Toggle invert state in crop mode via shared transform state."""
        self._transform_state.crop_invert = not self._transform_state.crop_invert

    def _adjust_crop_edge(self, edge: str, delta: int):
        """Adjust a crop edge by delta pixels. Positive = expand, negative = contract."""
        # Apply the primary edge adjustment
        self._crop_adjustment[edge] += delta

        # Enforce aspect ratio if locked
        target_ratio = self._transform_state.get_aspect_ratio_value()
        if target_ratio is not None and self._auto_detected_bounds is not None:
            self._enforce_aspect_ratio(edge, target_ratio)

        # Update crop reset button style
        self._update_crop_reset_button_styles()

        # Flash the edge red for visual feedback
        self._flash_edge = edge
        self._flash_timer.start(150)  # Flash for 150ms
        # Update the overlays immediately for visual feedback
        self._update_all_crop_overlays()
        # Also flash the main view panels border in normal view
        if not self._crop_mode_active:
            self.panel_inverted.set_border_flash(edge)
            self.adjustments_view._preview.set_border_flash(edge)
        # Schedule a content update to reprocess with new crop
        self._schedule_content_update()

    def _enforce_aspect_ratio(self, adjusted_edge: str, target_ratio: float):
        """Adjust perpendicular edges to maintain aspect ratio after an edge change.

        For edge (center handle) dragging, the opposite center point should stay
        anchored, so we adjust BOTH perpendicular edges equally.

        Args:
            adjusted_edge: The edge that was just adjusted ('left', 'right', 'top', 'bottom')
            target_ratio: Target width/height ratio (e.g., 1.5 for 3:2)
        """
        if self._auto_detected_bounds is None:
            return

        base_x, base_y, base_w, base_h = self._auto_detected_bounds
        adj = self._crop_adjustment

        # Current crop dimensions
        current_w = base_w + adj['left'] + adj['right']
        current_h = base_h + adj['top'] + adj['bottom']

        if current_w <= 0 or current_h <= 0:
            return

        # Use original detected bounds to determine orientation (stable reference)
        is_portrait = base_h > base_w
        effective_ratio = 1.0 / target_ratio if is_portrait else target_ratio

        # Determine if we adjusted a horizontal or vertical edge
        if adjusted_edge in ('left', 'right'):
            # Width changed, adjust height to match ratio
            # Keep vertical center fixed by adjusting top/bottom equally
            target_h = current_w / effective_ratio
            delta_h = target_h - current_h
            half_delta = round(delta_h / 2)
            adj['top'] += half_delta
            adj['bottom'] += round(delta_h) - half_delta  # Handle odd pixels
        else:
            # Height changed, adjust width to match ratio
            # Keep horizontal center fixed by adjusting left/right equally
            target_w = current_h * effective_ratio
            delta_w = target_w - current_w
            half_delta = round(delta_w / 2)
            adj['left'] += half_delta
            adj['right'] += round(delta_w) - half_delta  # Handle odd pixels

    def _clear_flash_edge(self):
        """Clear the flashing edge after timer expires."""
        self._flash_edge = None
        self._update_all_crop_overlays()
        # Also clear the main view panels border flash
        self.panel_inverted.set_border_flash(None)
        self.adjustments_view._preview.set_border_flash(None)

    def _on_crop_edge_dragged(self, edge: str, delta: int):
        """Handle crop edge drag from ImagePanel."""
        self._adjust_crop_edge(edge, delta)

    def _on_crop_corner_drag_started(self, corner: str):
        """Capture crop adjustment values when corner drag starts."""
        h_edge = 'left' if 'left' in corner else 'right'
        v_edge = 'top' if 'top' in corner else 'bottom'
        self._corner_drag_start = {
            'corner': corner,
            'h_edge': h_edge,
            'v_edge': v_edge,
            'h_val': self._crop_adjustment[h_edge],
            'v_val': self._crop_adjustment[v_edge],
            'dominant_axis': None,  # Will be locked once movement exceeds threshold
        }

    def _on_crop_corner_dragged(self, corner: str, delta_x: int, delta_y: int):
        """Handle crop corner drag - anchors the opposite corner.

        Receives cumulative deltas from drag start. Constrains for aspect ratio
        and applies values absolutely based on start values.
        """
        if self._auto_detected_bounds is None:
            return

        # Ensure we have start values
        if self._corner_drag_start is None or self._corner_drag_start['corner'] != corner:
            # Fallback: capture now (shouldn't happen normally)
            self._on_crop_corner_drag_started(corner)

        start = self._corner_drag_start
        h_edge = start['h_edge']
        v_edge = start['v_edge']

        # Convert mouse deltas to expansion values (positive = box gets bigger)
        if 'left' in corner:
            exp_w = -delta_x  # Moving left expands width
        else:
            exp_w = delta_x   # Moving right expands width

        if 'top' in corner:
            exp_h = -delta_y  # Moving up expands height
        else:
            exp_h = delta_y   # Moving down expands height

        # Get aspect ratio (width/height)
        target_ratio = self._transform_state.get_aspect_ratio_value()

        if target_ratio is not None and (exp_w != 0 or exp_h != 0):
            # Use original detected bounds to determine orientation (stable reference)
            base_x, base_y, base_w, base_h = self._auto_detected_bounds
            is_portrait = base_h > base_w
            effective_ratio = 1.0 / target_ratio if is_portrait else target_ratio

            # Determine dominant axis - lock it once movement exceeds threshold
            # This prevents snapping when crossing the axis equivalence point
            w_as_h = abs(exp_w) / effective_ratio
            h_magnitude = abs(exp_h)

            LOCK_THRESHOLD = 10  # Lock axis after 10px of equivalent movement

            if start['dominant_axis'] is None:
                # Not yet locked - check if we should lock
                if w_as_h >= LOCK_THRESHOLD or h_magnitude >= LOCK_THRESHOLD:
                    # Lock to whichever is larger
                    start['dominant_axis'] = 'width' if w_as_h >= h_magnitude else 'height'

            # Use locked axis if available, otherwise use current dominant
            if start['dominant_axis'] == 'width':
                # Width dominates, derive height from it
                exp_h = round(exp_w / effective_ratio)
            elif start['dominant_axis'] == 'height':
                # Height dominates, derive width from it
                exp_w = round(exp_h * effective_ratio)
            else:
                # Not yet locked - use current dominant (for small movements)
                if w_as_h >= h_magnitude:
                    exp_h = round(exp_w / effective_ratio)
                else:
                    exp_w = round(exp_h * effective_ratio)

        # Calculate new absolute values from start values
        new_h_val = start['h_val'] + exp_w
        new_v_val = start['v_val'] + exp_h

        # Apply absolutely (not incrementally)
        self._crop_adjustment[h_edge] = new_h_val
        self._crop_adjustment[v_edge] = new_v_val

        self._update_crop_reset_button_styles()
        self._update_all_crop_overlays()
        self._schedule_content_update()

    def _on_crop_box_moved(self, delta_x: int, delta_y: int):
        """Handle crop box movement (all edges move together)."""
        # Moving the box right means: left edge expands, right edge contracts
        # Moving the box down means: top edge expands, bottom edge contracts
        self._crop_adjustment['left'] -= delta_x
        self._crop_adjustment['right'] += delta_x
        self._crop_adjustment['top'] -= delta_y
        self._crop_adjustment['bottom'] += delta_y
        # Update crop reset button style
        self._update_crop_reset_button_styles()
        # Update overlays
        self._update_all_crop_overlays()
        self._schedule_content_update()

    def _update_all_crop_overlays(self):
        """Update crop overlay in both Detection and Development views."""
        self._update_crop_overlay()
        if hasattr(self, 'adjustments_view') and self._crop_mode_active:
            self.adjustments_view.set_crop_mode(
                True,
                full_image=self._rotated_full_image,
                inverted_image=self._rotated_full_image_inverted,
                bounds=self._auto_detected_bounds,
                adjustment=self._crop_adjustment,
                flash_edge=self._flash_edge
            )

    def _update_crop_overlay(self):
        """Update the crop overlay visualization."""
        # Pass crop mode data to the main panel
        self.panel_inverted.set_crop_mode(
            self._crop_mode_active,
            full_image=self._rotated_full_image,
            inverted_image=self._rotated_full_image_inverted,
            bounds=self._auto_detected_bounds,
            adjustment=self._crop_adjustment,
            flash_edge=self._flash_edge
        )

    def _update_base_color_display(self):
        """Update the base color swatch and label in sidebar."""
        color = self.panel_base.get_sampled_color_bgr()
        if color is not None:
            # Convert float32 (0-1) to uint8 (0-255) for display
            if color.dtype == np.float32 or np.max(color) <= 1.0:
                b, g, r = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            else:
                b, g, r = int(color[0]), int(color[1]), int(color[2])
            self.base_color_swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 2px solid #666;"
            )
            self.base_color_label.setText(f"R:{r:3d} G:{g:3d} B:{b:3d}")
        else:
            self.base_color_swatch.setStyleSheet("background-color: #808080; border: 2px solid #666;")
            self.base_color_label.setText("R: --  G: --  B: --")

    def _update_inverted(self):
        """Update the inverted negative panel."""
        import time
        # Get the extracted frame (crop adjustment already applied during extraction)
        extracted = self.panel_extracted.get_image()
        if extracted is None:
            self.panel_inverted.set_image(None)
            self._current_inverted = None
            return

        _t0 = time.time()
        base_color = self.panel_base.get_sampled_color_bgr()
        if base_color is None:
            # Just invert without color correction (handle both float32 and uint8)
            if extracted.dtype == np.float32:
                inverted = 1.0 - extracted
            else:
                inverted = 255 - extracted
        else:
            inverted = self._invert_negative(extracted, base_color)
        _t_invert_calc = time.time() - _t0

        self.panel_inverted.set_image(inverted)
        self._current_inverted = inverted

        # Auto-rotate on first load if no saved rotation
        if self._needs_auto_rotate:
            self._needs_auto_rotate = False
            self._try_auto_rotate_first_load()

        # Update thumbnail with inverted image
        self.thumbnail_bar.update_thumbnail(self.file_index, inverted)

        # Update adjustments view (for preset thumbnails to stay in sync)
        _t0 = time.time()
        self._update_adjustments_preview()
        _t_presets = time.time() - _t0

        print(f"      [PERF invert] calc={_t_invert_calc*1000:.0f}ms presets={_t_presets*1000:.0f}ms")

    def _update_adjustments_preview(self):
        """Update the adjustments view with the current inverted image."""
        if hasattr(self, '_current_inverted') and self._current_inverted is not None:
            # Build cache key from all variables that affect the preset thumbnail output
            cache_key = None
            if self._current_hash:
                fine_rot = self._transform_state.fine_rotation
                adj = self._crop_adjustment
                base_pos = self.panel_base.get_position()  # (x, y) or None

                # Format crop adjustment as compact string
                crop_str = f"{adj['left']},{adj['top']},{adj['right']},{adj['bottom']}"
                base_str = f"{base_pos[0]:.4f},{base_pos[1]:.4f}" if base_pos else "auto"

                cache_key = f"{self._current_hash}_r{self.current_rotation}_f{fine_rot:.1f}_c{crop_str}_b{base_str}"
            self.adjustments_view.set_image(self._current_inverted, cache_key)
        else:
            self.adjustments_view.set_image(None)

    def _invert_negative(self, img: np.ndarray, base_color: np.ndarray) -> np.ndarray:
        """
        Invert a color negative using the base (orange mask) color.
        Delegates to processing.invert_negative for the actual implementation.
        """
        from processing import invert_negative
        return invert_negative(img, base_color)

    def _export_frame(self):
        """Export the final adjusted image (with curves applied if any).

        Supports 16-bit TIFF for maximum quality preservation, or 8-bit PNG/JPEG.
        """
        # Get curves-adjusted image from adjustments view, or fall back to inverted
        adjusted = self.adjustments_view.get_adjusted_image()
        if adjusted is None and self._current_inverted is not None:
            adjusted = self._current_inverted
        if adjusted is None:
            return

        default_name = ""
        if self.current_path:
            default_name = Path(self.current_path).stem + "_adjusted.tiff"
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Frame", default_name,
            "16-bit TIFF (*.tiff);;PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if path:
            # Determine output bit depth based on format
            is_16bit = '16-bit TIFF' in selected_filter or path.lower().endswith('.tiff')

            # Convert from float32 (0-1) to output format
            if adjusted.dtype == np.float32:
                if is_16bit:
                    # 16-bit output (0-65535)
                    output = (np.clip(adjusted, 0, 1) * 65535).astype(np.uint16)
                else:
                    # 8-bit output (0-255)
                    output = (np.clip(adjusted, 0, 1) * 255).astype(np.uint8)
            else:
                # Already uint8
                if is_16bit:
                    # Upscale to 16-bit
                    output = (adjusted.astype(np.uint16) * 257)  # 255 -> 65535
                else:
                    output = adjusted

            cv2.imwrite(path, output)

    def closeEvent(self, event):
        """Save current settings and cleanup threads before closing."""
        self._save_current_settings()
        # Cancel any pending thumbnail loading to avoid crash on exit
        self.thumbnail_bar._cancel_pending_loads()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(230, 126, 34))  # Orange accent
    palette.setColor(QPalette.Highlight, QColor(230, 126, 34))  # Orange accent
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    window = NegativeDetectorGUI()
    window.show()

    # Load files from command line
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        window.set_file_list(files)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
