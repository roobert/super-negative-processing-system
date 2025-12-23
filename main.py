#!/usr/bin/env python3
"""
SUPER NEGATIVE PROCESSING SYSTEM - GUI Application

PySide6 application for interactive frame detection with live preview.
"""

import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

import sys
import argparse
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
from state import TransformState
from ui_constants import Colors, Styles, get_accent_button_style

# Import all widgets from the refactored widgets package
from widgets import (
    # Controls
    SliderWithButtons,
    VerticalToggleButton,
    SplitVerticalToggleButton,
    HorizontalToggleButton,
    # Dialogs
    KeybindingsDialog,
    SettingsDialog,
    # Thumbnail bar
    ThumbnailLoaderWorker,
    ThumbnailItem,
    ThumbnailBar,
    # Preset bar
    PresetThumbnailItem,
    PresetBarContainer,
    PresetBar,
    # Image panels
    CropWidget,
    BaseSelectionWidget,
    ImagePanel,
    # Adjustments
    CurvesWidget,
    AdjustmentsPreview,
    AdjustmentsView,
    # Collapsible panels
    TransformControlsWidget,
    CollapsibleTransformPanel,
    CollapsiblePresetPanel,
    CollapsibleAdjustmentsPanel,
    CollapsibleDebugPanel,
    CollapsibleControlsPanel,
    TabbedRightPanel,
)


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
                (["g"], "Grid view"),
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

        # G to toggle preset grid view (Development view only)
        if event.key() == Qt.Key_G and not event.modifiers():
            if self.view_stack.currentIndex() == 1:  # Development view
                self.adjustments_view.toggle_preset_grid()
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
            style = get_accent_button_style()
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


# Supported image extensions (RAW formats + common image formats)
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def expand_paths(paths: list[str], recursive: bool = False) -> list[str]:
    """Expand paths to list of image files.

    - Regular files are included if they have a supported extension
    - Directories are expanded to their image files (recursively if recursive=True)
    """
    result = []
    for path in paths:
        p = Path(path)
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                result.append(str(p))
        elif p.is_dir():
            pattern = '**/*' if recursive else '*'
            for child in sorted(p.glob(pattern)):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    result.append(str(child))
    return result


def main():
    # Parse arguments before QApplication consumes sys.argv
    parser = argparse.ArgumentParser(description='Super Negative Processing System')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Recursively load images from directories')
    parser.add_argument('paths', nargs='*', help='Image files or directories to load')
    args = parser.parse_args()

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
    if args.paths:
        files = expand_paths(args.paths, recursive=args.recursive)
        if files:
            window.set_file_list(files)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
