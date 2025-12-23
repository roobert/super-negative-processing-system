"""
SUPER NEGATIVE PROCESSING SYSTEM - Collapsible Panels

Collapsible panel containers for transform controls, presets, adjustments, debug, etc.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QCheckBox, QComboBox, QSlider, QSizePolicy, QStackedWidget, QTabBar
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve

import storage
from state import TransformState
from widgets.controls import VerticalToggleButton, SplitVerticalToggleButton, HorizontalToggleButton
from widgets.preset_bar import PresetBar
from widgets.image_panel import ImagePanel, BaseSelectionWidget


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
