"""
SUPER NEGATIVE PROCESSING SYSTEM - Collapsible Panels

Collapsible panel containers for transform controls, presets, adjustments, debug, etc.

Architecture:
- BaseCollapsiblePanel: Abstract base providing animation and toggle logic
- Concrete panels override: _get_initial_state(), _save_state(), _setup_content(), tooltips
"""

from abc import abstractmethod
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QCheckBox, QComboBox, QSlider, QSizePolicy, QStackedWidget, QTabBar
)
from PySide6.QtCore import Qt, Signal

import storage
from state import TransformState
from ui_constants import Colors, Dimensions, Styles, get_accent_button_style
from widgets.controls import VerticalToggleButton, SplitVerticalToggleButton, HorizontalToggleButton
from widgets.preset_bar import PresetBar
from widgets.image_panel import ImagePanel, BaseSelectionWidget


class BaseCollapsiblePanel(QWidget):
    """Abstract base class for collapsible panels with slide animation.

    Provides common animation, toggle, and state management logic.
    Subclasses must implement:
    - _get_initial_expanded_state() -> bool
    - _save_expanded_state(expanded: bool)
    - _setup_content() -> QWidget
    - _create_toggle_button() -> QWidget
    - _get_expanded_tooltip() -> str
    - _get_collapsed_tooltip() -> str
    """

    visibilityChanged = Signal(bool)

    # Subclasses should override these
    EXPANDED_WIDTH = 280
    COLLAPSED_WIDTH = 28

    def __init__(self, toggle_on_left: bool = False):
        """
        Args:
            toggle_on_left: If True, toggle button is on left side of content.
                           If False, toggle button is on right side.
        """
        super().__init__()
        self._toggle_on_left = toggle_on_left
        self._expanded = self._get_initial_expanded_state()
        self._content = None
        self._toggle_btn = None

        self._setup_ui()
        self._apply_state_immediate()

    @abstractmethod
    def _get_initial_expanded_state(self) -> bool:
        """Load and return initial expanded state from storage."""
        pass

    @abstractmethod
    def _save_expanded_state(self, expanded: bool):
        """Save expanded state to storage."""
        pass

    @abstractmethod
    def _setup_content(self) -> QWidget:
        """Create and return the content widget."""
        pass

    @abstractmethod
    def _create_toggle_button(self) -> QWidget:
        """Create and return the toggle button widget."""
        pass

    @abstractmethod
    def _get_expanded_tooltip(self) -> str:
        """Return tooltip text when panel is expanded."""
        pass

    @abstractmethod
    def _get_collapsed_tooltip(self) -> str:
        """Return tooltip text when panel is collapsed."""
        pass

    def _setup_ui(self):
        """Set up the panel layout with toggle button and content."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._content = self._setup_content()
        self._toggle_btn = self._create_toggle_button()
        self._toggle_btn.clicked.connect(self.toggle)

        if self._toggle_on_left:
            layout.addWidget(self._toggle_btn)
            layout.addWidget(self._content)
        else:
            layout.addWidget(self._content)
            layout.addWidget(self._toggle_btn)

        # Set explicit size policy for predictable layout behavior
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

    def toggle(self):
        """Toggle the panel open/closed."""
        if self._expanded:
            self._collapse()
        else:
            self._expand()

    def _expand(self):
        """Expand the panel."""
        self._expanded = True
        self._save_expanded_state(True)
        self._content.setFixedWidth(self.EXPANDED_WIDTH)
        self._content.show()
        self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
        self._toggle_btn.set_collapsed(False)
        self._toggle_btn.setToolTip(self._get_expanded_tooltip())
        # Force parent to update layout
        if self.parentWidget():
            self.parentWidget().updateGeometry()
            self.parentWidget().update()
        self.visibilityChanged.emit(True)

    def _collapse(self):
        """Collapse the panel."""
        self._expanded = False
        self._save_expanded_state(False)
        self._content.setFixedWidth(0)
        self._content.hide()
        self.setFixedWidth(self.COLLAPSED_WIDTH)
        self._toggle_btn.set_collapsed(True)
        self._toggle_btn.setToolTip(self._get_collapsed_tooltip())
        # Force parent to update layout
        if self.parentWidget():
            self.parentWidget().updateGeometry()
            self.parentWidget().update()
        self.visibilityChanged.emit(False)

    def _apply_state_immediate(self):
        """Apply current state."""
        if self._expanded:
            self._content.setFixedWidth(self.EXPANDED_WIDTH)
            self._content.show()
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._toggle_btn.set_collapsed(False)
            self._toggle_btn.setToolTip(self._get_expanded_tooltip())
        else:
            self._content.setFixedWidth(0)
            self._content.hide()
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._toggle_btn.set_collapsed(True)
            self._toggle_btn.setToolTip(self._get_collapsed_tooltip())

    def is_expanded(self) -> bool:
        """Return whether the panel is currently expanded."""
        return self._expanded

    def set_expanded(self, expanded: bool):
        """Set the panel state without animation."""
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._apply_state_immediate()
        self.visibilityChanged.emit(expanded)


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
            self.reset_fine_rotation_btn.setStyleSheet(get_accent_button_style())
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
            self.reset_crop_btn.setStyleSheet(get_accent_button_style())
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
            self.reset_fine_rotation_btn.setStyleSheet(get_accent_button_style())
            self.reset_fine_rotation_btn.setToolTip(f"Reset fine rotation to 0° (currently {value:.1f}°)")
        else:
            self.reset_fine_rotation_btn.setStyleSheet("")
            self.reset_fine_rotation_btn.setToolTip("Fine rotation at default: 0°")

    def set_crop_adjustment(self, adjustment: dict):
        """Update crop reset button style based on adjustment values."""
        has_adjustment = any(v != 0 for v in adjustment.values())
        if has_adjustment:
            self.reset_crop_btn.setStyleSheet(get_accent_button_style())
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
    savePresetRequested = Signal()  # Pass-through from PresetBar save button
    updatePresetRequested = Signal(str)  # Pass-through from PresetBar update button
    deletePresetRequested = Signal(str)  # Pass-through from PresetBar delete button
    applyToAllRequested = Signal(str)  # Pass-through from PresetBar apply-to-all button
    visibilityChanged = Signal(bool)  # Emitted when panel is shown/hidden
    fullModeChanged = Signal(bool)  # Emitted when entering/exiting full mode

    # States
    STATE_COLLAPSED = 'collapsed'
    STATE_EXPANDED = 'expanded'
    STATE_FULL = 'full'

    EXPANDED_WIDTH = 280  # Normal expanded width
    COLLAPSED_WIDTH = 28  # Width of toggle button strip

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
        self._apply_state_immediate()

    def _setup_ui(self):
        # Main horizontal layout: preset bar + toggle button
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # The preset bar (main content)
        self._preset_bar = PresetBar()
        self._preset_bar.presetSelected.connect(self.presetSelected.emit)
        self._preset_bar.savePresetRequested.connect(self.savePresetRequested.emit)
        self._preset_bar.updatePresetRequested.connect(self.updatePresetRequested.emit)
        self._preset_bar.deletePresetRequested.connect(self.deletePresetRequested.emit)
        self._preset_bar.applyToAllRequested.connect(self.applyToAllRequested.emit)
        layout.addWidget(self._preset_bar)

        # Split vertical toggle button on the right edge
        self._toggle_btn = SplitVerticalToggleButton("PRESETS")
        self._toggle_btn.topClicked.connect(self._on_top_clicked)
        self._toggle_btn.bottomClicked.connect(self._on_bottom_clicked)
        layout.addWidget(self._toggle_btn)

        # Set initial size
        self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)

    def set_full_width(self, width: int):
        """Set the width to use when in full expanded mode."""
        self._full_width = width
        # In full mode, both panel and preset_bar use Expanding policy
        # so we don't need to set fixed widths - the layout handles it

    def toggle(self):
        """Toggle between collapsed and normal expanded (for keyboard shortcut)."""
        if self._state == self.STATE_COLLAPSED:
            self._expand_normal()
        else:
            self._collapse()

    def toggle_grid_mode(self):
        """Toggle to/from grid (full) mode (for keyboard shortcut)."""
        if self._state == self.STATE_FULL:
            self._collapse()
        else:
            self._expand_full()

    def _on_top_clicked(self):
        """Handle top zone click - toggle collapsed/normal expanded."""
        if self._state == self.STATE_COLLAPSED:
            self._expand_normal()
        else:
            self._collapse()

    def _on_bottom_clicked(self):
        """Handle bottom zone click - toggle to/from full expanded."""
        if self._state == self.STATE_FULL:
            self._collapse()
        else:
            self._expand_full()

    def _expand_normal(self):
        """Expand to normal width with list view."""
        self._state = self.STATE_EXPANDED
        storage.get_storage().set_preset_panel_expanded(True)

        # Switch to list layout with fixed sizes
        self._preset_bar.set_grid_mode(False)
        self._preset_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self._preset_bar.setFixedWidth(self.EXPANDED_WIDTH)
        self._preset_bar.show()

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)

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

        # Switch to grid layout
        self._preset_bar.set_grid_mode(True)
        # Let preset bar expand with panel - set min but not max
        self._preset_bar.setMinimumWidth(100)
        self._preset_bar.setMaximumWidth(16777215)
        self._preset_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # In full mode, allow expansion
        self.setMinimumWidth(self.COLLAPSED_WIDTH + 100)
        self.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._preset_bar.show()

        self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_FULL)
        self.visibilityChanged.emit(True)

    def _collapse(self):
        """Collapse the panel."""
        was_full = self._state == self.STATE_FULL
        self._state = self.STATE_COLLAPSED
        storage.get_storage().set_preset_panel_expanded(False)

        if was_full:
            self.fullModeChanged.emit(False)

        # Switch back to list layout with fixed sizes
        self._preset_bar.set_grid_mode(False)
        self._preset_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self._preset_bar.hide()

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.setFixedWidth(self.COLLAPSED_WIDTH)

        self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_COLLAPSED)
        self.visibilityChanged.emit(False)

    def _apply_state_immediate(self):
        """Apply current state."""
        if self._state == self.STATE_FULL:
            # In full mode, allow expansion for both panel and preset bar
            self._preset_bar.setMinimumWidth(100)
            self._preset_bar.setMaximumWidth(16777215)
            self._preset_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self._preset_bar.set_grid_mode(True)
            self._preset_bar.show()
            self.setMinimumWidth(self.COLLAPSED_WIDTH + 100)
            self.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_FULL)
        elif self._state == self.STATE_EXPANDED:
            self._preset_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            self._preset_bar.setFixedWidth(self.EXPANDED_WIDTH)
            self._preset_bar.set_grid_mode(False)
            self._preset_bar.show()
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            self.setFixedWidth(self.EXPANDED_WIDTH + self.COLLAPSED_WIDTH)
            self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_EXPANDED)
        else:
            self._preset_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            self._preset_bar.hide()
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            self._toggle_btn.set_state(SplitVerticalToggleButton.STATE_COLLAPSED)

    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        # In full mode, preset_bar uses Expanding policy so no manual width update needed

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

    def add_user_preset(self, key: str, name: str, description: str = ""):
        """Add a new user preset to the bar."""
        self._preset_bar.add_user_preset(key, name, description)

    def remove_preset(self, key: str):
        """Remove a preset from the bar."""
        self._preset_bar.remove_preset(key)

    @property
    def preset_bar(self) -> PresetBar:
        """Access the underlying PresetBar widget."""
        return self._preset_bar


class CollapsibleAdjustmentsPanel(BaseCollapsiblePanel):
    """A collapsible container for the adjustments controls with slide animation."""

    EXPANDED_WIDTH = 320

    def __init__(self, content_widget: QWidget):
        """
        Args:
            content_widget: The widget to show/hide (scroll area with controls)
        """
        self._content_widget = content_widget
        super().__init__(toggle_on_left=True)  # Toggle button on left

    def _get_initial_expanded_state(self) -> bool:
        store = storage.get_storage()
        behavior = store.get_adjustments_panel_startup_behavior()
        if behavior == 'expanded':
            return True
        elif behavior == 'collapsed':
            return False
        else:  # 'last'
            return store.get_adjustments_panel_expanded()

    def _save_expanded_state(self, expanded: bool):
        storage.get_storage().set_adjustments_panel_expanded(expanded)

    def _setup_content(self) -> QWidget:
        return self._content_widget

    def _create_toggle_button(self) -> QWidget:
        return VerticalToggleButton("ADJUSTMENTS", side="left")

    def _get_expanded_tooltip(self) -> str:
        return "Hide adjustments panel (A)"

    def _get_collapsed_tooltip(self) -> str:
        return "Show adjustments panel (A)"


class CollapsibleDebugPanel(BaseCollapsiblePanel):
    """A collapsible container for detection debug panels (left sidebar)."""

    EXPANDED_WIDTH = 280

    def __init__(self):
        # These will be set during _setup_content, before base __init__ completes
        self.panel_negative = None
        self.panel_base = None
        self.panel_edges = None
        self.panel_extracted = None
        super().__init__(toggle_on_left=False)  # Toggle button on right

    def _get_initial_expanded_state(self) -> bool:
        store = storage.get_storage()
        behavior = store.get_debug_panel_startup_behavior()
        if behavior == 'expanded':
            return True
        elif behavior == 'collapsed':
            return False
        else:  # 'last'
            return store.get_debug_panel_expanded()

    def _save_expanded_state(self, expanded: bool):
        storage.get_storage().set_debug_panel_expanded(expanded)

    def _setup_content(self) -> QWidget:
        content = QWidget()
        content_layout = QVBoxLayout(content)
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

        return content

    def _wrap_panel(self, title: str, panel: QWidget) -> QWidget:
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(3, 3, 3, 3)
        group_layout.addWidget(panel)
        return group

    def _create_toggle_button(self) -> QWidget:
        return VerticalToggleButton("DEBUG")

    def _get_expanded_tooltip(self) -> str:
        return "Hide debug panels (` or §)"

    def _get_collapsed_tooltip(self) -> str:
        return "Show debug panels (` or §)"


class CollapsibleControlsPanel(BaseCollapsiblePanel):
    """A collapsible container for controls sidebar (right side).

    Content is set after construction via set_content().
    """

    EXPANDED_WIDTH = 320

    def __init__(self):
        self._content_widget = None  # Will be set via set_content()
        super().__init__(toggle_on_left=True)  # Toggle button on left

    def _get_initial_expanded_state(self) -> bool:
        store = storage.get_storage()
        behavior = store.get_controls_panel_startup_behavior()
        if behavior == 'expanded':
            return True
        elif behavior == 'collapsed':
            return False
        else:  # 'last'
            return store.get_controls_panel_expanded()

    def _save_expanded_state(self, expanded: bool):
        storage.get_storage().set_controls_panel_expanded(expanded)

    def _setup_content(self) -> QWidget:
        # Create container - actual content added via set_content()
        container = QWidget()
        self._content_layout = QVBoxLayout(container)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        return container

    def set_content(self, widget: QWidget):
        """Set the content widget for this panel."""
        self._content_widget = widget
        self._content_layout.addWidget(widget)

    def _create_toggle_button(self) -> QWidget:
        return VerticalToggleButton("CONTROLS", side="left")

    def _get_expanded_tooltip(self) -> str:
        return "Hide controls panel (~ or ±)"

    def _get_collapsed_tooltip(self) -> str:
        return "Show controls panel (~ or ±)"
