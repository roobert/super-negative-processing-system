"""
SUPER NEGATIVE PROCESSING SYSTEM - Control Widgets

Reusable control widgets: sliders, toggle buttons, etc.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from ui_constants import Colors, Dimensions, get_accent_button_style, get_danger_button_style


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
                self.reset_btn.setStyleSheet(get_accent_button_style())
                self.reset_btn.setToolTip(f"Reset → {self.default:.{self.decimals}f}")
        elif at_absolute and at_preset:
            # At both defaults (they're equal) - no reset needed
            self.reset_btn.setStyleSheet("")
            self.reset_btn.setToolTip(f"At default: {self.default:.{self.decimals}f}")
        elif at_preset and not at_absolute:
            # Orange: at preset default, can go to absolute
            self.reset_btn.setStyleSheet(get_accent_button_style())
            self.reset_btn.setToolTip(f"Reset to absolute → {self.default:.{self.decimals}f}")
        elif at_absolute and not at_preset:
            # Blue: at absolute default, can go to preset
            self.reset_btn.setStyleSheet(get_accent_button_style())
            self.reset_btn.setToolTip(f"Reset to preset → {self._preset_default:.{self.decimals}f}")
        else:
            # Red: tweaked away from preset, can go to preset
            self.reset_btn.setStyleSheet(get_danger_button_style())
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
            painter.fillRect(self.rect(), QColor(Colors.BACKGROUND_MEDIUM))
        else:
            painter.fillRect(self.rect(), QColor(Colors.BACKGROUND_DARK))

        # Border (left border for right-side button, right border for left-side button)
        painter.setPen(QPen(QColor(Colors.BACKGROUND_MEDIUM), 1))
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
            painter.setPen(QColor(Colors.TEXT_PRIMARY))
        else:
            painter.setPen(QColor(Colors.TEXT_MUTED))

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
            painter.fillRect(top_rect, QColor(Colors.BACKGROUND_MEDIUM))
        else:
            painter.fillRect(top_rect, QColor(Colors.BACKGROUND_DARK))

        # Draw bottom zone background
        bottom_rect = QRect(0, mid_y, self.width(), self.height() - mid_y)
        if self._hovered_zone == 'bottom':
            painter.fillRect(bottom_rect, QColor(Colors.BACKGROUND_MEDIUM))
        else:
            painter.fillRect(bottom_rect, QColor(Colors.BACKGROUND_DARK))

        # Draw divider line between zones
        painter.setPen(QPen(QColor(Colors.BORDER_DARK), 1))
        painter.drawLine(4, mid_y, self.width() - 4, mid_y)

        # Border on left side (panel is on right)
        painter.setPen(QPen(QColor(Colors.BACKGROUND_MEDIUM), 1))
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
            painter.setPen(QColor(Colors.TEXT_PRIMARY))
        else:
            painter.setPen(QColor(Colors.TEXT_MUTED))

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
            painter.setPen(QColor(Colors.TEXT_PRIMARY))
        elif self._state == self.STATE_FULL:
            painter.setPen(QColor(Colors.ACCENT_PRIMARY))  # Orange when full
        else:
            painter.setPen(QColor(Colors.TEXT_MUTED))

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
            painter.fillRect(self.rect(), QColor(Colors.BACKGROUND_MEDIUM))
        else:
            painter.fillRect(self.rect(), QColor(Colors.BACKGROUND_DARK))

        # Top border
        painter.setPen(QPen(QColor(Colors.BACKGROUND_MEDIUM), 1))
        painter.drawLine(0, 0, self.width(), 0)

        # Text styling
        font = painter.font()
        font.setPixelSize(11)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 120)
        painter.setFont(font)

        if self._hovered:
            painter.setPen(QColor(Colors.TEXT_PRIMARY))
        else:
            painter.setPen(QColor(Colors.TEXT_MUTED))

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
