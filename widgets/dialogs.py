"""
SUPER NEGATIVE PROCESSING SYSTEM - Dialog Widgets

Modal dialogs for settings and keyboard shortcuts help.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDialogButtonBox, QCheckBox, QPushButton, QWidget
)
from PySide6.QtCore import Qt

from state import TransformState
import storage


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
