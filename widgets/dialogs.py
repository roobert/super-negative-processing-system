"""
SUPER NEGATIVE PROCESSING SYSTEM - Dialog Widgets

Modal dialogs for settings and keyboard shortcuts help.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDialogButtonBox, QCheckBox, QPushButton, QWidget,
    QLineEdit, QTextEdit, QProgressBar
)
from PySide6.QtCore import Qt, Signal
from pathlib import Path
import time

from state import TransformState
import storage


class SavePresetDialog(QDialog):
    """Dialog for saving a new user preset."""

    def __init__(self, parent=None, default_name: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Save Preset")
        self.setMinimumWidth(350)
        self._preset_name = ""
        self._preset_description = ""
        self._default_name = default_name
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Name input
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setFixedWidth(80)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Enter preset name...")
        self._name_edit.setText(self._default_name)
        self._name_edit.textChanged.connect(self._validate)
        name_layout.addWidget(name_label)
        name_layout.addWidget(self._name_edit)
        layout.addLayout(name_layout)

        # Description input (optional)
        desc_layout = QHBoxLayout()
        desc_label = QLabel("Description:")
        desc_label.setFixedWidth(80)
        desc_label.setAlignment(Qt.AlignTop)
        self._desc_edit = QTextEdit()
        self._desc_edit.setPlaceholderText("Optional description...")
        self._desc_edit.setMaximumHeight(60)
        desc_layout.addWidget(desc_label)
        desc_layout.addWidget(self._desc_edit)
        layout.addLayout(desc_layout)

        # Info text
        info_label = QLabel("The current adjustments and curves will be saved as a new preset.")
        info_label.setStyleSheet("color: #888; font-size: 11px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Spacer
        layout.addStretch()

        # Dialog buttons
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self._save_button = self._button_box.button(QDialogButtonBox.Save)
        self._button_box.accepted.connect(self._save_and_accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

        # Initial validation
        self._validate()

    def _validate(self):
        """Validate input and enable/disable save button."""
        name = self._name_edit.text().strip()
        self._save_button.setEnabled(len(name) > 0)

    def _save_and_accept(self):
        """Save values and accept."""
        self._preset_name = self._name_edit.text().strip()
        self._preset_description = self._desc_edit.toPlainText().strip()
        self.accept()

    def get_preset_name(self) -> str:
        """Get the entered preset name."""
        return self._preset_name

    def get_preset_description(self) -> str:
        """Get the entered preset description."""
        return self._preset_description


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


class BatchProgressDialog(QDialog):
    """Modal progress dialog for batch operations with cancel support."""

    cancelled = Signal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self.setModal(True)
        # Disable close button - must use Cancel
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        self._start_time = None
        self._processed_count = 0
        self._is_finished = False

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Status label
        self._status_label = QLabel("Initializing...")
        self._status_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._status_label)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        # Details row
        details_layout = QHBoxLayout()

        self._item_label = QLabel("")
        self._item_label.setStyleSheet("color: #888; font-size: 11px;")
        details_layout.addWidget(self._item_label, 1)

        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet("color: #888; font-size: 11px;")
        details_layout.addWidget(self._eta_label)

        layout.addLayout(details_layout)

        # Worker info
        self._worker_label = QLabel("")
        self._worker_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._worker_label)

        # Spacer
        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)

        layout.addLayout(btn_layout)

    def set_worker_info(self, worker_count: int, memory_gb: float):
        """Display worker configuration info."""
        self._worker_label.setText(
            f"Using {worker_count} CPU workers  |  {memory_gb:.1f} GB available"
        )

    def set_progress(self, current: int, total: int, message: str = ""):
        """Update progress display."""
        if self._start_time is None:
            self._start_time = time.time()

        self._processed_count = current

        # Update progress bar
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._status_label.setText(f"Processing {current} of {total}...")

        # Show current item (just filename)
        if message:
            filename = Path(message).name if "/" in message or "\\" in message else message
            self._item_label.setText(filename)

        # Calculate ETA
        if current > 0 and not self._is_finished:
            elapsed = time.time() - self._start_time
            rate = current / elapsed
            if rate > 0:
                remaining = (total - current) / rate
                if remaining < 60:
                    self._eta_label.setText(f"~{int(remaining)}s remaining")
                else:
                    minutes = int(remaining / 60)
                    self._eta_label.setText(f"~{minutes}m remaining")
            else:
                self._eta_label.setText("")

    def set_finished(self, success_count: int, total: int):
        """Show completion status and change button to Close."""
        self._is_finished = True

        if success_count == total:
            self._status_label.setText(f"Complete: {success_count} images processed")
        else:
            failed = total - success_count
            self._status_label.setText(
                f"Complete: {success_count} processed, {failed} failed"
            )

        self._progress_bar.setValue(total)
        self._item_label.setText("")
        self._eta_label.setText("")

        # Calculate total time
        if self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed < 60:
                self._worker_label.setText(f"Completed in {elapsed:.1f} seconds")
            else:
                minutes = int(elapsed / 60)
                seconds = int(elapsed % 60)
                self._worker_label.setText(f"Completed in {minutes}m {seconds}s")

        # Change button to Close
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.accept)

    def _on_cancel(self):
        """Handle cancel button click."""
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Cancelling...")
        self.cancelled.emit()

    def closeEvent(self, event):
        """Handle window close - emit cancel if not finished."""
        if not self._is_finished:
            self.cancelled.emit()
        super().closeEvent(event)
