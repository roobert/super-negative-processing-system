"""
SUPER NEGATIVE PROCESSING SYSTEM - Shared State

Centralized state management for transform controls shared between views.
"""

from PySide6.QtCore import QObject, Signal
import storage


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
