"""
SUPER NEGATIVE PROCESSING SYSTEM - Widgets Package

Re-exports all widget classes for convenient imports.
"""

# Controls
from widgets.controls import (
    SliderWithButtons,
    VerticalToggleButton,
    SplitVerticalToggleButton,
    HorizontalToggleButton,
)

# Dialogs
from widgets.dialogs import (
    KeybindingsDialog,
    SettingsDialog,
    SavePresetDialog,
    BatchProgressDialog,
)

# Thumbnail bar
from widgets.thumbnail_bar import (
    ThumbnailLoaderWorker,
    ThumbnailItem,
    ThumbnailBar,
)

# Preset bar
from widgets.preset_bar import (
    PresetThumbnailItem,
    PresetBarContainer,
    PresetBar,
)

# Image panels
from widgets.image_panel import (
    CropWidget,
    BaseSelectionWidget,
    ImagePanel,
)

# Adjustments
from widgets.adjustments import (
    CurvesWidget,
    AdjustmentsPreview,
    AdjustmentsView,
)

# Collapsible panels
from widgets.panels import (
    BaseCollapsiblePanel,
    TransformControlsWidget,
    CollapsibleTransformPanel,
    CollapsiblePresetPanel,
    CollapsibleAdjustmentsPanel,
    CollapsibleDebugPanel,
    CollapsibleControlsPanel,
)

__all__ = [
    # Controls
    'SliderWithButtons',
    'VerticalToggleButton',
    'SplitVerticalToggleButton',
    'HorizontalToggleButton',
    # Dialogs
    'KeybindingsDialog',
    'SettingsDialog',
    'SavePresetDialog',
    'BatchProgressDialog',
    # Thumbnail bar
    'ThumbnailLoaderWorker',
    'ThumbnailItem',
    'ThumbnailBar',
    # Preset bar
    'PresetThumbnailItem',
    'PresetBarContainer',
    'PresetBar',
    # Image panels
    'CropWidget',
    'BaseSelectionWidget',
    'ImagePanel',
    # Adjustments
    'CurvesWidget',
    'AdjustmentsPreview',
    'AdjustmentsView',
    # Collapsible panels
    'BaseCollapsiblePanel',
    'TransformControlsWidget',
    'CollapsibleTransformPanel',
    'CollapsiblePresetPanel',
    'CollapsibleAdjustmentsPanel',
    'CollapsibleDebugPanel',
    'CollapsibleControlsPanel',
]
