"""
SUPER NEGATIVE PROCESSING SYSTEM - UI Constants

Centralized theme colors, dimensions, and styling constants.
Import from here instead of hardcoding values throughout the codebase.

Usage:
    from ui_constants import Colors, Dimensions, Styles

    button.setStyleSheet(f"background-color: {Colors.ACCENT_PRIMARY};")
    button.setFixedSize(*Dimensions.BUTTON_SMALL)
"""


class Colors:
    """Centralized color definitions for the application theme."""

    # === Background Colors (dark theme gradient) ===
    BACKGROUND_DARKEST = "#1a1a1a"  # Deepest background (overlays, some containers)
    BACKGROUND_DARK = "#2a2a2a"     # Standard widget background
    BACKGROUND_MEDIUM = "#3a3a3a"   # Elevated surfaces, selected tabs
    BACKGROUND_HOVER = "#333333"    # Hover state for dark backgrounds

    # === Border Colors ===
    BORDER_DARK = "#444444"         # Standard borders
    BORDER_MEDIUM = "#555555"       # Lighter borders
    BORDER_LIGHT = "#666666"        # Lightest borders

    # === Text Colors ===
    TEXT_PRIMARY = "#ffffff"        # Primary text (white)
    TEXT_SECONDARY = "#cccccc"      # Secondary text (light gray)
    TEXT_MUTED = "#888888"          # Muted/disabled text

    # === Accent Colors ===
    ACCENT_PRIMARY = "#e67e22"      # Primary accent (orange) - selection, highlights
    ACCENT_DANGER = "#e74c3c"       # Danger/destructive actions (red)
    ACCENT_SUCCESS = "#66aa66"      # Success/valid states (green)
    ACCENT_FAVORITE = "#f1c40f"     # Favorite star (yellow)
    ACCENT_FAVORITE_HOVER = "#f39c12"  # Favorite star hover

    # === Curve Channel Colors ===
    CHANNEL_RGB = "#888888"         # Combined RGB curve
    CHANNEL_RED = "#ff6666"         # Red channel curve
    CHANNEL_GREEN = "#66ff66"       # Green channel curve
    CHANNEL_BLUE = "#6666ff"        # Blue channel curve

    # === White Balance Gradient Colors ===
    WB_RED_MIN = "#00cccc"          # Cyan (opposite of red)
    WB_RED_MAX = "#ff4444"          # Red
    WB_GREEN_MIN = "#cc44cc"        # Magenta (opposite of green)
    WB_GREEN_MAX = "#44cc44"        # Green
    WB_BLUE_MIN = "#cccc44"         # Yellow (opposite of blue)
    WB_BLUE_MAX = "#4444ff"         # Blue

    # === Special Colors ===
    SWATCH_DEFAULT = "#808080"      # Default color swatch gray
    BORDER_VALID = "#66aa66"        # Valid/confirmed border (green)


class Dimensions:
    """Centralized dimension constants for consistent sizing."""

    # === Button Sizes (width, height) ===
    BUTTON_SMALL = (24, 24)         # Reset buttons, small icons
    BUTTON_MEDIUM = (28, 28)        # Toggle buttons, toolbar buttons
    BUTTON_ICON = (20, 20)          # Info/help icons

    # === Button Widths (single dimension) ===
    BUTTON_WIDTH_NARROW = 40        # Coarse +/- buttons, "Auto" button
    BUTTON_WIDTH_STANDARD = 50      # Standard +/- buttons, rotation buttons
    BUTTON_WIDTH_WIDE = 60          # "Invert" button
    BUTTON_WIDTH_EXTRA = 120        # "Enter Crop Mode" button

    # === Panel Dimensions ===
    PANEL_COLLAPSED_WIDTH = 28      # Collapsed panel (toggle button only)
    PANEL_WIDTH_NARROW = 280        # Narrow panels (debug, presets)
    PANEL_WIDTH_WIDE = 320          # Wide panels (controls, adjustments)

    # === Thumbnail Dimensions ===
    THUMBNAIL_SIZE = (104, 78)      # Standard thumbnail
    THUMBNAIL_BAR_WIDTH = 130       # Thumbnail sidebar width
    THUMBNAIL_CONTAINER_WIDTH = 110 # Thumbnail scroll container

    # === Preset Thumbnail Dimensions ===
    PRESET_ITEM_SIZE = (260, 195)   # Preset item total size
    PRESET_THUMB_SIZE = (250, 168)  # Preset thumbnail image
    PRESET_BAR_WIDTH = 280          # Preset bar width

    # === Label Widths ===
    LABEL_WIDTH_NARROW = 24         # Small numeric labels
    LABEL_WIDTH_STANDARD = 40       # Standard value labels
    LABEL_WIDTH_WIDE = 55           # Section labels
    LABEL_WIDTH_DIALOG = 120        # Dialog setting labels

    # === Slider Widths ===
    SLIDER_WIDTH_NARROW = 80        # Grid divisions slider
    SLIDER_WIDTH_STANDARD = 100     # Fine rotation slider

    # === Miscellaneous ===
    STAR_SIZE = (18, 18)            # Favorite star overlay
    SWATCH_SIZE = (40, 40)          # Color swatch display
    COMBO_WIDTH = 120               # ComboBox width


class Animation:
    """Animation timing constants."""

    DURATION_FAST = 200             # Fast animations (transform panel)
    DURATION_STANDARD = 250         # Standard panel animations


class Styles:
    """Pre-built style strings for common patterns."""

    # === Button Styles ===
    BUTTON_ACCENT = f"""
        QPushButton {{
            background-color: {Colors.ACCENT_PRIMARY};
            color: {Colors.TEXT_PRIMARY};
            font-weight: bold;
        }}
    """

    BUTTON_DANGER = f"""
        QPushButton {{
            background-color: {Colors.ACCENT_DANGER};
            color: {Colors.TEXT_PRIMARY};
            font-weight: bold;
        }}
    """

    # === Tab Bar Style ===
    TAB_BAR = f"""
        QTabBar {{
            background: transparent;
        }}
        QTabBar::tab {{
            padding: 6px 16px;
            margin: 0;
            border: none;
            background: {Colors.BACKGROUND_DARK};
            color: {Colors.TEXT_MUTED};
        }}
        QTabBar::tab:selected {{
            background: {Colors.BACKGROUND_MEDIUM};
            color: {Colors.TEXT_PRIMARY};
            border-bottom: 2px solid {Colors.ACCENT_PRIMARY};
        }}
        QTabBar::tab:hover:!selected {{
            background: {Colors.BACKGROUND_HOVER};
        }}
    """

    # === Scroll Area Style ===
    SCROLL_AREA_DARK = f"""
        QScrollArea {{
            border: none;
            background-color: {Colors.BACKGROUND_MEDIUM};
        }}
    """

    SCROLL_AREA_DARKEST = f"""
        QScrollArea {{
            border: none;
            background-color: {Colors.BACKGROUND_DARKEST};
        }}
    """

    # === Thumbnail Styles ===
    THUMBNAIL_DEFAULT = f"""
        QLabel {{
            background-color: {Colors.BACKGROUND_DARK};
            border: 2px solid {Colors.BORDER_DARK};
        }}
    """

    THUMBNAIL_SELECTED = f"""
        QLabel {{
            background-color: {Colors.BACKGROUND_DARK};
            border: 2px solid {Colors.ACCENT_PRIMARY};
        }}
    """

    # === Favorite Star Styles ===
    FAVORITE_STAR = f"""
        QLabel {{
            background-color: rgba(0, 0, 0, 0.6);
            color: {Colors.ACCENT_FAVORITE};
            border: none;
            border-radius: 3px;
            font-size: 12px;
        }}
    """

    FAVORITE_BUTTON_ACTIVE = f"""
        QPushButton {{
            background-color: {Colors.ACCENT_FAVORITE};
            color: {Colors.BACKGROUND_DARKEST};
            border: none;
            border-radius: 4px;
            font-size: 14px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {Colors.ACCENT_FAVORITE_HOVER};
        }}
    """

    FAVORITE_BUTTON_INACTIVE = f"""
        QPushButton {{
            background-color: {Colors.BACKGROUND_HOVER};
            color: {Colors.TEXT_MUTED};
            border: none;
            border-radius: 4px;
            font-size: 14px;
        }}
        QPushButton:hover {{
            background-color: {Colors.BORDER_DARK};
            color: {Colors.ACCENT_FAVORITE};
        }}
    """


def get_accent_button_style() -> str:
    """Return the accent button style string (for dynamic application)."""
    return f"QPushButton {{ background-color: {Colors.ACCENT_PRIMARY}; color: white; font-weight: bold; }}"


def get_danger_button_style() -> str:
    """Return the danger button style string (for dynamic application)."""
    return f"QPushButton {{ background-color: {Colors.ACCENT_DANGER}; color: white; font-weight: bold; }}"
