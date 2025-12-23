# CLAUDE.md

## Project Overview

**Super Negative Processing System** - PySide6 GUI for digitizing film negatives. Handles frame detection, color inversion, curves adjustment, and export.

## Tech Stack

- **PySide6** - Qt GUI framework
- **OpenCV** - Image processing
- **NumPy** - Array operations
- **scipy** - PCHIP interpolation for curves
- **rawpy** - RAW file demosaicing

## Running

```bash
pip install -r requirements.txt
python main.py
```

## Architecture

| File | Purpose |
|------|---------|
| `main.py` | Main window, app entry point |
| `state.py` | Signal-based state management |
| `storage.py` | SQLite persistence |
| `processing.py` | Image loading, frame detection, inversion |
| `auto_rotate.py` | Content-aware rotation detection |
| `presets.py` | Adjustment preset definitions |
| `ui_constants.py` | Colors, dimensions, styles |
| `widgets/` | UI components (see `widgets/__init__.py`) |

## Performance Patterns

- **RAW caching**: Demosaiced RAW files cached to disk via `storage.py` (keyed by file hash)
- **Background loading**: Use `QThread` + `QObject` workers for heavy I/O (see `thumbnail_bar.py`)
- **Cache invalidation**: Composite cache keys include all transform params that affect output

## Key Conventions

- **Images**: float32 NumPy arrays, BGR format, normalized 0-1
- **Colors/Styles**: Always use `ui_constants.py`, never hardcode
- **New panels**: Extend `BaseCollapsiblePanel` in `widgets/panels.py`
- **Panel sizing**: Use `setFixedWidth()` for fixed panels; use `QSizePolicy.Expanding` (no fixed width) for panels that should fill available space
- **Shared state**: Use `TransformState` signals in `state.py`
- **Persistence**: Use `storage.get_storage()` for settings

## Widget Reference

| File | Components |
|------|------------|
| `widgets/controls.py` | SliderWithButtons, VerticalToggleButton, HorizontalToggleButton |
| `widgets/panels.py` | BaseCollapsiblePanel, TabbedRightPanel, Transform controls |
| `widgets/image_panel.py` | ImagePanel, CropWidget, selection widgets |
| `widgets/adjustments.py` | CurvesWidget, AdjustmentsPreview, AdjustmentsView |
| `widgets/thumbnail_bar.py` | ThumbnailBar, ThumbnailItem |
| `widgets/dialogs.py` | KeybindingsDialog, SettingsDialog |
