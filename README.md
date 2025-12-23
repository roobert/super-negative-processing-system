# SUPER NEGATIVE PROCESSING SYSTEM

A professional-grade GUI application for digitizing scanned film negatives. Automatically detects frame boundaries, inverts negatives to positives with color correction, and provides comprehensive development controls with classic film stock presets.

## Key Features

### Frame Detection & Extraction
- **Automatic frame boundary detection** from scanned film strips using edge detection and Hough line transforms
- **Intelligent frame isolation** separating film frames from scanner backgrounds
- **Precision corner detection** using RANSAC line fitting and Shi-Tomasi corner refinement with sub-pixel accuracy
- **Automatic deskewing** to straighten rotated frames
- **35mm aspect ratio optimization** (1.5:1) with rectangle fitting constraints

### Automatic Orientation Detection
- **Multi-method voting system** with 8+ independent detection algorithms
- **Face detection** using Haar cascades
- **Person and upper body detection** via HOG (Histogram of Oriented Gradients)
- **Eye detection** with pair validation
- **Sky color analysis** (blue regions indicate top of image)
- **Gradient and brightness distribution** analysis
- **Horizon line detection**
- **Optional OCR text detection** via pytesseract

### Color Processing & Inversion
- **Negative-to-positive conversion** with automatic color correction
- **Base color sampling** from film stock border for accurate inversion
- **White balance controls** with RGB multipliers (0.5-2.0 range)
- **Auto-levels** on inversion using 1st-99th percentile normalization
- **16-bit processing pipeline** with float32 normalization for maximum precision

### Development Controls
- **Exposure, contrast, highlights, shadows** independent adjustment
- **Temperature and tint** color correction
- **Vibrance and saturation** controls
- **Blacks, whites, and gamma** fine-tuning
- **Sharpening** with real-time preview
- **Per-channel tone curves** with PCHIP interpolation for smooth splines

### Film Stock Presets
- **16 built-in film presets** emulating classic stocks:
  - Kodak Portra 400, Portra 160, Portra 800
  - Kodak Ektar 100
  - Kodak Tri-X 400, T-Max 100
  - Fuji Pro 400H, Superia 400
  - Fuji Velvia 50, Provia 100F
  - Fuji Acros 100
  - Ilford HP5 Plus, Delta 3200
  - CineStill 800T
  - Lomography 400
  - And more...
- **Preset thumbnails** with visual preview generation
- **Draggable preset reordering** with favorites system
- **Keyboard shortcuts** (1-9) for quick preset application

### RAW File Support
- **Comprehensive RAW format support**: NEF, CR2, CR3, ARW, DNG, ORF, RW2, RAF, PEF, SRW
- **Standard format support**: TIFF, PNG, JPEG
- **On-disk RAW caching** to 16-bit PNG for fast repeated access
- **Full bit-depth preservation** during processing

### Interactive GUI
- **Dual-tab interface** with Detection and Development workflows
- **Thumbnail sidebar** with lazy-loading and background processing
- **Interactive crop overlay** with aspect ratio locking
- **Zoom and pan** with scroll wheel support
- **Fullscreen mode** with automatic panel hiding
- **Collapsible panels** with startup state memory
- **Comprehensive keyboard navigation**

### Workflow & Storage
- **Per-image settings persistence** remembering adjustments on reload
- **SQLite database** for settings and thumbnail caching
- **LRU cache eviction** (50MB limit) for preset thumbnails
- **Image identification** via SHA-256 hashing
- **Batch processing** support via CLI
- **Favorites system** for images and presets

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
PySide6>=6.5.0
rawpy>=0.19.0
```

**Optional:**
- `pytesseract` - For OCR-based text rotation detection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py [image_path_or_directory]
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` | Switch between Detection/Development tabs |
| `1-9` | Apply preset by number |
| `Arrow keys` | Navigate between images |
| `Scroll wheel` | Zoom in/out |
| `F` | Toggle fullscreen |
| `R` | Reset adjustments |

## Architecture

```
main.py             - PySide6 GUI application (main entry point)
processing.py       - Frame detection and image processing core
auto_rotate.py      - Multi-method automatic rotation detection
presets.py          - Film stock preset definitions
storage.py          - SQLite persistence and caching
```

## Storage Locations

- **Settings database**: `~/.config/super-negative-processing-system/cache.db`
- **RAW cache**: `~/.config/super-negative-processing-system/raw_cache/`

## License

See LICENSE file for details.
