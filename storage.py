"""
SQLite-based storage for image settings and thumbnails.
"""

import sqlite3
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# Database location
DB_DIR = Path.home() / ".config" / "super-negative-processing-system"
DB_FILE = DB_DIR / "cache.db"
RAW_CACHE_DIR = DB_DIR / "raw_cache"

# Thumbnail settings
THUMB_WIDTH = 100
THUMB_HEIGHT = 74
THUMB_QUALITY = 85  # JPEG quality

# Preset thumbnail cache settings
MAX_PRESET_CACHE_MB = 50  # Evict oldest entries when cache exceeds this size

# RAW cache settings - using PNG for lossless quality


class Storage:
    """SQLite storage for per-image settings and thumbnails."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_FILE
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        """Ensure the database and cache directories exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    hash TEXT PRIMARY KEY,
                    settings TEXT,
                    thumbnail BLOB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # App-wide preferences table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            # Preset thumbnail cache (image_hash + preset_key -> thumbnail)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preset_thumbnails (
                    image_hash TEXT,
                    preset_key TEXT,
                    thumbnail BLOB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (image_hash, preset_key)
                )
            """)
            conn.commit()

    def save_settings(self, image_hash: str, settings: dict):
        """Save settings for an image."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO images (hash, settings, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(hash) DO UPDATE SET
                    settings = excluded.settings,
                    updated_at = CURRENT_TIMESTAMP
            """, (image_hash, json.dumps(settings)))
            conn.commit()

    def load_settings(self, image_hash: str) -> Optional[dict]:
        """Load settings for an image. Returns None if not found."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT settings FROM images WHERE hash = ?",
                (image_hash,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return None

    def save_thumbnail(self, image_hash: str, img: np.ndarray):
        """Save a thumbnail for an image (as JPEG blob)."""
        if img is None:
            return

        # Convert float32 (0-1) to uint8 (0-255) for thumbnail
        if img.dtype == np.float32:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Convert BGR to RGB if needed, then resize
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        h, w = img_rgb.shape[:2]
        scale = min(THUMB_WIDTH / w, THUMB_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_scaled = cv2.resize(img_rgb, (new_w, new_h))

        # Convert back to BGR for JPEG encoding
        img_bgr = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2BGR)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
        blob = buffer.tobytes()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO images (hash, thumbnail, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(hash) DO UPDATE SET
                    thumbnail = excluded.thumbnail,
                    updated_at = CURRENT_TIMESTAMP
            """, (image_hash, blob))
            conn.commit()

    def load_thumbnail(self, image_hash: str) -> Optional[np.ndarray]:
        """Load a thumbnail for an image. Returns BGR numpy array or None."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT thumbnail FROM images WHERE hash = ?",
                (image_hash,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                # Decode JPEG blob
                arr = np.frombuffer(row[0], dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return img
            return None

    def save_preset_thumbnail(self, image_hash: str, preset_key: str, img: np.ndarray):
        """Save a preset preview thumbnail (as JPEG blob)."""
        if img is None:
            return

        # Convert float32 (0-1) to uint8 (0-255) for JPEG encoding
        if img.dtype == np.float32:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Encode as JPEG (image is already sized appropriately)
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
        blob = buffer.tobytes()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preset_thumbnails (image_hash, preset_key, thumbnail, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(image_hash, preset_key) DO UPDATE SET
                    thumbnail = excluded.thumbnail,
                    updated_at = CURRENT_TIMESTAMP
            """, (image_hash, preset_key, blob))
            conn.commit()

        # Evict old entries if cache is too large
        self._evict_preset_cache_if_needed()

    def _evict_preset_cache_if_needed(self):
        """Evict oldest preset thumbnails if cache exceeds size limit."""
        with sqlite3.connect(self.db_path) as conn:
            # Check current cache size
            cursor = conn.execute("SELECT SUM(LENGTH(thumbnail)) FROM preset_thumbnails")
            total_bytes = cursor.fetchone()[0] or 0
            max_bytes = MAX_PRESET_CACHE_MB * 1024 * 1024

            if total_bytes > max_bytes:
                # Delete oldest entries until under 80% of limit
                target_bytes = int(max_bytes * 0.8)
                bytes_to_delete = total_bytes - target_bytes

                # Get oldest entries by updated_at
                cursor = conn.execute("""
                    SELECT image_hash, preset_key, LENGTH(thumbnail) as size
                    FROM preset_thumbnails
                    ORDER BY updated_at ASC
                """)

                deleted_bytes = 0
                to_delete = []
                for row in cursor.fetchall():
                    if deleted_bytes >= bytes_to_delete:
                        break
                    to_delete.append((row[0], row[1]))
                    deleted_bytes += row[2]

                # Delete the selected entries
                for image_hash, preset_key in to_delete:
                    conn.execute(
                        "DELETE FROM preset_thumbnails WHERE image_hash = ? AND preset_key = ?",
                        (image_hash, preset_key)
                    )
                conn.commit()

    def load_preset_thumbnail(self, image_hash: str, preset_key: str) -> Optional[np.ndarray]:
        """Load a preset preview thumbnail. Returns BGR numpy array or None."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT thumbnail FROM preset_thumbnails WHERE image_hash = ? AND preset_key = ?",
                (image_hash, preset_key)
            )
            row = cursor.fetchone()
            if row and row[0]:
                arr = np.frombuffer(row[0], dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return img
            return None

    def load_all_preset_thumbnails(self, image_hash: str) -> dict:
        """Load all preset thumbnails for an image. Returns {preset_key: BGR array}."""
        result = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT preset_key, thumbnail FROM preset_thumbnails WHERE image_hash = ?",
                (image_hash,)
            )
            for row in cursor.fetchall():
                if row[1]:
                    arr = np.frombuffer(row[1], dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    result[row[0]] = img
        return result

    def save_all(self, image_hash: str, settings: dict, thumbnail: np.ndarray):
        """Save both settings and thumbnail in a single transaction."""
        if thumbnail is not None:
            # Convert float32 (0-1) to uint8 (0-255) for thumbnail
            if thumbnail.dtype == np.float32:
                thumbnail = (np.clip(thumbnail, 0, 1) * 255).astype(np.uint8)

            # Prepare thumbnail blob
            if len(thumbnail.shape) == 3 and thumbnail.shape[2] == 3:
                img_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = thumbnail

            h, w = img_rgb.shape[:2]
            scale = min(THUMB_WIDTH / w, THUMB_HEIGHT / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_scaled = cv2.resize(img_rgb, (new_w, new_h))
            img_bgr = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
            blob = buffer.tobytes()
        else:
            blob = None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO images (hash, settings, thumbnail, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(hash) DO UPDATE SET
                    settings = excluded.settings,
                    thumbnail = excluded.thumbnail,
                    updated_at = CURRENT_TIMESTAMP
            """, (image_hash, json.dumps(settings), blob))
            conn.commit()

    def delete(self, image_hash: str):
        """Delete settings and thumbnail for an image."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM images WHERE hash = ?", (image_hash,))
            conn.commit()

    def clear_all(self):
        """Clear all stored data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM images")
            conn.commit()

    def get_stats(self) -> dict:
        """Get storage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*), SUM(LENGTH(thumbnail)) FROM images")
            row = cursor.fetchone()
            return {
                'image_count': row[0] or 0,
                'thumbnail_bytes': row[1] or 0,
            }

    def get_favorite_presets(self) -> list:
        """Get list of favorite preset keys."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'favorite_presets'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return []

    def set_favorite_presets(self, favorites: list):
        """Save list of favorite preset keys."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('favorite_presets', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(favorites),))
            conn.commit()

    def get_preset_order(self) -> list:
        """Get custom preset order (list of preset keys)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'preset_order'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return []

    def set_preset_order(self, order: list):
        """Save custom preset order."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('preset_order', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(order),))
            conn.commit()

    def get_preset_panel_expanded(self) -> bool:
        """Get preset panel expanded state. Defaults to True."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'preset_panel_expanded'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return True  # Default to expanded

    def set_preset_panel_expanded(self, expanded: bool):
        """Save preset panel expanded state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('preset_panel_expanded', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(expanded),))
            conn.commit()

    def get_adjustments_panel_expanded(self) -> bool:
        """Get adjustments panel expanded state. Defaults to True."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'adjustments_panel_expanded'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return True  # Default to expanded

    def set_adjustments_panel_expanded(self, expanded: bool):
        """Save adjustments panel expanded state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('adjustments_panel_expanded', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(expanded),))
            conn.commit()

    def get_show_preset_name_on_change(self) -> bool:
        """Get whether to show preset name overlay on change. Defaults to True."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'show_preset_name_on_change'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return True  # Default to showing

    def set_show_preset_name_on_change(self, show: bool):
        """Save whether to show preset name overlay on change."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('show_preset_name_on_change', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(show),))
            conn.commit()

    def get_preset_panel_startup_behavior(self) -> str:
        """Get preset panel startup behavior. Returns 'last', 'expanded', or 'collapsed'."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'preset_panel_startup_behavior'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return 'last'  # Default to remembering last position

    def set_preset_panel_startup_behavior(self, behavior: str):
        """Save preset panel startup behavior ('last', 'expanded', or 'collapsed')."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('preset_panel_startup_behavior', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(behavior),))
            conn.commit()

    def get_adjustments_panel_startup_behavior(self) -> str:
        """Get adjustments panel startup behavior. Returns 'last', 'expanded', or 'collapsed'."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'adjustments_panel_startup_behavior'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return 'last'  # Default to remembering last position

    def set_adjustments_panel_startup_behavior(self, behavior: str):
        """Save adjustments panel startup behavior ('last', 'expanded', or 'collapsed')."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('adjustments_panel_startup_behavior', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(behavior),))
            conn.commit()

    def get_debug_panel_expanded(self) -> bool:
        """Get debug panel expanded state. Defaults to True."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'debug_panel_expanded'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return True  # Default to expanded

    def set_debug_panel_expanded(self, expanded: bool):
        """Save debug panel expanded state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('debug_panel_expanded', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(expanded),))
            conn.commit()

    def get_debug_panel_startup_behavior(self) -> str:
        """Get debug panel startup behavior. Returns 'last', 'expanded', or 'collapsed'."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'debug_panel_startup_behavior'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return 'last'  # Default to remembering last position

    def set_debug_panel_startup_behavior(self, behavior: str):
        """Save debug panel startup behavior ('last', 'expanded', or 'collapsed')."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('debug_panel_startup_behavior', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(behavior),))
            conn.commit()

    def get_controls_panel_expanded(self) -> bool:
        """Get controls panel expanded state. Defaults to True."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'controls_panel_expanded'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return True  # Default to expanded

    def set_controls_panel_expanded(self, expanded: bool):
        """Save controls panel expanded state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('controls_panel_expanded', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(expanded),))
            conn.commit()

    def get_controls_panel_startup_behavior(self) -> str:
        """Get controls panel startup behavior. Returns 'last', 'expanded', or 'collapsed'."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'controls_panel_startup_behavior'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return 'last'  # Default to remembering last position

    def set_controls_panel_startup_behavior(self, behavior: str):
        """Save controls panel startup behavior ('last', 'expanded', or 'collapsed')."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('controls_panel_startup_behavior', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(behavior),))
            conn.commit()

    def get_startup_tab(self) -> str:
        """Get startup tab preference. Returns 'detection', 'development', or 'last'."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'startup_tab'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return 'development'  # Default to development tab

    def set_startup_tab(self, tab: str):
        """Save startup tab preference ('detection', 'development', or 'last')."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('startup_tab', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(tab),))
            conn.commit()

    def get_last_opened_tab(self) -> int:
        """Get the last opened tab index (0=Detection, 1=Development)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'last_opened_tab'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return 1  # Default to development tab

    def set_last_opened_tab(self, index: int):
        """Save the last opened tab index (0=Detection, 1=Development)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('last_opened_tab', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(index),))
            conn.commit()

    def get_default_aspect_ratio(self) -> str:
        """Get default crop aspect ratio key."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'default_aspect_ratio'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return '35mm'  # Default to 35mm

    def set_default_aspect_ratio(self, ratio_key: str):
        """Save default crop aspect ratio key."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('default_aspect_ratio', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(ratio_key),))
            conn.commit()

    def get_crop_invert_startup_behavior(self) -> str:
        """Get crop invert mode startup behavior. Returns 'last', 'on', or 'off'."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'crop_invert_startup_behavior'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return 'off'  # Default to off

    def set_crop_invert_startup_behavior(self, behavior: str):
        """Save crop invert mode startup behavior ('last', 'on', or 'off')."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('crop_invert_startup_behavior', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(behavior),))
            conn.commit()

    def get_crop_invert_state(self) -> bool:
        """Get the last crop invert state. Defaults to False."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'crop_invert_state'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return False

    def set_crop_invert_state(self, state: bool):
        """Save the current crop invert state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('crop_invert_state', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(state),))
            conn.commit()

    def get_favorite_images(self) -> set:
        """Get set of favorite image hashes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'favorite_images'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return set(json.loads(row[0]))
            return set()

    def set_favorite_images(self, favorites: set):
        """Save set of favorite image hashes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('favorite_images', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(list(favorites)),))
            conn.commit()

    def toggle_favorite_image(self, image_hash: str) -> bool:
        """Toggle favorite status for an image. Returns new favorite state."""
        favorites = self.get_favorite_images()
        if image_hash in favorites:
            favorites.discard(image_hash)
            is_favorite = False
        else:
            favorites.add(image_hash)
            is_favorite = True
        self.set_favorite_images(favorites)
        return is_favorite

    def get_user_presets(self) -> dict:
        """Get all user-saved presets. Returns {key: preset_dict}."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM preferences WHERE key = 'user_presets'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return {}

    def save_user_preset(self, key: str, preset: dict):
        """Save a user preset. Preset should have 'name', 'description', 'adjustments', 'curves'."""
        presets = self.get_user_presets()
        presets[key] = preset
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO preferences (key, value)
                VALUES ('user_presets', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (json.dumps(presets),))
            conn.commit()

    def delete_user_preset(self, key: str) -> bool:
        """Delete a user preset by key. Returns True if deleted, False if not found."""
        presets = self.get_user_presets()
        if key in presets:
            del presets[key]
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO preferences (key, value)
                    VALUES ('user_presets', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """, (json.dumps(presets),))
                conn.commit()
            return True
        return False

    def save_raw_cache(self, file_hash: str, img: np.ndarray):
        """Cache a demosaiced RAW image to disk as 16-bit lossless PNG.

        Input: float32 BGR image normalized to 0.0-1.0 range
        Storage: 16-bit PNG (preserves full RAW dynamic range)
        """
        if img is None:
            return
        cache_path = RAW_CACHE_DIR / f"{file_hash}.png"
        # Convert from float32 (0-1) to uint16 (0-65535) for storage
        img_16bit = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        # Use fastest PNG compression (1) - still lossless but faster to write
        cv2.imwrite(str(cache_path), img_16bit, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def load_raw_cache(self, file_hash: str) -> Optional[np.ndarray]:
        """Load a cached demosaiced RAW image. Returns float32 BGR (0-1) or None."""
        png_path = RAW_CACHE_DIR / f"{file_hash}.png"
        if png_path.exists():
            # Load preserving bit depth
            img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.dtype == np.uint16:
                    # New 16-bit format
                    return img.astype(np.float32) / 65535.0
                elif img.dtype == np.uint8:
                    # Old 8-bit format (for backwards compatibility during transition)
                    return img.astype(np.float32) / 255.0
            return None
        # Check for old JPG format for backwards compatibility
        jpg_path = RAW_CACHE_DIR / f"{file_hash}.jpg"
        if jpg_path.exists():
            img = cv2.imread(str(jpg_path))
            if img is not None:
                return img.astype(np.float32) / 255.0
        return None

    def has_raw_cache(self, file_hash: str) -> bool:
        """Check if a demosaiced RAW image is cached."""
        png_path = RAW_CACHE_DIR / f"{file_hash}.png"
        if png_path.exists():
            return True
        jpg_path = RAW_CACHE_DIR / f"{file_hash}.jpg"
        return jpg_path.exists()

    def clear_raw_cache(self):
        """Clear all cached demosaiced RAW images."""
        for ext in ["*.png", "*.jpg"]:
            for f in RAW_CACHE_DIR.glob(ext):
                f.unlink()

    def get_raw_cache_stats(self) -> dict:
        """Get RAW cache statistics."""
        files = list(RAW_CACHE_DIR.glob("*.png")) + list(RAW_CACHE_DIR.glob("*.jpg"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            'file_count': len(files),
            'total_bytes': total_size,
        }


# Global storage instance
_storage = None


def get_storage() -> Storage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
