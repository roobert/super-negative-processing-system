"""
Film Presets for SUPER NEGATIVE PROCESSING SYSTEM

Presets contain adjustments and curves that emulate various film stocks.
Each preset is a dictionary with 'adjustments' and 'curves' keys matching
the structure used in image_settings.json.
"""

# Default identity curves (no change)
IDENTITY_CURVES = {
    'rgb': [(0, 0), (255, 255)],
    'r': [(0, 0), (255, 255)],
    'g': [(0, 0), (255, 255)],
    'b': [(0, 0), (255, 255)],
}

# Default adjustments (all neutral)
DEFAULT_ADJUSTMENTS = {
    'exposure': 0,
    'wb_r': 1.0,  # White balance red multiplier (0.5 - 2.0)
    'wb_g': 1.0,  # White balance green multiplier
    'wb_b': 1.0,  # White balance blue multiplier
    'contrast': 0,
    'highlights': 0,
    'shadows': 0,
    'temperature': 0,
    'vibrance': 0,
    'saturation': 0,
    'blacks': 0,
    'whites': 0,
    'gamma': 1.0,
    'sharpening': 0,
}


def _make_preset(name: str, description: str, adjustments: dict = None, curves: dict = None) -> dict:
    """Helper to create a preset with defaults filled in."""
    adj = DEFAULT_ADJUSTMENTS.copy()
    if adjustments:
        adj.update(adjustments)

    crv = {ch: list(pts) for ch, pts in IDENTITY_CURVES.items()}
    if curves:
        for ch, pts in curves.items():
            crv[ch] = pts

    return {
        'name': name,
        'description': description,
        'adjustments': adj,
        'curves': crv,
    }


# ============================================================================
# FILM PRESETS
# ============================================================================

PRESETS = {
    # None preset - restores to default
    'none': _make_preset(
        'None',
        'No preset applied - neutral settings',
    ),

    # Kodak Portra 400 - Natural skin tones, slightly warm, soft contrast
    'kodak_portra_400': _make_preset(
        'Kodak Portra 400',
        'Natural colors, excellent skin tones, warm highlights',
        adjustments={
            'temperature': 8,
            'contrast': -10,
            'highlights': -15,
            'shadows': 20,
            'vibrance': 10,
            'saturation': -5,
            'gamma': 1.05,
        },
        curves={
            'rgb': [(0, 8), (64, 70), (192, 200), (255, 250)],  # Lifted blacks, rolled highlights
            'r': [(0, 0), (128, 132), (255, 255)],  # Slight warmth in midtones
            'g': [(0, 0), (255, 255)],
            'b': [(0, 5), (255, 250)],  # Slightly lifted shadows, subdued highlights
        }
    ),

    # Kodak Portra 160 - Lower contrast, pastel colors
    'kodak_portra_160': _make_preset(
        'Kodak Portra 160',
        'Low contrast, pastel tones, professional portrait film',
        adjustments={
            'temperature': 5,
            'contrast': -15,
            'highlights': -20,
            'shadows': 25,
            'vibrance': 5,
            'saturation': -10,
            'gamma': 1.10,
        },
        curves={
            'rgb': [(0, 15), (64, 75), (192, 195), (255, 245)],  # Very lifted blacks, soft highs
            'r': [(0, 0), (255, 255)],
            'g': [(0, 0), (255, 255)],
            'b': [(0, 8), (255, 248)],
        }
    ),

    # Kodak Ektar 100 - Vivid colors, high saturation, great for landscapes
    'kodak_ektar_100': _make_preset(
        'Kodak Ektar 100',
        'Vivid colors, fine grain, excellent for landscapes',
        adjustments={
            'temperature': -5,
            'contrast': 15,
            'highlights': -10,
            'shadows': 10,
            'vibrance': 25,
            'saturation': 15,
            'gamma': 1.0,
        },
        curves={
            'rgb': [(0, 0), (64, 58), (192, 200), (255, 255)],  # S-curve for punch
            'r': [(0, 0), (128, 125), (255, 255)],
            'g': [(0, 0), (255, 255)],
            'b': [(0, 0), (128, 135), (255, 255)],  # Boost blues
        }
    ),

    # Kodak Gold 200 - Consumer film, warm tones, nostalgic
    'kodak_gold_200': _make_preset(
        'Kodak Gold 200',
        'Warm, nostalgic look, classic consumer film',
        adjustments={
            'temperature': 15,
            'contrast': 5,
            'highlights': -5,
            'shadows': 15,
            'vibrance': 15,
            'saturation': 10,
            'gamma': 1.0,
        },
        curves={
            'rgb': [(0, 5), (255, 252)],
            'r': [(0, 0), (128, 138), (255, 255)],  # Warm push
            'g': [(0, 0), (128, 130), (255, 255)],
            'b': [(0, 0), (128, 118), (255, 250)],  # Reduce blues
        }
    ),

    # Kodak Tri-X 400 - Classic B&W with rich blacks
    'kodak_trix_400': _make_preset(
        'Kodak Tri-X 400',
        'Classic black & white, rich tones, iconic grain structure',
        adjustments={
            'saturation': -100,
            'contrast': 20,
            'highlights': -10,
            'shadows': 15,
            'blacks': 5,
            'gamma': 0.95,
        },
        curves={
            'rgb': [(0, 0), (48, 35), (128, 130), (208, 220), (255, 255)],  # Classic film S-curve
            'r': [(0, 0), (255, 255)],
            'g': [(0, 0), (255, 255)],
            'b': [(0, 0), (255, 255)],
        }
    ),

    # Ilford HP5 Plus 400 - Another classic B&W, slightly softer
    'ilford_hp5_400': _make_preset(
        'Ilford HP5 Plus 400',
        'Versatile black & white, smooth tones, wide latitude',
        adjustments={
            'saturation': -100,
            'contrast': 10,
            'highlights': -15,
            'shadows': 20,
            'gamma': 1.0,
        },
        curves={
            'rgb': [(0, 5), (64, 60), (192, 200), (255, 252)],
            'r': [(0, 0), (255, 255)],
            'g': [(0, 0), (255, 255)],
            'b': [(0, 0), (255, 255)],
        }
    ),

    # Fujifilm Velvia 50 - High saturation slide film
    'fuji_velvia_50': _make_preset(
        'Fuji Velvia 50',
        'Ultra-vivid colors, high contrast, legendary slide film',
        adjustments={
            'temperature': -8,
            'contrast': 25,
            'highlights': -15,
            'shadows': -5,
            'vibrance': 35,
            'saturation': 25,
            'blacks': 3,
            'gamma': 0.90,
        },
        curves={
            'rgb': [(0, 0), (48, 38), (208, 220), (255, 255)],  # Strong S-curve
            'r': [(0, 0), (128, 125), (255, 255)],
            'g': [(0, 0), (128, 135), (255, 255)],  # Boost greens
            'b': [(0, 0), (128, 140), (255, 255)],  # Strong blues
        }
    ),

    # Fujifilm Provia 100F - Balanced slide film
    'fuji_provia_100f': _make_preset(
        'Fuji Provia 100F',
        'Balanced colors, natural saturation, professional slide film',
        adjustments={
            'temperature': -3,
            'contrast': 15,
            'highlights': -10,
            'shadows': 5,
            'vibrance': 15,
            'saturation': 10,
            'gamma': 0.95,
        },
        curves={
            'rgb': [(0, 0), (64, 58), (192, 198), (255, 255)],
            'r': [(0, 0), (255, 255)],
            'g': [(0, 0), (128, 132), (255, 255)],
            'b': [(0, 0), (128, 132), (255, 255)],
        }
    ),

    # Fujifilm Superia 400 - Consumer film, versatile
    'fuji_superia_400': _make_preset(
        'Fuji Superia 400',
        'Versatile consumer film, slightly cool, good colors',
        adjustments={
            'temperature': -5,
            'contrast': 8,
            'highlights': -8,
            'shadows': 12,
            'vibrance': 12,
            'saturation': 5,
            'gamma': 1.0,
        },
        curves={
            'rgb': [(0, 3), (255, 253)],
            'r': [(0, 0), (255, 255)],
            'g': [(0, 0), (128, 132), (255, 255)],
            'b': [(0, 5), (128, 135), (255, 255)],  # Cool blue push
        }
    ),

    # Fujifilm Pro 400H - Wedding/portrait film, soft pastels
    'fuji_pro_400h': _make_preset(
        'Fuji Pro 400H',
        'Soft pastels, beautiful skin tones, wedding favorite',
        adjustments={
            'temperature': 3,
            'contrast': -12,
            'highlights': -18,
            'shadows': 22,
            'vibrance': 8,
            'saturation': -8,
            'gamma': 1.05,
        },
        curves={
            'rgb': [(0, 12), (192, 198), (255, 248)],
            'r': [(0, 0), (255, 255)],
            'g': [(0, 2), (255, 253)],
            'b': [(0, 8), (128, 135), (255, 252)],
        }
    ),

    # Cinestill 800T - Tungsten-balanced cinema film
    'cinestill_800t': _make_preset(
        'CineStill 800T',
        'Tungsten cinema film, halation glow, night photography',
        adjustments={
            'temperature': -25,
            'contrast': 10,
            'highlights': 5,
            'shadows': 15,
            'vibrance': 20,
            'saturation': 5,
            'gamma': 1.0,
        },
        curves={
            'rgb': [(0, 5), (255, 252)],
            'r': [(0, 0), (64, 55), (192, 195), (255, 255)],
            'g': [(0, 0), (128, 125), (255, 255)],
            'b': [(0, 8), (128, 145), (255, 255)],  # Strong blue cast
        }
    ),

    # Agfa Vista 200 - Punchy colors, slightly magenta
    'agfa_vista_200': _make_preset(
        'Agfa Vista 200',
        'Punchy colors, magenta cast, discontinued favorite',
        adjustments={
            'temperature': 5,
            'contrast': 12,
            'highlights': -8,
            'shadows': 10,
            'vibrance': 18,
            'saturation': 12,
            'gamma': 1.0,
        },
        curves={
            'rgb': [(0, 3), (255, 253)],
            'r': [(0, 0), (128, 135), (255, 255)],
            'g': [(0, 0), (128, 122), (255, 255)],  # Reduce greens
            'b': [(0, 0), (128, 130), (255, 255)],
        }
    ),

    # Lomography 400 - Cross-processed look
    'lomo_400': _make_preset(
        'Lomography 400',
        'Experimental colors, vignette-ready, lo-fi aesthetic',
        adjustments={
            'temperature': 10,
            'contrast': 20,
            'highlights': 10,
            'shadows': -10,
            'vibrance': 30,
            'saturation': 20,
            'gamma': 0.95,
        },
        curves={
            'rgb': [(0, 0), (48, 30), (208, 225), (255, 255)],  # Strong S-curve
            'r': [(0, 0), (128, 140), (255, 255)],
            'g': [(0, 5), (128, 125), (255, 250)],
            'b': [(0, 10), (128, 130), (255, 245)],
        }
    ),

    # Faded Film - Vintage faded look
    'faded_film': _make_preset(
        'Faded Film',
        'Vintage faded look, lifted blacks, nostalgic',
        adjustments={
            'temperature': 8,
            'contrast': -20,
            'highlights': -25,
            'shadows': 30,
            'vibrance': -15,
            'saturation': -20,
            'gamma': 1.1,
        },
        curves={
            'rgb': [(0, 25), (64, 80), (192, 195), (255, 235)],  # Very lifted, compressed
            'r': [(0, 5), (255, 250)],
            'g': [(0, 3), (255, 252)],
            'b': [(0, 8), (255, 248)],
        }
    ),

    # High Contrast B&W
    'high_contrast_bw': _make_preset(
        'High Contrast B&W',
        'Dramatic black & white, deep shadows, bright highlights',
        adjustments={
            'saturation': -100,
            'contrast': 35,
            'highlights': 10,
            'shadows': -15,
            'blacks': 8,
            'whites': 5,
            'gamma': 0.90,
        },
        curves={
            'rgb': [(0, 0), (32, 15), (224, 240), (255, 255)],
            'r': [(0, 0), (255, 255)],
            'g': [(0, 0), (255, 255)],
            'b': [(0, 0), (255, 255)],
        }
    ),
}

# Ordered list for UI display
PRESET_ORDER = [
    'none',
    'kodak_portra_400',
    'kodak_portra_160',
    'kodak_ektar_100',
    'kodak_gold_200',
    'kodak_trix_400',
    'ilford_hp5_400',
    'fuji_velvia_50',
    'fuji_provia_100f',
    'fuji_superia_400',
    'fuji_pro_400h',
    'cinestill_800t',
    'agfa_vista_200',
    'lomo_400',
    'faded_film',
    'high_contrast_bw',
]


def get_preset(key: str) -> dict:
    """Get a preset by key (built-in or user). Returns None preset if not found."""
    # Check built-in presets first
    if key in PRESETS:
        return PRESETS[key]
    # Check user presets
    import storage
    user_presets = storage.get_storage().get_user_presets()
    return user_presets.get(key, PRESETS['none'])


def get_preset_list() -> list:
    """Get list of (key, name, description) tuples in display order.

    Returns built-in presets first, then user presets.
    """
    result = []
    # Built-in presets in defined order
    for key in PRESET_ORDER:
        preset = PRESETS.get(key)
        if preset:
            result.append((key, preset['name'], preset['description']))

    # User presets (sorted by name)
    import storage
    user_presets = storage.get_storage().get_user_presets()
    user_items = [(k, p['name'], p.get('description', '')) for k, p in user_presets.items()]
    user_items.sort(key=lambda x: x[1].lower())  # Sort by name
    result.extend(user_items)

    return result


def is_user_preset(key: str) -> bool:
    """Check if a preset key is a user-created preset."""
    return key not in PRESETS


def create_user_preset(name: str, description: str, adjustments: dict, curves: dict) -> str:
    """Create a new user preset and save it to storage.

    Args:
        name: Display name for the preset
        description: Brief description
        adjustments: Adjustment values dict
        curves: Curves dict with rgb, r, g, b keys

    Returns:
        The generated preset key
    """
    import storage
    import time

    # Generate a unique key from name + timestamp
    key = f"user_{name.lower().replace(' ', '_')}_{int(time.time())}"

    preset = {
        'name': name,
        'description': description,
        'adjustments': adjustments.copy(),
        'curves': {ch: list(pts) for ch, pts in curves.items()},
    }

    storage.get_storage().save_user_preset(key, preset)
    return key


def delete_user_preset(key: str) -> bool:
    """Delete a user preset.

    Args:
        key: The preset key to delete

    Returns:
        True if deleted, False if not found or is a built-in preset
    """
    if key in PRESETS:
        return False  # Can't delete built-in presets
    import storage
    return storage.get_storage().delete_user_preset(key)
