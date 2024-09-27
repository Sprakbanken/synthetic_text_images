import random

import colorspacious
import numpy as np


def get_dark_mode_color_pair(
    rng: random.Random,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Create a color pair suitable for dark mode."""
    text_color = get_random_light_color(rng)
    background_color = get_random_dark_color(rng)

    return text_color, background_color


def get_light_mode_color_pair(
    rng: random.Random,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Create a color pair suitable for light mode."""
    text_color = get_random_dark_color(rng)
    background_color = get_random_light_color(rng)

    return text_color, background_color


def is_dark_mode(text_color: tuple[int, int, int], background_color: tuple[int, int, int]) -> bool:
    """Determine if the color combination is suitable for dark mode."""
    text_luminance = colorspacious.cspace_convert(text_color, "sRGB1", "CAM02-UCS")[0]
    background_luminance = colorspacious.cspace_convert(background_color, "sRGB1", "CAM02-UCS")[0]

    return bool(text_luminance > background_luminance)


def get_random_dark_color(rng: random.Random) -> tuple[float, float, float]:
    """Create a random dark color in JCh color space and convert it to sRGB1."""
    # Generate random values for J, C, and h
    J = rng.uniform(0, 20)  # Low lightness for dark color
    C = rng.uniform(0, 20)  # Chroma
    h = rng.uniform(0, 360)  # Hue

    # JCh to sRGB1 conversion
    srgb1_color = colorspacious.cspace_convert([J, C, h], "JCh", "sRGB255")

    return tuple(np.clip((srgb1_color).astype(int), 0, 255).tolist())


def get_random_light_color(rng: random.Random) -> tuple[float, float, float]:
    """Create a random light color in JCh color space and convert it to sRGB1."""
    # Generate random values for J, C, and h
    J = rng.uniform(80, 100)  # High lightness for light color
    C = rng.uniform(0, 100)  # Chroma
    h = rng.uniform(0, 360)  # Hue

    # JCh to sRGB1 conversion
    srgb1_color = colorspacious.cspace_convert([J, C, h], "JCh", "sRGB1")

    return tuple(np.clip((srgb1_color * 255).astype(int), 0, 255).tolist())


def get_random_highlight_color(rng: random.Random) -> tuple[float, float, float]:
    """Create a random highlight color in JCh color space and convert it to sRGB1."""
    # Generate random values for J, C, and h
    J = rng.uniform(40, 70)  # High lightness for light color
    C = rng.uniform(80, 100)  # Chroma
    h = rng.uniform(0, 360)  # Hue

    # JCh to sRGB1 conversion
    srgb1_color = colorspacious.cspace_convert([J, C, h], "JCh", "sRGB1")

    return tuple(np.clip((srgb1_color * 255).astype(int), 0, 255).tolist())
