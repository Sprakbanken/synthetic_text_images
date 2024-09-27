import random

from synthetic_ocr_data.image_creation import get_random_light_color


def test_average_rgb_value_high(rng):
    """The average RGB value of a light color should be high."""
    for i in range(100):
        color = get_random_light_color(rng)
        mean_rgb = sum(color) / 3
        assert mean_rgb > 255 / 2, color
