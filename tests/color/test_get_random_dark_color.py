import random

from text_generation.image_creation import get_random_dark_color


def test_average_rgb_value_low(rng):
    """The average RGB value of a dark color should be low."""
    for i in range(100):
        color = get_random_dark_color(rng)
        mean_rgb = sum(color) / 3
        assert mean_rgb < 255 / 2, color
