from synthetic_ocr_data.color import get_dark_mode_color_pair, get_light_mode_color_pair, is_dark_mode


def test_darkmode_colors_are_classified_correctly(rng):
    """Dark mode colors should be classified correctly."""
    for i in range(100):
        text_color, background_color = get_dark_mode_color_pair(rng)
        assert is_dark_mode(text_color, background_color)


def test_lightmode_colors_are_classified_correctly(rng):
    """Light mode colors should be classified correctly."""
    for i in range(100):
        text_color, background_color = get_light_mode_color_pair(rng)
        assert not is_dark_mode(text_color, background_color)
