from PIL import ImageFont

from synthetic_ocr_data.fonts import check_font_support


def test_font_support_returns_true():
    """When a font supports all characters in a text it should return True."""

    text = "Bures boahtin!"
    font = ImageFont.load_default()

    assert check_font_support(text, font.path)


def test_font_support_returns_false(fanwood_italic_font: ImageFont.FreeTypeFont):
    """When a font does not support all characters in a text it should return False."""

    text = "ÁáÂâÄäČčƷʒǮǯĐđǦǧǤǥǨǩŊŋÕõŠšŽžÖö"

    assert not check_font_support(text, fanwood_italic_font.path)
