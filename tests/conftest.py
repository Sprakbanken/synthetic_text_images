import random
from pathlib import Path

import pytest
from PIL import Image, ImageFont

from text_generation.image_creation import create_line_image


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def font_directory():
    return Path(__file__).parent.parent / "fonts"


@pytest.fixture
def fanwood_italic_font(font_directory) -> ImageFont.FreeTypeFont:
    font_path = (
        font_directory / "fanwood-master/fanwood-master/webfonts/fanwood_text_italic-webfont.ttf"
    )
    return ImageFont.truetype(font_path, size=40)


@pytest.fixture
def ubuntu_sans_font(font_directory) -> ImageFont.FreeTypeFont:
    font_path = font_directory / "Ubuntu_Sans/UbuntuSans-VariableFont_wdth,wght.ttf"
    return ImageFont.truetype(font_path, size=40)


@pytest.fixture
def example_text() -> str:
    return "Bures boahtin!"


@pytest.fixture
def example_line_image(example_text: str, ubuntu_sans_font: ImageFont.FreeTypeFont) -> Image.Image:
    color_pair = ((0, 0, 0), (255, 255, 255))
    top_margin = 10
    bottom_margin = 20
    left_margin = 30
    right_margin = 40
    image, box = create_line_image(
        text=example_text,
        font=ubuntu_sans_font,
        color_pair=color_pair,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        left_margin=left_margin,
        right_margin=right_margin,
    )
    return image, box