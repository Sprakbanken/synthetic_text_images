from pathlib import Path

import pytest
from PIL import ImageFont


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
