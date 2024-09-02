import pytest
from PIL import ImageFont

from text_generation import get_bounding_box_and_image_size


@pytest.mark.parametrize("text", ["Bures boahtin!", "Velkommen!"])
@pytest.mark.parametrize("top_margin", [0, 20])
@pytest.mark.parametrize("bottom_margin", [0, 20])
@pytest.mark.parametrize("left_margin", [0, 20])
@pytest.mark.parametrize("right_margin", [0, 20])
def test_margin_and_bounding_box_and_image_size_adds_up(
    text, top_margin, bottom_margin, left_margin, right_margin
):
    """When adding the margin to the bounding box it should add up with the image size."""

    font = ImageFont.load_default()
    (left, top, right, bottom), (image_width, image_height) = get_bounding_box_and_image_size(
        text, font, top_margin, bottom_margin, left_margin, right_margin
    )

    assert right - left + left_margin + right_margin == image_width
    assert bottom - top + top_margin + bottom_margin == image_height


def test_longer_text_has_larger_image_size():
    """When the text is longer the image size should be larger."""

    text = "Bures boahtin!"
    font = ImageFont.load_default()
    top_margin = 10
    bottom_margin = 20
    left_margin = 30
    right_margin = 40

    (left, top, right, bottom), (image_width, image_height) = get_bounding_box_and_image_size(
        text, font, top_margin, bottom_margin, left_margin, right_margin
    )

    text2 = text * 2
    (left2, top2, right2, bottom2), (image_width2, image_height2) = get_bounding_box_and_image_size(
        text2, font, top_margin, bottom_margin, left_margin, right_margin
    )
    assert image_width2 > image_width
    assert image_height2 == image_height
