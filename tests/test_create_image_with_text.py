import numpy as np
from PIL import Image

from text_generation import create_image_with_text


def test_image_is_created(ubuntu_sans_font):
    """When calling create_image_with_text an image file should be created."""

    text = "Bures boahtin!"
    color_pair = ((0, 0, 0), (255, 255, 255))
    top_margin = 10
    bottom_margin = 20
    left_margin = 30
    right_margin = 40

    image, bbox = create_image_with_text(
        text=text,
        font=ubuntu_sans_font,
        color_pair=color_pair,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        left_margin=left_margin,
        right_margin=right_margin,
    )

    assert isinstance(image, Image.Image)
    assert bbox is not None

    left, top, right, bottom = bbox
    assert top > 0
    assert left > 0
    assert right > left
    assert bottom > top

    width, height = image.size

    # We check less than or equal because the boudning box might have a little margin
    # for some characters and this is also included in the margin, so when adding the margin
    # to the bounding box it can be a little larger than the image.
    assert width <= right + right_margin
    assert height <= bottom + bottom_margin

def test_created_image_is_not_empty(ubuntu_sans_font):
    """The created image should not be empty."""
    text = "Bures boahtin!"
    color_pair = ((0, 0, 0), (255, 255, 255))
    top_margin = 10
    bottom_margin = 20
    left_margin = 30
    right_margin = 40

    image, _ = create_image_with_text(
        text=text,
        font=ubuntu_sans_font,
        color_pair=color_pair,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        left_margin=left_margin,
        right_margin=right_margin,
    )

    image_array = np.array(image)
    corner_color = image_array[0, 0]
    assert not np.all(corner_color == image_array)
