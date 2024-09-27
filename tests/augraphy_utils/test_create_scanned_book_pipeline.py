import random

import numpy as np

from synthetic_ocr_data.augraphy_utils import Augmenter, create_scanned_book_pipeline


def test_pipeline_is_created(example_line_image):
    """A pipeline should be created and it should be possible to augment an image with it."""
    image, bbox = example_line_image
    rng = random.Random(42)
    pipeline = create_scanned_book_pipeline(bbox, image.size, rng, (0, 0, 0), (255, 255, 255))
    augmented_image = pipeline.augment(np.array(image))
    assert augmented_image is not None
    assert augmented_image["output"] is not None
    assert isinstance(pipeline, Augmenter)


def test_pipeline_output_is_not_blank(example_line_image):
    """The output of the pipeline should not be a blank image"""

    image, bbox = example_line_image
    rng = random.Random(42)
    pipeline = create_scanned_book_pipeline(bbox, image.size, rng, (0, 0, 0), (255, 255, 255))
    augmented_image = pipeline.augment(np.array(image))
    augmented_image_array = augmented_image["output"]

    corner_pixel = augmented_image_array[0, 0]
    assert not np.all(corner_pixel == augmented_image_array)
