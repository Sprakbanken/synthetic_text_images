import random

import numpy as np
from augraphy import AugmentationSequence, AugraphyPipeline, augmentations

from synthetic_ocr_data.image_processing import distort_line_image
from synthetic_ocr_data.augraphy_utils import PipelineWrapper


def test_line_image_is_distorted(example_line_image):
    """The line image should be distorted."""
    image, bbox = example_line_image
    ink_phase = [
        augmentations.Geometric(translation=(0.2, 0.2)),
        augmentations.Squish(
            squish_direction=1,
            squish_location=[0, 1],
            squish_number_range=(5, 5),
            squish_distance_range=(5, 5),
            squish_line=0,
            squish_line_thickness_range=(1, 1),
        ),
    ]

    paper_phase = []

    post_phase = [
        augmentations.Folding(
            fold_x=20,
            fold_deviation=(0, 0),
            fold_count=1,
            fold_noise=0,
            fold_angle_range=(0, 0),
            gradient_width=(0.1, 0.1),
            gradient_height=(0.01, 0.01),
            backdrop_color=(0, 0, 0),
        )
    ]

    pipeline = PipelineWrapper(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        pre_phase=[],
        bounding_boxes=[bbox],
        seed=0,
    )
    random.seed(0)
    distorted_image, bbox, log = distort_line_image(image, pipeline)

    assert distorted_image is not None
    assert bbox is not None
    assert log is not None

    # using the pipeline directly should still give the same result with the same seed
    random.seed(0)
    augmented_image = pipeline.augment(np.array(image))["output"]

    np.testing.assert_allclose(np.array(distorted_image), augmented_image)
