import random

import numpy as np
from augraphy import augmentations

from synthetic_ocr_data.augraphy_utils import PipelineWrapper, create_scanned_book_pipeline


def test_serialised_augmentations_can_be_recovered(example_line_image):
    image, bbox = example_line_image
    fold_x_rate = 0.3
    fold_x = int(fold_x_rate * image.size[1])
    pre_phase = []
    ink_phase = [augmentations.BrightnessTexturize(texturize_range=(0.9, 0.9), deviation=0.1, p=1)]

    paper_phase = []

    post_phase = [
        augmentations.Folding(
            fold_x=fold_x,
            fold_deviation=(0, 0),
            fold_count=1,
            fold_noise=0,
            fold_angle_range=(0, 0),
            gradient_width=(0.1, 0.1),
            gradient_height=(0.01, 0.01),
            backdrop_color=(0, 0, 0),
            p=1,
        )
    ]

    seed = 0

    pipelinewrapper = PipelineWrapper(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        pre_phase=pre_phase,
        bounding_boxes=None,
        seed=seed,
    )

    serialised_pipeline_wrapper = pipelinewrapper.to_dict()
    recovered_pipeline_wrapper = PipelineWrapper.from_dict(serialised_pipeline_wrapper)

    image_array = np.array(image)

    random.seed(seed)
    distorted_image_original_pipeline = pipelinewrapper.augment(image_array)["output"]
    random.seed(seed)
    distorted_image_recovered_pipeline = recovered_pipeline_wrapper.augment(image_array)["output"]
    np.testing.assert_allclose(
        distorted_image_original_pipeline,
        distorted_image_recovered_pipeline,
    )


def test_serialised_augmentations_can_be_recovered_full_pipeline(example_line_image):
    image, bbox = example_line_image
    fold_x_rate = 0.3
    fold_x = int(fold_x_rate * image.size[1])
    seed = 0

    pipelinewrapper = create_scanned_book_pipeline(
        bbox=bbox,
        image_size=image.size,
        rng=random.Random(0),
        text_color=(255, 0, 0),
        background_color=(0, 20, 20),
        bleed_through_candidates=tuple(),
        font_size=100,
    )

    serialised_pipeline_wrapper = pipelinewrapper.to_dict()
    recovered_pipeline_wrapper = PipelineWrapper.from_dict(serialised_pipeline_wrapper)

    image_array = np.array(image)

    random.seed(seed)
    distorted_image_original_pipeline = pipelinewrapper.augment(image_array)["output"]
    random.seed(seed)
    distorted_image_recovered_pipeline = recovered_pipeline_wrapper.augment(image_array)["output"]
    np.testing.assert_allclose(
        distorted_image_original_pipeline,
        distorted_image_recovered_pipeline,
    )
