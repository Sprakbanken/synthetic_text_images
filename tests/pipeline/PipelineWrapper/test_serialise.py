import random

import numpy as np
from augraphy import augmentations

from text_generation.pipeline import PipelineWrapper


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
