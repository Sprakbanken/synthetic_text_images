import json
import random
from pathlib import Path
from shutil import copy2
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from text_generation.pipeline import Augmenter

BoundingBox = tuple[int, int, int, int]


def get_bbox_aware_crop_box(
    image_size: tuple[int, int],
    bounding_box: BoundingBox,
    buffer_margin: int = 0,
    rng: random.Random | None = None,
) -> BoundingBox:
    """
    Get a random bounding box-aware crop box within an image.

    Args:
        image_size (tuple[int, int]): The size of the image (width, height).
        bounding_box (tuple[int, int, int, int]): The bounding box coordinates (left, top, right, bottom).
        buffer_margin (int, optional): The buffer margin around the bounding box. Defaults to 0.
        rng ([type], optional): The random number generator. Defaults to None.

    Returns:
        tuple[int, int, int, int]: The crop box coordinates (top, left, bottom, right).

    Examples:
        >>> image_size = (100, 100)
        >>> bounding_box = (10, 10, 90, 90)
        >>> get_bbox_aware_crop_box(image_size, bounding_box, buffer_margin=10, rng=random.Random(0))
        (0, 0, 100, 100)

        >>> image_size = (200, 200)
        >>> bounding_box = (50, 50, 150, 150)
        >>> crop_box = get_bbox_aware_crop_box(image_size, bounding_box, buffer_margin=0, rng=random.Random(1))
        >>> 0 <= crop_box[0] < bounding_box[0]
        True
        >>> 0 <= crop_box[1] < bounding_box[1]
        True
        >>> image_size[1] > crop_box[2] >= bounding_box[2]
        True
        >>> image_size[0] > crop_box[3] >= bounding_box[3]
        True
    """
    if rng is None:
        rng = random.Random()

    bbox_top, bbox_left, bbox_bottom, bbox_right = bounding_box

    crop_top = rng.randint(0, bbox_top - buffer_margin)
    crop_left = rng.randint(0, bbox_left - buffer_margin)

    crop_bottom = rng.randint(bbox_bottom + buffer_margin, image_size[1])
    crop_right = rng.randint(bbox_right + buffer_margin, image_size[0])
    return crop_top, crop_left, crop_bottom, crop_right


def get_bounding_box_and_image_size(
    text: str,
    font: ImageFont.FreeTypeFont,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Get the bounding box and image size of a text with a specific font, font size and margins."""

    # Calculate text size
    (left, top, right, bottom) = font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    # Calculate image size with margins
    image_width = text_width + left_margin + right_margin
    image_height = text_height + top_margin + bottom_margin

    # Set bounding box coordinates with margins
    bbox = (left + left_margin, top + top_margin, right + left_margin, bottom + top_margin)
    return bbox, (image_width, image_height)



def distort_line_image(
    image: Image.Image, pipeline: Augmenter
) -> tuple[Image.Image, tuple[int, int, int, int] | None, dict]:
    """Distort an image with an AugraphyPipeline."""
    image_array = np.array(image)
    out = pipeline.augment(image_array)
    distorted_image_array = out["output"]
    if out["bounding_boxes"] is None:
        distorted_bbox = None
    else:
        distorted_bbox = out["bounding_boxes"][0]  # Only one line, so only one bounding box
    log = pipeline.augment(image_array)["log"]

    return Image.fromarray(distorted_image_array), distorted_bbox, log


def distort_line_images(
    metadata: pd.DataFrame,
    output_dir: Path,
    rng: random.Random,
    pipeline_creator: Callable[[BoundingBox, random.Random], Augmenter],
) -> None:
    """Distort images and save the results."""
    original_images_dir = output_dir / "original_images"

    distorted_images_dir = output_dir / "distorted_images"
    distorted_images_dir.mkdir(exist_ok=True, parents=True)

    logs_dir = output_dir / "augraphy_logs"
    logs_dir.mkdir(exist_ok=True, parents=True)

    distorted_metadata = metadata.copy()

    for _, row in distorted_metadata.iterrows():
        image_path = output_dir / row["output_path"]
        # Copy the image to the original images directory
        original_image_path = original_images_dir / image_path.name
        copy2(image_path, original_image_path)

        image = Image.open(image_path)
        bounding_box = (row["bbox_left"], row["bbox_top"], row["bbox_right"], row["bbox_bottom"])

        pipeline = pipeline_creator(bounding_box, rng)
        distorted_image, distorted_bbox, log = distort_line_image(image, pipeline)

        # Save the distorted image
        distorted_image_path = distorted_images_dir / image_path.name
        distorted_image.save(distorted_image_path)

        # Save the log
        log_path = logs_dir / image_path.with_suffix(".json").name
        with open(log_path, "w") as f:
            json.dump(log, f)

        row["output_path"] = original_image_path.relative_to(output_dir)
        row["distorted_output_path"] = distorted_image_path.relative_to(output_dir)
        row["augraphy_log_path"] = log_path.relative_to(output_dir)

        row["distorted_bbox_left"] = distorted_bbox[0]
        row["distorted_bbox_top"] = distorted_bbox[1]
        row["distorted_bbox_right"] = distorted_bbox[2]
        row["distorted_bbox_bottom"] = distorted_bbox[3]

    distorted_metadata.to_csv(output_dir / "metadata.csv", index=False)
