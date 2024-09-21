from __future__ import annotations

import json
import random
from shutil import copy2
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from text_generation.utils import parse_int_tuple

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from text_generation.augraphy_utils import Augmenter, PipelineCreator

BoundingBox = tuple[int, int, int, int]


def get_bbox_aware_crop_box(
    image_size: tuple[int, int],
    bounding_box: BoundingBox,
    buffer_margin_left: int = 0,
    buffer_margin_top: int = 0,
    buffer_margin_right: int = 0,
    buffer_margin_bottom: int = 0,
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
        tuple[int, int, int, int]: The crop box coordinates (left, top, right, bottom).

    Examples:
        >>> image_size = (100, 100)
        >>> bounding_box = (10, 10, 90, 90)
        >>> get_bbox_aware_crop_box(image_size, bounding_box, buffer_margin_left=10, buffer_margin_top=10, buffer_margin_right=10, buffer_margin_bottom=10, rng=random.Random(0))
        (0, 0, 100, 100)

        >>> image_size = (200, 200)
        >>> bounding_box = (50, 50, 150, 150)
        >>> crop_box = get_bbox_aware_crop_box(image_size, bounding_box, rng=random.Random(1))
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

    bbox_left, bbox_top, bbox_right, bbox_bottom = bounding_box

    crop_top = rng.randint(0, bbox_top - buffer_margin_top)
    crop_left = rng.randint(0, bbox_left - buffer_margin_left)

    crop_bottom = rng.randint(bbox_bottom + buffer_margin_bottom, image_size[1])
    crop_right = rng.randint(bbox_right + buffer_margin_right, image_size[0])
    return crop_left, crop_top, crop_right, crop_bottom


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

    # Calculate image size with margins
    image_width = right + left_margin + right_margin
    image_height = bottom + top_margin + bottom_margin

    # Set bounding box coordinates with margins
    bbox = (left + left_margin, top + top_margin, right + left_margin, bottom + top_margin)
    return bbox, (image_width, image_height)


def distort_line_image(
    image: Image.Image, pipeline: Augmenter
) -> tuple[Image.Image, BoundingBox | None, dict]:
    """Distort an image with an AugraphyPipeline."""
    image_array = np.array(image)
    out = pipeline.augment(image_array)
    distorted_image_array = out["output"]
    if not out["bounding_boxes"]:
        distorted_bbox = None
    else:
        distorted_bbox = out["bounding_boxes"][0]  # Only one line, so only one bounding box

    return Image.fromarray(distorted_image_array), distorted_bbox, pipeline.to_dict()


def maybe_copy(src: Path, dst: Path) -> None:
    """Copy the file if ``src`` and ``dst`` are different."""
    if src.resolve() != dst.resolve():
        copy2(src, dst)


def distort_line_images(
    line_images_dir: Path,
    output_dir: Path,
    rng: random.Random,
    pipeline_creator: PipelineCreator,
    distorted_subdir_name: str = "distorted_line_images",
    outline_bbox: bool = False,
) -> None:
    """Distort images and save the results."""
    metadata = pd.read_csv(line_images_dir / "metadata.csv")
    original_images_dir = line_images_dir / "line_images"
    if not original_images_dir.exists():
        raise ValueError(f"Directory {original_images_dir} does not exist.")

    distorted_images_dir = output_dir / distorted_subdir_name
    distorted_images_dir.mkdir(exist_ok=True, parents=True)

    logs_dir = output_dir / "augraphy_logs"
    logs_dir.mkdir(exist_ok=True, parents=True)

    distorted_metadata = metadata.copy()

    # TODO: should this read from metadata instead?
    line_image_paths = list(original_images_dir.glob("*.png"))

    for idx, row in tqdm(distorted_metadata.iterrows()):
        # Copy the image to the original images directory
        original_image_path = line_images_dir / row["undistorted_file_name"]

        image_path = output_dir / row["undistorted_file_name"]
        image_path.parent.mkdir(exist_ok=True, parents=True)
        maybe_copy(original_image_path, image_path)

        image = Image.open(image_path)
        bounding_box = (row["bbox_left"], row["bbox_top"], row["bbox_right"], row["bbox_bottom"])
        background_color = parse_int_tuple(row["background_color"])
        text_color = parse_int_tuple(row["text_color"])
        font_size = row["font_size"]

        pipeline = pipeline_creator(
            bbox=bounding_box,
            image_size=image.size,
            rng=rng,
            bleed_through_candidates=line_image_paths,
            text_color=text_color,
            background_color=background_color,
            font_size=font_size,
        )
        distorted_image, distorted_bbox, log = distort_line_image(image, pipeline)

        # Save the distorted image
        distorted_image_path = distorted_images_dir / image_path.name
        if outline_bbox:
            draw = ImageDraw.Draw(distorted_image)
            draw.rectangle((distorted_bbox), outline="red", width=2)
        distorted_image.save(distorted_image_path)

        # Save the log
        log_path = logs_dir / image_path.with_suffix(".json").name
        with open(log_path, "w") as f:
            json.dump(log, f)

        distorted_metadata.loc[idx, "undistorted_file_name"] = image_path.relative_to(output_dir)
        distorted_metadata.loc[idx, "file_name"] = distorted_image_path.relative_to(output_dir)
        distorted_metadata.loc[idx, "augraphy_log_path"] = log_path.relative_to(output_dir)

        if distorted_bbox is None:
            distorted_bbox = [np.nan, np.nan, np.nan, np.nan]
        distorted_metadata.loc[idx, "distorted_bbox_left"] = distorted_bbox[0]
        distorted_metadata.loc[idx, "distorted_bbox_top"] = distorted_bbox[1]
        distorted_metadata.loc[idx, "distorted_bbox_right"] = distorted_bbox[2]
        distorted_metadata.loc[idx, "distorted_bbox_bottom"] = distorted_bbox[3]

    distorted_metadata.to_csv(output_dir / "metadata.csv", index=False)
