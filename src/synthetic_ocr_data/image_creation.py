import hashlib
import logging
import random
import uuid
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from synthetic_ocr_data.color import (
    get_dark_mode_color_pair,
    get_light_mode_color_pair,
    get_random_dark_color,
    get_random_light_color,
    is_dark_mode,
)
from synthetic_ocr_data.fonts import FontInfo, check_font_support
from synthetic_ocr_data.image_processing import get_bounding_box_and_image_size
from synthetic_ocr_data.text_processing import TextLineType

logger = logging.getLogger(__name__)


def create_line_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    color_pair: tuple[tuple[int, int, int], tuple[int, int, int]],
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Create an image with text using a specific font, font size, color, backgroundcolor and margins."""
    # Create a font object

    # Define text and background colors
    text_color, background_color = color_pair

    # Get the bounding box and image size
    (left, top, right, bottom), (image_width, image_height) = get_bounding_box_and_image_size(
        text, font, top_margin, bottom_margin, left_margin, right_margin
    )

    # Create a new image with given size and background color
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Draw the text on the image
    draw.text((left_margin, top_margin), text, font=font, fill=text_color)

    return image, (left, top, right, bottom)


def get_suitable_font(
    text: str,
    font_infos: Sequence[FontInfo],
    size_range: range,
    rng: random.Random,
    max_tries: int = 10,
) -> ImageFont.FreeTypeFont:
    """Get a font that supports all characters in the text."""
    for i in range(max_tries):
        font_info = rng.choice(font_infos)
        font = font_info.load_font(rng, size_range)
        if check_font_support(text, font.path):
            return font
        logger.info(
            "Selected font '%s', but it does not support all characters in the text '%s'. Trying another font (attempt %d/%d).",
            font_info.name,
            text,
            i + 1,
            max_tries,
        )
    raise ValueError(f"No suitable font found for string {text} after {max_tries} tries.")


def create_line_images(
    text_lines: Iterable[TextLineType],
    output_dir: Path,
    fonts: Sequence[FontInfo],
    size_range: range,
    color_pairs: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]],
    top_margins: Sequence[int],
    bottom_margins: Sequence[int],
    left_margins: Sequence[int],
    right_margins: Sequence[int],
    rng: random.Random,
) -> None:
    """Create images with text and save metadata to a CSV file."""
    metadata = []

    for i, line in enumerate(text_lines):
        sha1 = hashlib.sha1(f"INDEX-{i}-LINE-{line.text}".encode()).digest()
        unique_id = uuid.UUID(bytes=sha1[:16], version=4)

        try:
            font = get_suitable_font(
                line.text,
                fonts,
                size_range,
                rng,
                100,
            )
        except ValueError:
            logger.exception(
                "Could not find a suitable font for text '%s'. Skipping line.", line.text
            )
            continue

        color_pair = rng.choice(color_pairs)
        top_margin = rng.choice(top_margins)
        bottom_margin = rng.choice(bottom_margins)
        left_margin = rng.choice(left_margins)
        right_margin = rng.choice(right_margins)
        image_dir = output_dir / "line_images"
        image_dir.mkdir(exist_ok=True, parents=True)
        output_path = image_dir / f"{unique_id}.jpg"

        image, bbox = create_line_image(
            text=line.text,
            font=font,
            color_pair=color_pair,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            left_margin=left_margin,
            right_margin=right_margin,
        )
        image_size = image.size

        # Save the image
        image.save(output_path, quality=90)

        # Store metadata
        metadata.append(
            {
                "unique_id": unique_id,
                "text": line.text,
                "text_line_id": line.text_line_id,
                "raw_text": line.raw_text,
                "text_transform": line.transform,
                "font_path": font.path,
                "font_size": font.size,
                "text_color": color_pair[0],
                "background_color": color_pair[1],
                "top_margin": top_margin,
                "bottom_margin": bottom_margin,
                "left_margin": left_margin,
                "right_margin": right_margin,
                "bbox_left": bbox[0],
                "bbox_top": bbox[1],
                "bbox_right": bbox[2],
                "bbox_bottom": bbox[3],
                "image_width": image_size[0],
                "image_height": image_size[1],
                "undistorted_file_name": output_path.relative_to(output_dir),
            }
        )
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)
