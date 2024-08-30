import random
import uuid
from pathlib import Path
from typing import Generator, Sequence

import cv2
import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont
from PIL import Image, ImageColor, ImageDraw, ImageFont


def check_font_support(text: str, font_path: Path) -> bool:
    """Check if a font supports all characters in a text."""
    font = TTFont(font_path)
    cmap = font["cmap"].getBestCmap()

    for char in text:
        if ord(char) not in cmap:
            return False
    return True


def create_image_with_text(
    text: str,
    font_path: Path,
    font_size: int,
    color_pair: tuple[tuple[int, int, int], tuple[int, int, int]],
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
    output_path: Path,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Create an image with text using a specific font, font size, color, backgroundcolor and margins."""
    # Create a font object
    font = ImageFont.truetype(font_path, font_size)

    # Define text and background colors
    text_color, background_color = color_pair

    # Calculate text size
    (left, top, right, bottom) = font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    # Calculate image size with margins
    image_width = text_width + left_margin + right_margin
    image_height = text_height + top_margin + bottom_margin

    # Create a new image with white background
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Calculate text position
    text_x = left_margin
    text_y = top_margin

    # Draw the text on the image
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    # Save the image
    image.save(output_path)
    return (left, top, right, bottom), (image_width, image_height)


def create_random_dark_color(rng: random.Random) -> tuple[int, int, int]:
    """Create a random dark color."""
    return rng.randint(0, 100), rng.randint(0, 100), rng.randint(0, 100)


def create_random_light_color(rng: random.Random) -> tuple[int, int, int]:
    """Create a random light color."""
    return rng.randint(200, 255), rng.randint(200, 255), rng.randint(200, 255)


def create_datafiles(
    text_lines: Generator[str, None, None],
    output_dir: Path,
    font_paths: Sequence[Path],
    font_sizes: Sequence[int],
    color_pairs: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]],
    top_margins: Sequence[int],
    bottom_margins: Sequence[int],
    left_margins: Sequence[int],
    right_margins: Sequence[int],
    rng: random.Random,
) -> None:
    """Create images with text and save metadata to a CSV file."""
    metadata = []
    for id_, line in enumerate(text_lines):
        unique_id = uuid.uuid4()

        font_path = rng.choice(font_paths)
        if not check_font_support(line, font_path):
            raise ValueError(f"Font {font_path} does not support text: {line}")

        font_size = rng.choice(font_sizes)
        color_pair = rng.choice(color_pairs)
        top_margin = rng.choice(top_margins)
        bottom_margin = rng.choice(bottom_margins)
        left_margin = rng.choice(left_margins)
        right_margin = rng.choice(right_margins)
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True)
        output_path = image_dir / f"{unique_id}.png"

        bbox, image_size = create_image_with_text(
            text=line,
            font_path=font_path,
            font_size=font_size,
            color_pair=color_pair,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            left_margin=left_margin,
            right_margin=right_margin,
            output_path=output_path,
        )
        metadata.append(
            {
                "unique_id": unique_id,
                "text": line,
                "font_path": font_path,
                "font_size": font_size,
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
                "output_path": output_path.relative_to(output_dir),
            }
        )
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)
