import random
import uuid
from pathlib import Path
from typing import Generator, Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from text_generation.image import get_bounding_box_and_image_size


def create_random_dark_color(rng: random.Random) -> tuple[int, int, int]:
    """Create a random dark color."""
    return rng.randint(0, 100), rng.randint(0, 100), rng.randint(0, 100)


def create_random_light_color(rng: random.Random) -> tuple[int, int, int]:
    """Create a random light color."""
    return rng.randint(200, 255), rng.randint(200, 255), rng.randint(200, 255)


def create_datafiles(
    text_lines: Generator[str, None, None],
    output_dir: Path,
    fonts: Sequence[ImageFont.FreeTypeFont],
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

        font = rng.choice(fonts)
        color_pair = rng.choice(color_pairs)
        top_margin = rng.choice(top_margins)
        bottom_margin = rng.choice(bottom_margins)
        left_margin = rng.choice(left_margins)
        right_margin = rng.choice(right_margins)
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True)
        output_path = image_dir / f"{unique_id}.png"

        image, bbox = create_line_image(
            text=line,
            font=font,
            color_pair=color_pair,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            left_margin=left_margin,
            right_margin=right_margin,
        )
        image_size = image.size

        # Save the image
        image.save(output_path)

        # Store metadata
        metadata.append(
            {
                "unique_id": unique_id,
                "text": line,
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
                "output_path": output_path.relative_to(output_dir),
            }
        )
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)


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

    # Create a new image with white background
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Draw the text on the image
    draw.text((left_margin, top_margin), text, font=font, fill=text_color)

    return image, (left, top, right, bottom)
