from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from fontTools.ttLib import TTFont
import uuid
from typing import Generator, Sequence
import random
import pandas as pd

def check_font_support(text, font_path):
    font = TTFont(font_path)
    cmap = font['cmap'].getBestCmap()
    
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
        output_path: Path
    ) -> None:
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
    image = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    
    # Calculate text position
    text_x = left_margin
    text_y = top_margin
    
    # Draw the text on the image
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    
    # Save the image
    image.save(output_path)


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
    metadata = []
    for id_, line in enumerate(text_lines):
        unique_id = uuid.uuid4()

        font_path = rng.choice(font_paths)
        assert check_font_support(line, font_path)
        font_size = rng.choice(font_sizes)
        color_pair = rng.choice(color_pairs)
        top_margin = rng.choice(top_margins)
        bottom_margin = rng.choice(bottom_margins)
        left_margin = rng.choice(left_margins)
        right_margin = rng.choice(right_margins)
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True)
        output_path = image_dir / f"{unique_id}.png"

        create_image_with_text(
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
        metadata.append({
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
            "output_path": output_path.relative_to(output_dir),
        })
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)


if __name__ == "__main__":
    text_file_path = Path(__file__).parent / "example_text.txt"
    font_path = "fonts/Ubuntu_Sans/UbuntuSans-VariableFont_wdth,wght.ttf"
    font_paths = [Path(font_path)]
    text_lines = text_file_path.read_text(encoding="utf-8").splitlines()
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    font_sizes =  [16, 40]
    color_pairs = [((0, 0, 0), (255, 255, 255)), ((255, 255, 255), (0, 0, 0)), ((0, 0, 0), (255, 0, 0))]
    top_margins = [10, 20]
    bottom_margins = [10, 20]
    left_margins = [10, 20]
    right_margins = [10, 20]

    rng = random.Random(42)
    create_datafiles(
        text_lines=text_lines,
        output_dir=output_dir,
        font_paths=font_paths,
        font_sizes=font_sizes,
        color_pairs=color_pairs,
        top_margins=top_margins,
        bottom_margins=bottom_margins,
        left_margins=left_margins,
        right_margins=right_margins,
        rng=rng,
    )
