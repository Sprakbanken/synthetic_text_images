import random
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

from fontTools.ttLib import TTFont
from PIL import ImageFont


@dataclass
class FontInfo:
    directory: Path
    name: str
    family: str

    @cached_property
    def font_files(self) -> list[Path]:
        return list(self.directory.glob("**/*.ttf"))

    def load_font(self, rng: random.Random, size_range: range) -> ImageFont:
        font_file = rng.choice(self.font_files)
        size = rng.choice(size_range)
        return ImageFont.truetype(font_file, size)

    @classmethod
    def from_record(cls, font_directory: Path, record: Mapping[str, str]) -> Self:
        return cls(
            directory=font_directory / record["directory"],
            name=record["name"],
            family=record["family"],
        )


def check_font_support(text: str, font_path: Path) -> bool:
    """Check if a font supports all characters in a text."""
    font = TTFont(font_path)
    cmap = font["cmap"].getBestCmap()

    for char in text:
        if ord(char) not in cmap:
            return False
    try:
        _ = ImageFont.truetype(font_path).getbbox(text)
    except OSError:
        # To handle the "OSError: execution context too long" error message
        return False
    return True
