from pathlib import Path

from fontTools.ttLib import TTFont


def check_font_support(text: str, font_path: Path) -> bool:
    """Check if a font supports all characters in a text."""
    font = TTFont(font_path)
    cmap = font["cmap"].getBestCmap()

    for char in text:
        if ord(char) not in cmap:
            return False
    return True
