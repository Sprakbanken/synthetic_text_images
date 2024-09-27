from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
import subprocess
from typing import Self


def parse_int_tuple(rgb_string: str) -> tuple[int, int, int]:
    """
    Parses a string representing a tuple of integers.

    Parameters
    ----------
    rgb_string : str
        A string representing a tuple of integers, e.g., "(255, 0, 127)".

    Returns
    -------
    tuple of int
        A tuple of integers.

    Examples
    --------
    >>> parse_int_tuple("(255, 0, 127)")
    (255, 0, 127)
    >>> parse_int_tuple("255, 0, 127")
    (255, 0, 127)
    >>> parse_int_tuple("INVALID_STRING")
    Traceback (most recent call last):
        ...
    ValueError: Invalid input string: INVALID_STRING
    """
    match = re.match(r"\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?", rgb_string)
    if not match:
        raise ValueError(f"Invalid input string: {rgb_string}")
    return tuple(int(g) for g in match.groups())


class Language(StrEnum):
    SMA = "Southern Saami"
    SME = "Northern Saami"
    SMJ = "Lule Saami"
    SMN = "Inari Saami"


@dataclass
class GitInfo:
    commit: str
    submodule_commit: str
    language: Language
    submodule_repo: str

    @classmethod
    def from_language_code(cls, language: str) -> Self:
        language = Language[language.upper()]
        language_code = language._name_.lower()

        root_path = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True).stdout.decode().strip())
        submodule_path = root_path / "input" / "saami_ocr_nodalida25" / "raw" / f"corpus-{language_code}"

        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True).stdout.decode().strip()
        corpus_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=submodule_path, capture_output=True, check=True).stdout.decode().strip()
        
        submodule_repo = f"https://github.com/giellalt/corpus-{language_code}"

        return cls(commit=git_hash, submodule_commit=corpus_hash, language=language, submodule_repo=submodule_repo)
