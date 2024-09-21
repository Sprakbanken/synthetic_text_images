import re
from typing import Tuple


def parse_int_tuple(rgb_string: str) -> Tuple[int, int, int]:
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