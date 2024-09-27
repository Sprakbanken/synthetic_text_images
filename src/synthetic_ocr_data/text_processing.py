import itertools
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Protocol
import uuid

logger = logging.getLogger(__name__)
# Configure logging
# logger.setLevel(logging.DEBUG)


def clean(text: str) -> str:
    # Copied from the github.com/sprakbanken/sami_ocr repository
    bad_chars_and_replacements = [
        # replace non-breaking space with normal space
        ("\xa0", " "),
        # replace variants of quotation marks
        ("«", '"'),
        ("»", '"'),
        ("”", '"'),
        # replace variants of apostrophe
        ("ʼ", "'"),
        ("’", "'"),
        ("ʹ", "'"),
        # replace Ds
        ("Ð", "Đ"),
        ("Ɖ", "Đ"),
        # replace em dash with en dash
        ("—", "–"),
    ]
    for bad_char, replacement in bad_chars_and_replacements:
        text = text.replace(bad_char, replacement)
    return text


class TextLineType(Protocol):
    text: str
    text_line_id: str
    raw_text: str
    transform: str


@dataclass
class TextLine:
    text: str
    text_line_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def raw_text(self) -> str:
        return self.text

    @property
    def transform(self) -> str:
        return "identity"


@dataclass
class TransformedTextLine:
    raw_text: str
    text: str
    text_line_id: str
    transform: str


def calculate_candidate_length(current_chunk: str, candidate_word: str) -> int:
    if current_chunk:
        return len(current_chunk) + 1 + len(candidate_word)
    else:
        return len(candidate_word)


def add_word_to_chunk(current_chunk: str, candidate_word: str) -> str:
    if current_chunk:
        return current_chunk + " " + candidate_word
    else:
        return candidate_word


def restart_chunk(candidate_word: str, max_length: str) -> str:
    if len(candidate_word) > max_length:
        logger.debug(
            "candidate word (%s) too long, discarding and starting the new chunk empty",
            candidate_word,
        )
        return ""
    else:
        logger.debug("Starting new chunk with candidate word (%s) ", candidate_word)
        return candidate_word


def chunkify_words(
    words: Iterable[str],
    min_length: int,
    max_length: int,
    target_length_decider: Callable[[], int],
) -> Generator[TextLine, None, None]:
    current_chunk = ""
    words = iter(words)
    while True:
        logger.debug("Current chunk: %s", current_chunk)
        target_length = target_length_decider()
        logger.debug("Target length: %s", target_length)
        candidate_word = next(words, None)
        logger.debug("Candidate word: %s", candidate_word)
        if candidate_word is None:
            if len(current_chunk) >= min_length:
                yield TextLine(text=current_chunk, text_line_id=str(uuid.uuid4()))
            else:
                logger.debug("Last chunk too short, discarding")
            break

        if calculate_candidate_length(current_chunk, candidate_word) <= target_length:
            logger.debug("Chunk within target length, adding candidate word to chunk")
            current_chunk = add_word_to_chunk(current_chunk, candidate_word)

        elif len(current_chunk) < min_length:
            logger.debug("Chunk too short without candidate word")
            if calculate_candidate_length(current_chunk, candidate_word) <= max_length:
                logger.debug("Candidate word within max length, adding candidate word")
                current_chunk = add_word_to_chunk(current_chunk, candidate_word)
            else:
                logger.debug("Candidate word too long, restarting chunk")
                current_chunk = restart_chunk(candidate_word, max_length)
        else:
            logger.debug(
                "Chunk is long enough without candidate word, and too long with candidate word"
                "adding current chunk to chunks and restarting current chunk"
            )
            yield TextLine(text=current_chunk, text_line_id=str(uuid.uuid4()))
            current_chunk = restart_chunk(candidate_word, max_length)


class TargetLengthDecider:
    def __init__(self, mu: int, sigma: int, min_length: int, max_length: int, rng: random.Random):
        self.mu = mu
        self.sigma = sigma
        self.min_length = min_length
        self.max_length = max_length
        self.rng = rng

    def __call__(self) -> int:
        target_length = int(self.rng.gauss(self.mu, self.sigma))
        clipped_target_length = max(self.min_length, min(self.max_length, target_length))
        return clipped_target_length


def add_transformed_text_lines(
    chunks: Iterable[TextLine], *transforms: Callable[[str], str]
) -> Generator[TransformedTextLine, None, None]:
    """Transforms the text in the chunks using the provided transform function."""
    for chunk in chunks:
        yield TransformedTextLine(
            raw_text=chunk.text,
            text=chunk.text,
            text_line_id=chunk.text_line_id,
            transform="identity",
        )

        for transform in transforms:
            transformed = transform(chunk.text)
            if transformed != chunk.text:
                yield TransformedTextLine(
                    raw_text=chunk.text,
                    text=transformed,
                    text_line_id=chunk.text_line_id,
                    transform=transform.__name__,
                )
