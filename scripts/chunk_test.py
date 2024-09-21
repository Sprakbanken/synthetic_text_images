from collections.abc import Iterable
import itertools
import logging
import random
from pathlib import Path
from typing import Callable, Generator


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
) -> Generator[str, None, None]:
    current_chunk = ""
    while True:
        logger.debug("Current chunk: %s", current_chunk)
        target_length = target_length_decider()
        logger.debug("Target length: %s", target_length)
        candidate_word = next(words, None)
        logger.debug("Candidate word: %s", candidate_word)
        if candidate_word is None:
            if len(current_chunk) >= min_length:
                yield current_chunk
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
                "Chunk is long enough without candidate word, and to long with candidate word"
                "adding current chunk to chunks and restarting current chunk"
            )
            yield current_chunk
            current_chunk = restart_chunk(candidate_word, max_length)


logger = logging.getLogger(__name__)
# Configure logging
# logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

sentences = [
    "lorem ipsum dolor sit amet",
    "hei på deg",
    "12345678910111213",
    "dette er en test",
    "hei",
    "på",
    "deg",
]

chain = itertools.chain.from_iterable((sentence.split() for sentence in sentences))
target_length = 35
min_length = 5
max_length = 100

current_chunk = ""


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


rng = random.Random(0)
target_length_decider = TargetLengthDecider(target_length, 10, min_length, max_length, rng)
chunks = list(chunkify_words(chain, min_length, max_length, target_length_decider))
# Add logging statements
logging.debug(f"Chunks: {chunks}")

print(chunks)

text_file_path = Path(__file__).parent / "lipsum.txt"
sentences = text_file_path.read_text().splitlines()
words = itertools.chain.from_iterable((sentence.split() for sentence in sentences))

chunks = list(chunkify_words(words, min_length, max_length, target_length_decider))
for chunk in chunks:
    print(chunk)
