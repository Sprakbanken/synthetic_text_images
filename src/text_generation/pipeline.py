import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, Sequence

import cv2
import numpy as np
from augraphy import Augmentation, AugraphyPipeline, BleedThrough, augmentations

BoundingBox = tuple[int, int, int, int]


# TODO: types. the augment method returns more than just an image
class Augmenter(Protocol):
    def augment(self, image: np.ndarray) -> dict[str, Any]:
        """Apply augmentation to the given image and return the augmented image."""
        ...


@dataclass
class PipelineWrapper:
    ink_phase: list[Augmentation]
    paper_phase: list[Augmentation]
    post_phase: list[Augmentation]
    pre_phase: list[Augmentation]
    seed: int
    bounding_boxes: list[tuple[int, int, int, int]] = None

    @staticmethod
    def serialise_augmentation(augmentation) -> dict[str, Any]:
        return augmentation.__class__.__name__, augmentation.__dict__.copy()

    @staticmethod
    def deserialise_augmentation(name: str, params: dict[str, Any]) -> Augmentation:
        AugmentationType = getattr(
            augmentations,
            name,
        )  # TODO: Fallback dersom AugmentationType ikke finnes
        if AugmentationType is None:
            if name == "CustomImageBleedThrough":
                AugmentationType = CustomImageBleedThrough
            else:
                raise ValueError(f"Augmentation {name} not found")
        out = AugmentationType.__new__(AugmentationType)
        out.__dict__.update(params)
        return out

    def get_pipeline(self) -> AugraphyPipeline:
        return AugraphyPipeline(
            ink_phase=self.ink_phase,
            paper_phase=self.paper_phase,
            post_phase=self.post_phase,
            pre_phase=self.pre_phase,
            bounding_boxes=self.bounding_boxes,
            random_seed=self.seed,
        )

    def augment(self, image: np.ndarray):  # TODO: types
        pipeline = self.get_pipeline()
        random.seed(self.seed)
        np.random.seed(self.seed)
        cv2.setRNGSeed(self.seed)
        return pipeline.augment(image)

    def serialise(self) -> dict[str, Any]:
        return {
            "ink_phase": [self.serialise_augmentation(a) for a in self.ink_phase],
            "paper_phase": [self.serialise_augmentation(a) for a in self.paper_phase],
            "post_phase": [self.serialise_augmentation(a) for a in self.post_phase],
            "pre_phase": [self.serialise_augmentation(a) for a in self.pre_phase],
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            ink_phase=[cls.deserialise_augmentation(a, p) for a, p in data["ink_phase"]],
            paper_phase=[cls.deserialise_augmentation(a, p) for a, p in data["paper_phase"]],
            post_phase=[cls.deserialise_augmentation(a, p) for a, p in data["post_phase"]],
            pre_phase=[cls.deserialise_augmentation(a, p) for a, p in data["pre_phase"]],
            seed=data["seed"],
        )


class CustomImageBleedThrough(BleedThrough):
    def __init__(
        self,
        intensity_range=(0.1, 0.9),
        color_range=(0, 224),
        ksize=(17, 17),
        sigmaX=1,
        alpha=0.2,
        offsets=(20, 20),
        image_bleedthrough_foreground_path=None,
        p=1,
    ):
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.color_range = color_range
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.alpha = alpha
        self.offsets = offsets
        self.image_bleedthrough_foreground_path = image_bleedthrough_foreground_path

    def __call__(self, image, **kwargs):
        return image

    # create foreground image for bleedthrough effect
    def create_bleedthrough_foreground(self, image: np.ndarray):
        """Create foreground image for bleedthrough effect.

        :param image: The background image of the bleedthrough effect.
        :type image: numpy.array (numpy.uint8)
        """
        if self.image_bleedthrough_foreground_path is not None:
            image_bleedthrough_foreground = cv2.imread(self.image_bleedthrough_foreground_path)
            image_bleedthrough_foreground = cv2.resize(
                image_bleedthrough_foreground,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            image_bleedthrough_foreground = image

        # flip left-right only, flip top-bottom get inverted text, which is not realistic
        image_bleedthrough_foreground = cv2.flip(image_bleedthrough_foreground, 1)

        return image_bleedthrough_foreground


def create_scanned_book_pipeline(
    bbox: BoundingBox,
    rng: random.Random,
    bleed_through_candidates: Sequence[Path | str] | None = None,
) -> Augmenter:
    # TODO: find a bounding box safe translation
    ink_phase = [
        augmentations.Geometric(translation=(0.01, 0.01)),
    ]

    paper_phase = []

    post_phase = [
        augmentations.Folding(
            fold_x=None,
            fold_deviation=(0, 0),
            fold_count=rng.randrange(4, 7),
            fold_noise=0,
            fold_angle_range=(0, 0),
            gradient_width=(0.1, 0.1),
            gradient_height=(0.01, 0.01),
            backdrop_color=(0, 0, 0),
        )
    ]

    pipeline = PipelineWrapper(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        pre_phase=[],
        bounding_boxes=[bbox],
        seed=rng.randrange(0, 1000),
    )
    return pipeline
