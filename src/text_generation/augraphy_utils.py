import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, Sequence, runtime_checkable

import cv2
import numpy as np
from augraphy import Augmentation, AugraphyPipeline, BleedThrough, augmentations

from text_generation.color import get_random_highlight_color
from text_generation.image_processing import get_bbox_aware_crop_box
from text_generation.image_creation import (
    is_dark_mode,
    get_random_dark_color,
    get_random_light_color,
)

BoundingBox = tuple[int, int, int, int]


# TODO: types. the augment method returns more than just an image
@runtime_checkable
class Augmenter(Protocol):
    def augment(self, image: np.ndarray) -> dict[str, Any]:
        """Apply augmentation to the given image and return the augmented image."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the Augmenter."""
        ... @ runtime_checkable


class PipelineCreator(Protocol):
    def __call__(
        self,
        bbox: BoundingBox,
        image_size: tuple[int, int],
        rng: random.Random,
        bleed_through_candidates: Sequence[Path | str] | None = None,
        background_color: tuple[int, int, int] | None = None,
    ) -> Augmenter: ...


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

    def augment(self, image: np.ndarray) -> dict[str, Any]:
        pipeline = self.get_pipeline()
        random.seed(self.seed)
        np.random.seed(self.seed)
        cv2.setRNGSeed(self.seed)
        return pipeline.augment(image)

    def to_dict(self) -> dict[str, Any]:
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


def maybe_append(list_: list, item: Any, probability: float, rng: random.Random) -> None:
    """Append an item to a list with a certain probability."""
    if rng.random() < probability:
        list_.append(item)


def create_scanned_book_pipeline(
    bbox: BoundingBox,
    image_size: tuple[int, int],
    rng: random.Random,
    text_color: tuple[int, int, int],
    background_color: tuple[int, int, int],
    bleed_through_candidates: Sequence[Path | str] = tuple(),
    font_size: int | None = None,
) -> Augmenter:
    bleed_through_alpha = min(max(rng.normalvariate(0.3, 0.4), 0), 0.8)

    if background_color is None:
        background_color = (0, 0, 0)
    if bleed_through_candidates:
        bleed_through_candidates = [str(p) for p in bleed_through_candidates]
        image_bleedthrough_foreground_path = rng.choice(bleed_through_candidates)
    else:
        image_bleedthrough_foreground_path = None

    pre_phase = []
    ink_phase = []
    paper_phase = []
    post_phase = []
    maybe_append(
        ink_phase,
        augmentations.InkBleed(intensity_range=(0.4, 0.7), kernel_size=(5, 5), severity=(0.2, 0.4)),
        probability=1,
        rng=rng,
    )

    if is_dark_mode(text_color=text_color, background_color=background_color):
        colorswap_color = get_random_light_color(rng=rng)
    else:
        colorswap_color = get_random_dark_color(rng=rng)

    if font_size and font_size > 40:
        maybe_append(  # TODO: only add this if font-size is large enlough
            ink_phase,
            augmentations.DotMatrix(
                dot_matrix_shape="circle",
                dot_matrix_dot_width_range=(1, 1),
                dot_matrix_dot_height_range=(1, 1),
                dot_matrix_min_width_range=(1, 1),
                dot_matrix_max_width_range=(50, 50),
                dot_matrix_min_height_range=(1, 1),
                dot_matrix_max_height_range=(50, 50),
                dot_matrix_min_area_range=(10, 10),
                dot_matrix_max_area_range=(800, 800),
                dot_matrix_median_kernel_value_range=(10, 10),
                dot_matrix_gaussian_kernel_value_range=(1, 1),
                dot_matrix_rotate_value_range=(0, 0),
            ),
            probability=0.1,
            rng=rng,
        )

    maybe_append(
        ink_phase,
        augmentations.Letterpress(
            n_samples=(200, 500),
            n_clusters=(300, 800),
            std_range=(1500, 5000),
            value_range=(0, 128),
            value_threshold_range=(128, 128),
            blur=1,
        ),
        probability=0.05,
        rng=rng,
    )

    maybe_append(
        paper_phase,
        CustomImageBleedThrough(
            intensity_range=(0, 0),
            color_range=(0, 100),
            ksize=(17, 17),
            sigmaX=0.1,
            alpha=bleed_through_alpha,
            offsets=(rng.randint(-10, 10), rng.randint(1 - 0, 10)),
            image_bleedthrough_foreground_path=image_bleedthrough_foreground_path,
            p=1,
        ),
        probability=0.2,
        rng=rng,
    )

    maybe_append(
        paper_phase,
        augmentations.WaterMark(
            watermark_word=rng.choice(["kopiija", "COPY"]),
            p=1,
        ),
        probability=0.05,
        rng=rng,
    )

    maybe_append(
        paper_phase,
        augmentations.NoiseTexturize(
            sigma_range=(2, 3),
            turbulence_range=(2, 5),
            texture_width_range=(50, 500),
            texture_height_range=(50, 500),
        ),
        probability=0.5,
        rng=rng,
    )

    maybe_append(post_phase, augmentations.Dithering(dither="floyd"), probability=0.05, rng=rng)

    maybe_append(
        post_phase,
        augmentations.DepthSimulatedBlur(
            blur_center="random",
            blur_major_axes_length_range=(120, 200),
            blur_minor_axes_length_range=(120, 200),
        ),
        probability=0.05,
        rng=rng,
    )

    markup_augmentation = rng.choice(
        [
            augmentations.Markup(
                num_lines_range=(1, 1),
                markup_length_range=(0.5, 1),
                markup_thickness_range=(2, 2),
                markup_type="underline",
                markup_ink="marker",
                markup_color=get_random_highlight_color(rng),
                repetitions=1,
                large_word_mode=True,
                single_word_mode=False,
            ),
            augmentations.Markup(
                num_lines_range=(5, 7),
                markup_length_range=(0.5, 1),
                markup_thickness_range=(1, 2),
                markup_type="strikethrough",
                markup_ink="pencil",
                markup_color=get_random_highlight_color(rng),
                repetitions=2,
                large_word_mode=True,
                single_word_mode=False,
            ),
            augmentations.Markup(
                num_lines_range=(1, 1),
                markup_length_range=(0.5, 1),
                markup_thickness_range=(5, 5),
                markup_type="highlight",
                markup_ink="highlighter",
                markup_color=get_random_highlight_color(rng),
                repetitions=1,
                large_word_mode=1,
                single_word_mode=False,
            ),
        ]
    )
    maybe_append(
        post_phase,
        markup_augmentation,
        probability=0.1,
        rng=rng,
    )

    maybe_append(
        post_phase,
        augmentations.DirtyDrum(
            line_width_range=(2, 5),
            line_concentration=0.3,
            direction=2,
            noise_intensity=0.4,
            noise_value=(0, 5),
            ksize=(3, 3),
            sigmaX=0,
        ),
        probability=0.1,
        rng=rng,
    )

    maybe_append(
        post_phase,
        augmentations.Folding(
            fold_x=None,
            fold_deviation=(0, 0),
            fold_count=rng.randrange(1, 4),
            fold_noise=0,
            fold_angle_range=(0, 20),
            gradient_width=(0.1, 0.1),
            gradient_height=(0.01, 0.025),
            backdrop_color=background_color,
        ),
        probability=0.2,
        rng=rng,
    )

    maybe_append(
        post_phase,
        augmentations.BadPhotoCopy(
            noise_type=rng.choice([1, 2]),
            noise_side=rng.choice(["left", "right"]),
            noise_iteration=(1, 1),
            noise_size=(1, 1),
            noise_sparsity=(0.4, 0.5),
            noise_concentration=(0.2, 0.2),
            blur_noise=1,
            blur_noise_kernel=(5, 5),
            wave_pattern=0,
            edge_effect=1,
        ),
        probability=0.1,
        rng=rng,
    )
    maybe_append(
        post_phase,
        augmentations.SubtleNoise(
            subtle_range=rng.randint(1, 10),
        ),
        probability=0.8,
        rng=rng,
    )
    maybe_append(
        post_phase,
        augmentations.SubtleNoise(
            subtle_range=rng.randint(20, 30),
        ),
        probability=0.2,
        rng=rng,
    )

    maybe_append(
        pre_phase,
        augmentations.Geometric(rotate_range=(-0.5, 0.5), padding_value=background_color),
        probability=0.5,
        rng=rng,
    )
    maybe_append(
        post_phase,
        augmentations.Geometric(
            crop=get_bbox_aware_crop_box(
                image_size=image_size,
                bounding_box=bbox,
                buffer_margin_left=-5,
                buffer_margin_top=-5,
                buffer_margin_right=-5,
                buffer_margin_bottom=-5,
            )
        ),
        probability=1,
        rng=rng,
    )
    maybe_append(post_phase, augmentations.Jpeg(quality_range=(50, 100)), probability=0.5, rng=rng)

    pipeline = PipelineWrapper(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        pre_phase=pre_phase,
        bounding_boxes=[bbox],
        seed=rng.randrange(0, 1000),
    )
    return pipeline
