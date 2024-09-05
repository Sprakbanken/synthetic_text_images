from pathlib import Path

import augraphy.augmentations as augmentations
import cv2
import pandas as pd
from augraphy import AugraphyPipeline
from PIL import Image


def create_pipeline_from_log(log):
    ink_phase = []
    paper_phase = []
    post_phase = []
    for i, (augmentation_name, augmentation_status, augmentation_parameters) in enumerate(
        zip(log["augmentation_name"], log["augmentation_status"], log["augmentation_parameters"])
    ):
        if augmentation_status:
            augmentation_type = getattr(augmentations, augmentation_name)
            post_phase.append(augmentation_type(**augmentation_parameters))
    return AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)


def draw_bounding_box_on_image(image, bounding_box):
    image = image.copy()
    left, top, right, bottom = bounding_box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return image


image_path = Path("scripts/output/images/f0588cf1-a6a1-4af5-8942-c9f6ab9451f0.png")
image = cv2.imread(str(image_path))
metadata = pd.read_csv("scripts/output/metadata.csv")
# Bounding box column: bbox_left,bbox_top,bbox_right,bbox_bottom
# id column: unique_id (same as image name)

image_metadata = metadata.query(f"unique_id == '{image_path.stem}'")

fold_x_rate = 0.3
fold_x = int(fold_x_rate * image.shape[1])


ink_phase = [augmentations.BrightnessTexturize(texturize_range=(0.9, 0.99), deviation=0.1)]

paper_phase = []

post_phase = [
    augmentations.Folding(
        fold_x=fold_x,
        fold_deviation=(0, 0),
        fold_count=1,
        fold_noise=0,
        fold_angle_range=(0, 0),
        gradient_width=(0.1, 0.1),
        gradient_height=(0.01, 0.01),
        backdrop_color=(0, 0, 0),
        p=1,
    )
]

bounding_boxes = image_metadata[
    ["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]
].values.tolist()
pipeline = AugraphyPipeline(
    ink_phase=ink_phase,
    paper_phase=paper_phase,
    post_phase=post_phase,
    bounding_boxes=bounding_boxes,
)

example_out = pipeline.augment(image)

# save image
augmented_image = example_out["output"]
augmented_image_pil = Image.fromarray(augmented_image)
augmented_image_pil.save("output.png")

log = example_out["log"]
print(log)

debug_image = draw_bounding_box_on_image(augmented_image, example_out["bounding_boxes"][0])
debug_image_pil = Image.fromarray(debug_image)
debug_image_pil.save("output_debug.png")

debug_output = draw_bounding_box_on_image(image, bounding_boxes[0])
debug_output_pil = Image.fromarray(debug_output)
debug_output_pil.save("input_debug.png")
