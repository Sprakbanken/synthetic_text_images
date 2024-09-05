from augraphy import *
import cv2
import numpy as np

ink_phase   = [InkBleed(p=0.7),
               BleedThrough(p=0.7)]
paper_phase = [WaterMark(p=0.7),
               DirtyDrum(p=0.7)]
post_phase  = [DirtyRollers(p=0.7)]
pipeline    = AugraphyPipeline(ink_phase, paper_phase, post_phase, log=True, save_outputs=True)

image = np.full((1200, 1200,3), 250, dtype="uint8")
cv2.putText(
    image,
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
    (80, 250),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    0,
    3,
)
image_path = "scripts/output/images/4d4bbec8-9cfa-4e96-a5ff-b9fbcbf97ff3.png"
# image_path = "output/images/b0188314-29a1-4ea2-ab31-adfe0f38b74a.png"
image = cv2.imread(image_path)

result = pipeline.augment(image)
augmented_image = result["output"]
breakpoint()
