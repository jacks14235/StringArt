
import matrix
from PIL import Image
import matplotlib.pyplot as plt
import os

H=256
P=90
LINE_WIDTH = 0.3
IMAGE_DIR = "images"
PRECALC_DIR = "precalc"
IMAGE_NAME = "lion.jpg"

image_path = os.path.join(IMAGE_DIR, IMAGE_NAME)
precalc_prefix = os.path.join(
    PRECALC_DIR,
    f"precalc_pixels_{H}_{P}_dist_t{LINE_WIDTH:.3f}",
)
print(f"image_path={image_path}")
print(f"precalc_prefix={precalc_prefix}")

target = Image.open(image_path).convert("L").resize((H, H))
plt.imshow(target, cmap='gray')
plt.axis('off')
plt.show()


# M, solved = matrix.full_test(H, P, target)
M, solved = matrix.full_test(H, P, target, sparse=False, line_width=LINE_WIDTH)

data = matrix.render_image(M, solved, H, threshold=0.01, line_width=LINE_WIDTH)
plt.imshow(data, cmap='gray')
plt.axis('off')
plt.show()


