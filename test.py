from PIL import Image, ImageDraw
from antialias import draw_line_antialiased
import numpy as np

data = np.ones((400, 400), dtype=np.uint8) * 255
draw_line_antialiased(data, 20, 50, 200, 343, 0)
img = Image.fromarray(data, mode='L')

img.save('test.png')