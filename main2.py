
import matrix
from PIL import Image
import matplotlib.pyplot as plt

H=512
P=180

target = Image.open('lion.jpg').resize((H, H))
plt.imshow(target, cmap='gray')
plt.axis('off')
plt.show()


# M, solved = matrix.full_test(H, P, target)
M, solved = matrix.full_test(H, P, target, sparse=True)

data = matrix.render_image(M, solved, H, threshold=0.01, line_width=0.3)
plt.imshow(data, cmap='gray')
plt.axis('off')
plt.show()


