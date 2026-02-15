import numpy as np
import math
from antialias import draw_line_distance
from tqdm import tqdm
from PIL import Image, ImageDraw

def precalc(h, num_pegs, line_width=1):
    peg_pos = np.array([[
        h / 2 + h / 2 * math.cos(2 * math.pi * (x / num_pegs)), 
        h / 2 + h / 2 * math.sin(2 * math.pi * (x / num_pegs))
    ]
    for x in range(num_pegs)])
    n_lines = num_pegs*(num_pegs-1)//2

    M = np.ones((n_lines, h, h), dtype=np.uint8) * 255
    print(f'initializing matrix of size {M.size}')
    index = 0
    for i in tqdm(range(num_pegs)):
        c = 0
        for j in range(i+1, num_pegs):
            start = peg_pos[i]
            end = peg_pos[j]
            draw_line_distance(M[index], start[0], start[1], end[0], end[1], line_width)
            index += 1
    M = M.reshape((n_lines, h**2))
    return M.transpose()

def render_image(M, x, h, threshold=0.5, line_width=1):
    """Render by drawing lines with draw_line_distance. Overlapping lines stay dark (multiplicative)."""
    idx = np.where(x > threshold)[0]
    if len(idx) == 0:
        return np.full((h, h), 255, dtype=np.uint8)
    n_lines = M.shape[1]
    num_pegs = int((1 + math.sqrt(1 + 8 * n_lines)) / 2)
    peg_pos = np.array([
        [h / 2 + h / 2 * math.cos(2 * math.pi * (i / num_pegs)),
         h / 2 + h / 2 * math.sin(2 * math.pi * (i / num_pegs))]
        for i in range(num_pegs)
    ])
    canvas = np.ones((h, h), dtype=np.uint8) * 255
    idx_set = set(idx)
    line_idx = 0
    for i in range(num_pegs):
        for j in range(i + 1, num_pegs):
            if line_idx in idx_set:
                start, end = peg_pos[i], peg_pos[j]
                draw_line_distance(canvas, start[0], start[1], end[0], end[1], thickness=line_width)
            line_idx += 1
    return canvas

def solve(M, image, h):
    """Least-squares solve for x in M @ x ≈ image (M is tall, so no exact solution)."""
    img = image.reshape((h**2,)).astype(np.float64)
    x, *_ = np.linalg.lstsq(M.astype(np.float64), img, rcond=None)
    return x


def solve_gpu(M, image, h):
    """Least-squares solve on GPU (CUDA if available). On MPS, lstsq is not implemented so we use CPU for the solve."""
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # lstsq is not implemented on MPS; running on CPU avoids NotImplementedError
    M_t = torch.from_numpy(M.astype(np.float64)).to(device)
    img = image.reshape((h**2,)).astype(np.float64)
    b = torch.from_numpy(img).to(device).unsqueeze(1)
    x, *_ = torch.linalg.lstsq(M_t, b, driver="gels")
    return x.squeeze(1).cpu().numpy()


def solve_sparse(M, image, h, show=False):
    """Least-squares solve using a sparse matrix and iterative solver (lsmr). Same x in M @ x ≈ image.
    Set show=True to print iteration progress (itn, normr) to the console."""
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import lsmr
    img = image.reshape((h**2,)).astype(np.float64)
    M_sp = csc_matrix(M.astype(np.float64))
    x = lsmr(M_sp, img, show=show)[0]
    return x


def rect_test(h):
    img = Image.new('L', (h, h), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((3*h//8, 0, 5*h//8, h), fill=0)
    return img

def circle_test(h):
    img = Image.new('L', (h, h), 255)
    draw = ImageDraw.Draw(img)
    draw.circle((h//2, h//4), h//6, fill=0)
    return img

def full_test(h, num_pegs, target_image, sparse=False):
    M = precalc(h, num_pegs, line_width=0.3)
    if sparse:
        solved = solve_sparse(M, np.array(target_image), h)
    else:
        solved = solve(M, np.array(target_image), h)
    return M, solved

def full_test_gpu(h, num_pegs, target_image):
    M = precalc(h, num_pegs, line_width=0.3)
    solved = solve_gpu(M, np.array(target_image), h)
    return M, solved

def main():
    H = 512
    P = 90
    test = circle_test(H)
    test.save('test.png')
    # full_test(H, P, test)

if __name__ == '__main__':
    main()