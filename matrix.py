import numpy as np
import math
from antialias import draw_line_distance
from tqdm import tqdm
from PIL import Image, ImageDraw


def _validate_colors(colors):
    if colors is None:
        return np.array([[0, 0, 0]], dtype=np.float64)
    arr = np.asarray(colors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("colors must be a 2D array with shape (n_colors, 3)")
    return np.clip(arr, 0, 255)


def _to_darkness_vector(image, h, color=False):
    """Convert uint8 image (0=black,255=white) to darkness vector (0=white,255=black)."""
    img = np.asarray(image, dtype=np.float64)
    if color:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("color=True requires an RGB image with shape (h, h, 3)")
        # Channel-major flatten: [R(h^2), G(h^2), B(h^2)]
        chans = [img[:, :, c].reshape((h**2,)) for c in range(3)]
        return np.concatenate([255.0 - c for c in chans], axis=0)
    return (255.0 - img).reshape((h**2,))


def _precalc_base_darkness(h, num_pegs, line_width=1):
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
    M = M.reshape((n_lines, h**2)).transpose()  # (h^2, n_lines)
    # Flip to darkness basis: sparse-friendly (background ~= 0)
    return (255 - M).astype(np.float64)


def precalc(h, num_pegs, line_width=1, sparse=False, color=False, colors=None):
    M = _precalc_base_darkness(h, num_pegs, line_width=line_width)
    if color:
        palette = _validate_colors(colors)
        n_colors = palette.shape[0]
        n_lines = M.shape[1]
        h2 = h ** 2
        channel_scale = 1.0 - (palette / 255.0)  # (n_colors, 3)

        M_color = np.zeros((h2 * 3, n_lines * n_colors), dtype=np.float64)
        for color_idx in range(n_colors):
            col_start = color_idx * n_lines
            col_end = col_start + n_lines
            for ch in range(3):
                row_start = ch * h2
                row_end = row_start + h2
                M_color[row_start:row_end, col_start:col_end] = M * channel_scale[color_idx, ch]
        M = M_color
    if sparse:
        from scipy.sparse import csc_matrix
        return csc_matrix(M)
    return M

def _line_pairs(num_pegs):
    pairs = []
    for i in range(num_pegs):
        for j in range(i + 1, num_pegs):
            pairs.append((i, j))
    return pairs


def render_image(M, x, h, threshold=0.5, line_width=1, color=False, colors=None):
    """Render by drawing lines with draw_line_distance. Overlapping lines stay dark (multiplicative)."""
    idx = np.where(x > threshold)[0]
    if len(idx) == 0:
        if color:
            return np.full((h, h, 3), 255, dtype=np.uint8)
        return np.full((h, h), 255, dtype=np.uint8)
    if color:
        palette = _validate_colors(colors)
        n_colors = palette.shape[0]
        n_lines = M.shape[1] // n_colors
    else:
        n_lines = M.shape[1]
    num_pegs = int((1 + math.sqrt(1 + 8 * n_lines)) / 2)
    peg_pos = np.array([
        [h / 2 + h / 2 * math.cos(2 * math.pi * (i / num_pegs)),
         h / 2 + h / 2 * math.sin(2 * math.pi * (i / num_pegs))]
        for i in range(num_pegs)
    ])
    pairs = _line_pairs(num_pegs)
    if color:
        canvas = np.ones((h, h, 3), dtype=np.uint8) * 255
        for k in idx:
            color_idx = k // n_lines
            line_idx = k % n_lines
            i, j = pairs[line_idx]
            start, end = peg_pos[i], peg_pos[j]
            for ch in range(3):
                channel_canvas = canvas[:, :, ch]
                # Convert white->thread-color per channel using same distance AA rasterizer.
                base = np.ones((h, h), dtype=np.uint8) * int(palette[color_idx, ch])
                temp = channel_canvas.copy()
                draw_line_distance(temp, start[0], start[1], end[0], end[1], thickness=line_width)
                mask = temp < 255
                channel_canvas[mask] = np.minimum(channel_canvas[mask], base[mask])
                canvas[:, :, ch] = channel_canvas
        return canvas
    else:
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


def render_exact(M, x, h, as_uint8=True, color=False):
    """Render the exact linear regression reconstruction (no thresholding).

    M is stored in darkness space (0=white, 255=black), so:
      darkness = M @ x
      image = 255 - darkness
    """
    darkness = M @ np.asarray(x, dtype=np.float64)
    darkness = np.asarray(darkness, dtype=np.float64)
    if color:
        h2 = h * h
        r = darkness[0:h2].reshape((h, h))
        g = darkness[h2:2 * h2].reshape((h, h))
        b = darkness[2 * h2:3 * h2].reshape((h, h))
        image = np.stack([255.0 - r, 255.0 - g, 255.0 - b], axis=-1)
    else:
        image = 255.0 - darkness.reshape((h, h))
    if as_uint8:
        return np.rint(np.clip(image, 0, 255)).astype(np.uint8)
    return image

def solve(M, image, h, color=False):
    """Least-squares solve for x in M @ x ≈ image (M is tall, so no exact solution)."""
    if hasattr(M, "toarray"):
        M = M.toarray()
    b = _to_darkness_vector(image, h, color=color)
    x, *_ = np.linalg.lstsq(M.astype(np.float64), b, rcond=None)
    return x


def solve_nnls(M, image, h, color=False):
    """Non-negative least-squares solve for x in M @ x ≈ image with x >= 0."""
    from scipy.optimize import nnls
    if hasattr(M, "toarray"):
        M = M.toarray()
    b = _to_darkness_vector(image, h, color=color)
    x, _ = nnls(M.astype(np.float64), b.astype(np.float64))
    return x


def solve_gpu(M, image, h, color=False):
    """Least-squares solve on GPU (CUDA if available). On MPS, lstsq is not implemented so we use CPU for the solve."""
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if hasattr(M, "toarray"):
        M = M.toarray()
    # lstsq is not implemented on MPS; running on CPU avoids NotImplementedError
    M_t = torch.from_numpy(M.astype(np.float64)).to(device)
    b_np = _to_darkness_vector(image, h, color=color)
    b = torch.from_numpy(b_np).to(device).unsqueeze(1)
    x, *_ = torch.linalg.lstsq(M_t, b, driver="gels")
    return x.squeeze(1).cpu().numpy()


def solve_sparse(M, image, h, show=False, color=False):
    """Least-squares solve using a sparse matrix and iterative solver (lsmr). Same x in M @ x ≈ image.
    Set show=True to print iteration progress (itn, normr) to the console."""
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import lsmr
    b = _to_darkness_vector(image, h, color=color)
    M_sp = csc_matrix(M.astype(np.float64))
    x = lsmr(M_sp, b, show=show)[0]
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

def full_test(h, num_pegs, target_image, sparse=False, line_width=0.3, color=False, colors=None):
    M = precalc(h, num_pegs, line_width=line_width, sparse=sparse, color=color, colors=colors)
    if sparse:
        solved = solve_sparse(M, np.array(target_image), h, color=color)
    else:
        solved = solve(M, np.array(target_image), h, color=color)
    return M, solved

def full_test_gpu(h, num_pegs, target_image, color=False, colors=None):
    M = precalc(h, num_pegs, line_width=0.3, color=color, colors=colors)
    solved = solve_gpu(M, np.array(target_image), h, color=color)
    return M, solved

def main():
    H = 512
    P = 90
    test = circle_test(H)
    test.save('test.png')
    # full_test(H, P, test)

if __name__ == '__main__':
    main()