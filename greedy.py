"""Greedy forward selection for binary (0/1) line selection with antialiased rendering."""
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class _NoTqdm:
        def __init__(self, iterable, desc="", disable=False, **kw):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, **kw):
            pass
    tqdm = _NoTqdm

import matrix


def _error(rendered, target):
    """MSE between rendered and target (both flattened, float)."""
    return np.mean((rendered.astype(np.float64) - target.astype(np.float64)) ** 2)


def greedy_solve(M, target, h, max_lines=None, line_width=1, verbose=True):
    """Select lines greedily (binary 0/1) to best approximate target.

    Uses precalc's M (antialiased line images) for fast composition.
    Returns x: binary array, 1 = line selected.

    Args:
        M: Precomputed matrix from matrix.precalc (h² × n_lines).
        target: Target image (h, h), grayscale 0-255.
        h: Image size.
        max_lines: Stop after this many lines (default: no limit).
        line_width: Passed to render_image for final render (M uses precalc's width).
        verbose: Show progress bar.
    """
    target_flat = np.array(target).reshape(h * h).astype(np.float64)
    n_lines = M.shape[1]
    x = np.zeros(n_lines, dtype=np.uint8)
    # Current composed image (float, 0-255) - start white
    current = np.full(h * h, 255.0, dtype=np.float64)
    best_error = _error(current, target_flat)
    selected = []

    M_float = M.astype(np.float64) / 255.0  # columns as multiplicative factors

    iterator = tqdm(iterable=range(n_lines), desc="greedy", disable=not verbose)

    for _ in iterator:
        best_j = -1
        best_new_error = best_error

        for j in range(n_lines):
            if x[j] == 1:
                continue
            # Composing line j: current *= M[:, j] / 255
            new_img = current * M_float[:, j]
            err = _error(new_img, target_flat)
            if err < best_new_error:
                best_new_error = err
                best_j = j

        if best_j < 0 or (max_lines is not None and len(selected) >= max_lines):
            break
        if best_new_error >= best_error:
            break

        x[best_j] = 1
        selected.append(best_j)
        current = current * M_float[:, best_j]
        best_error = best_new_error
        iterator.set_postfix(lines=len(selected), mse=f"{best_error:.1f}")

    # Return binary x; render_image will use draw_line_distance for output
    return x


def run(h, num_pegs, target_image, max_lines=None, line_width=1):
    """Convenience: precalc + greedy + render."""
    M = matrix.precalc(h, num_pegs, line_width=line_width)
    x = greedy_solve(M, np.array(target_image), h, max_lines=max_lines, line_width=line_width)
    return M, x, matrix.render_image(M, x, h, threshold=0.5, line_width=line_width)


if __name__ == "__main__":
    target = matrix.circle_test(64)
    M, x, img = run(64, 24, target, max_lines=50)
    print(f"Selected {np.sum(x)} lines")
    from PIL import Image
    Image.fromarray(img).save("greedy_out.png")
