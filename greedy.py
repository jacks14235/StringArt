"""Greedy forward selection for binary (0/1) line selection with antialiased rendering."""
import numpy as np
from preprocess import circle_crop
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

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


def greedy_solve(M, target, h, max_lines=None, line_width=1, verbose=True,
                 callback=None, callback_throttle=1, save_indices_path=None):
    """Select lines greedily (binary 0/1) to best approximate target.

    Uses precalc's M (antialiased line images) for fast composition.
    Returns (x, selected_indices): x is binary array, selected_indices is list of
    line indices in selection order (for recreating up to N lines).

    Args:
        M: Precomputed matrix from matrix.precalc (h² × n_lines).
        target: Target image (h, h), grayscale 0-255.
        h: Image size.
        max_lines: Stop after this many lines (default: no limit).
        line_width: Passed to render_image for final render (M uses precalc's width).
        verbose: Show progress bar.
        callback: Optional callable(img, n_lines) called after each update.
        callback_throttle: Call callback every N lines (1=every line, 5=every 5th, etc.).
        save_indices_path: If set, save selected indices (in order) to this path.
            Saved on completion and on KeyboardInterrupt.
    """
    target_flat = np.array(target).reshape(h * h).astype(np.float64)
    n_lines = M.shape[1]
    x = np.zeros(n_lines, dtype=bool)
    # Current composed image (float, 0-255) - start white
    current = np.full(h * h, 255.0, dtype=np.float64)
    best_error = _error(current, target_flat)
    selected = []

    # M is in darkness space (0=white, 255=black). Light factor = 1 - darkness/255.
    if hasattr(M, "toarray"):
        M = M.toarray()
    M_dark = np.asarray(M, dtype=np.float64)
    light_factor = 1.0 - M_dark / 255.0  # (h², n_lines): 1=no line, 0=full line
    light_factor_sq = light_factor * light_factor
    n_pixels = float(h * h)
    target_sq_sum = float(np.dot(target_flat, target_flat))

    def _save_indices():
        if save_indices_path and selected:
            np.save(save_indices_path, np.array(selected, dtype=np.int64))

    iterator = tqdm(iterable=range(n_lines), desc="greedy", disable=not verbose)
    try:
        for _ in iterator:
            if max_lines is not None and len(selected) >= max_lines:
                break

            available = ~x
            if not np.any(available):
                break

            # Vectorized candidate scoring:
            # err_j = mean((current * lf_j - target)^2)
            #       = (dot(current^2, lf_j^2) - 2*dot(current*target, lf_j) + dot(target, target)) / n_pixels
            current_sq = current * current
            current_target = current * target_flat
            s1 = current_sq @ light_factor_sq[:, available]
            s2 = (-2.0 * current_target) @ light_factor[:, available]
            candidate_errors = (s1 + s2 + target_sq_sum) / n_pixels

            best_pos = int(np.argmin(candidate_errors))
            best_new_error = float(candidate_errors[best_pos])
            if best_new_error >= best_error:
                break

            best_j = int(np.flatnonzero(available)[best_pos])
            x[best_j] = 1
            selected.append(best_j)
            current = current * light_factor[:, best_j]
            best_error = best_new_error
            iterator.set_postfix(lines=len(selected), mse=f"{best_error:.1f}")

            if callback and (len(selected) <= 1 or len(selected) % callback_throttle == 0):
                img = np.rint(current).clip(0, 255).astype(np.uint8).reshape((h, h))
                callback(img, len(selected))
    except KeyboardInterrupt:
        _save_indices()
        if verbose:
            print(f"\nInterrupted. Saved {len(selected)} indices to {save_indices_path}")
        raise

    _save_indices()
    return x.astype(np.uint8), selected


def _opencv_callback(window_name="greedy"):
    """Factory: returns a callback that displays via OpenCV. Call cv2.destroyAllWindows() when done."""
    def cb(img, n_lines):
        cv2.imshow(window_name, img)
        cv2.setWindowTitle(window_name, f"{n_lines} lines")
        cv2.waitKey(1)
    return cb


def indices_to_x(selected_indices, n_lines, n=None):
    """Convert ordered indices to binary x. Use n to recreate up to first n lines."""
    x = np.zeros(n_lines, dtype=np.uint8)
    indices = selected_indices[:n] if n is not None else selected_indices
    for i in indices:
        x[i] = 1
    return x


def run(h, num_pegs, target_image, max_lines=None, line_width=1,
        live_display=False, display_throttle=1, save_indices_path=None):
    """Convenience: precalc + greedy + render.

    Args:
        live_display: If True, show OpenCV window with live updates.
        display_throttle: Update display every N lines (1=every line). Requires live_display=True.
        save_indices_path: If set, save selected line indices (in order) to this path.
    """
    M = matrix.precalc(h, num_pegs, line_width=line_width)
    callback = None
    if live_display and cv2 is not None:
        callback = _opencv_callback()
    elif live_display and cv2 is None:
        raise ImportError("live_display requires opencv-python (pip install opencv-python)")
    x, selected = greedy_solve(M, np.array(target_image), h, max_lines=max_lines, line_width=line_width,
                               callback=callback, callback_throttle=display_throttle,
                               save_indices_path=save_indices_path)
    if live_display and cv2 is not None:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return M, x, matrix.render_image(M, x, h, threshold=0.5, line_width=line_width), selected


if __name__ == "__main__":
    H = 512
    P = 120
    target = Image.open('lion.jpg')
    target = circle_crop(target, H, 0).convert('L')
    M, x, img, selected = run(H, P, target, max_lines=4000, line_width=0.3,
                              live_display=True, display_throttle=1,
                              save_indices_path="greedy_indices.npy")
    print(f"Selected {len(selected)} lines")
    Image.fromarray(img).save("greedy_out.png")
