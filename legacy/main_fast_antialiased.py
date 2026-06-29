import math
import time
from PIL import Image, ImageDraw
import numpy as np
import os
import sys


physicalSize = 12
pixelSize = 512
IMAGE_DIR = "images"
PRECALC_DIR = "precalc"
IMAGE_NAME = "tiger2.jpg"
image_path = os.path.join(IMAGE_DIR, IMAGE_NAME)
num_pegs = 180
extras = []  # [[163,461],[264,453],[212,237]]
string_color = 255 / 3.0
line_thickness = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0


pegs = [[
        pixelSize / 2 + pixelSize / 2 * math.cos(2 * math.pi * (x / num_pegs)),
        pixelSize / 2 + pixelSize / 2 * math.sin(2 * math.pi * (x / num_pegs))
    ]
    for x in range(num_pegs)]

pegs.extend(extras)
peg_pos = np.array(pegs, dtype=np.float32)


def preprocess(path, points):
    img = Image.open(path)
    img = img.crop(((img.width - img.height) // 2, 0, img.width - ((img.width - img.height) // 2), img.height))
    img = img.resize((pixelSize, pixelSize))
    copy = img.copy()
    img = img.convert('L')
    draw = ImageDraw.Draw(copy)
    r = 5
    for point in points:
        draw.ellipse([(point[0] - r, point[1] - r), (point[0] + r, point[1] + r)], fill='red')
    copy.save("temp.png")
    return img


def distance_line_indices_weights(x1, y1, x2, y2, size, thickness):
    # Mirrors the distance-based coverage logic from antialias.draw_line_distance,
    # but returns sparse indices/weights for fast cached rendering.
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    r = thickness / 2.0
    margin = r + 1.5

    ax = x2 - x1
    ay = y2 - y1
    seg_len_sq = ax * ax + ay * ay

    x_min = max(0, int(min(x1, x2) - margin))
    x_max = min(size, int(max(x1, x2) + margin) + 1)
    y_min = max(0, int(min(y1, y2) - margin))
    y_max = min(size, int(max(y1, y2) + margin) + 1)
    if x_min >= x_max or y_min >= y_max:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    px = np.arange(x_min, x_max, dtype=np.float32)[:, np.newaxis] + 0.5
    py = np.arange(y_min, y_max, dtype=np.float32)[np.newaxis, :] + 0.5
    vx = px - x1
    vy = py - y1

    if seg_len_sq < 1e-10:
        d = np.hypot(vx, vy)
    else:
        t = (vx * ax + vy * ay) / seg_len_sq
        t = np.clip(t, 0, 1)
        cx = x1 + t * ax
        cy = y1 + t * ay
        d = np.hypot(px - cx, py - cy)

    coverage = np.clip(r + 0.5 - d, 0, 1).astype(np.float32)
    x_idx, y_idx = np.nonzero(coverage > 0)
    if x_idx.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    x_coords = x_min + x_idx
    y_coords = y_min + y_idx
    valid = (x_coords >= 0) & (x_coords < size) & (y_coords >= 0) & (y_coords < size)
    if not np.any(valid):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    x_coords = x_coords[valid]
    y_coords = y_coords[valid]
    idxs = (x_coords + size * y_coords).astype(np.int32)
    weights = coverage[x_idx[valid], y_idx[valid]]
    return idxs, weights


def precalc_pixels():
    os.makedirs(PRECALC_DIR, exist_ok=True)
    filepath = os.path.join(PRECALC_DIR, f'precalc_pixels_{pixelSize}_{num_pegs}_dist_t{line_thickness:.3f}')
    try:
        loaded_vals = np.load(filepath + '_vals.npy')
        loaded_masks = np.load(filepath + '_masks.npy')
        loaded_counts = np.load(filepath + '_counts.npy')
        max_index = pixelSize * pixelSize
        if (
            loaded_vals.shape[:2] == (num_pegs, num_pegs)
            and loaded_masks.shape == loaded_vals.shape
            and loaded_counts.shape == (num_pegs, num_pegs)
            and np.max(loaded_vals, initial=0) < max_index
            and np.max(loaded_counts, initial=0) <= loaded_vals.shape[2]
        ):
            print('Loaded ' + filepath)
            return loaded_vals, loaded_masks, loaded_counts
        print('Cached precalc invalid, rebuilding...')
    except FileNotFoundError:
        pass

    counts = np.zeros((num_pegs, num_pegs), dtype=np.int32)
    max_line_pixels = 0

    for i in range(num_pegs):
        for j in range(num_pegs):
            if i == j:
                continue
            start = peg_pos[i]
            end = peg_pos[j]
            idxs, _ = distance_line_indices_weights(start[0], start[1], end[0], end[1], pixelSize, line_thickness)
            c = idxs.size
            counts[i, j] = c
            if c > max_line_pixels:
                max_line_pixels = c

    vals = np.zeros((num_pegs, num_pegs, max_line_pixels), dtype=np.int32)
    masks = np.zeros((num_pegs, num_pegs, max_line_pixels), dtype=np.float32)

    for i in range(num_pegs):
        for j in range(num_pegs):
            c = counts[i, j]
            if c <= 0:
                continue
            start = peg_pos[i]
            end = peg_pos[j]
            idxs, weights = distance_line_indices_weights(start[0], start[1], end[0], end[1], pixelSize, line_thickness)
            vals[i, j, :c] = idxs
            masks[i, j, :c] = weights

    np.save(filepath + '_vals.npy', vals)
    np.save(filepath + '_masks.npy', masks)
    np.save(filepath + '_counts.npy', counts)
    return vals, masks, counts


def try_line(pixels, pre_points):
    vals, masks, counts = pre_points
    pixels_dark = 255.0 - pixels
    darkness_vals = pixels_dark[vals] * masks
    darkness_sums = np.sum(darkness_vals, axis=-1, dtype=np.float32)

    darkness_avgs = np.full(darkness_sums.shape, -1.0, dtype=np.float32)
    np.divide(darkness_sums, counts, out=darkness_avgs, where=counts > 0)
    np.fill_diagonal(darkness_avgs, -1.0)

    return np.unravel_index(np.argmax(darkness_avgs, axis=None), darkness_avgs.shape)


def draw_line_antialiased(i, j, pre_points, pixels, new_pixels):
    vals, masks, counts = pre_points
    count = counts[i, j]
    if count <= 0:
        return

    line_vals = vals[i, j, :count]
    line_masks = masks[i, j, :count]
    delta = string_color * line_masks

    pixels[line_vals] = np.minimum(255.0, pixels[line_vals] + delta)
    new_pixels[line_vals] = np.maximum(0.0, new_pixels[line_vals] - delta)


def doIt():
    image_base = os.path.splitext(IMAGE_NAME)[0]
    filename = (
        'results/'
        + image_base
        + '_'
        + str(pixelSize)
        + '_'
        + str(num_pegs)
        + '_t'
        + str(line_thickness)
    )
    os.makedirs(filename, exist_ok=True)
    img = preprocess(image_path, extras)
    new_pixels = np.full((pixelSize * pixelSize), 255.0, dtype=np.float32)
    pixels = np.array(img, dtype=np.float32).flatten()

    precalc_start = time.time()
    print("Starting")
    points = precalc_pixels()
    precalc_end = time.time()
    print(f'precalc: {precalc_end - precalc_start}\n')

    loops_start = time.time()
    for i in range(5000):
        dark_pegs = try_line(pixels, points)
        print(f'loop {i}: {time.time() - loops_start:.3f}s')
        draw_line_antialiased(dark_pegs[0], dark_pegs[1], points, pixels, new_pixels)

        if ((i + 1) % 100) == 0:
            temp_img = Image.fromarray(new_pixels.reshape((pixelSize, pixelSize)).astype(np.uint8))
            temp_img.save(f'./{filename}/img_{i + 1}.png')


doIt()