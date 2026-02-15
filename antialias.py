"""Library to draw an antialiased line."""
# http://stackoverflow.com/questions/3122049/drawing-an-anti-aliased-line-with-thepython-imaging-library
# https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
import math
from PIL import Image
import numpy as np


def plot(data, x, y, c, col, steep, dash_interval):
    """Draws an antiliased pixel on a line."""
    w, h = data.shape
    if steep:
        x, y = y, x
    if x < w and y < h and x >= 0 and y >= 0:
        # c = c * (float(col[3]) / 255.0)
        p = data[x,y]
        x = int(x)
        y = int(y)
        data[x,y] = int((p * (1 - c)))


def iround(x):
    """Rounds x to the nearest integer."""
    return ipart(x + 0.5)


def ipart(x):
    """Floors x."""
    return math.floor(x)


def fpart(x):
    """Returns the fractional part of x."""
    return x - math.floor(x)


def rfpart(x):
    """Returns the 1 minus the fractional part of x."""
    return 1 - fpart(x)


def draw_line_antialiased(data, x1, y1, x2, y2, col, dash_interval=None):
    """Draw an antialised line in the PIL ImageDraw.

    Implements the Xialon Wu antialiasing algorithm.

    col - color
    """
    w, h = data.shape
    dx = x2 - x1
    if not dx:
        data[x1,:] = 0
        return

    dy = y2 - y1
    steep = abs(dx) < abs(dy)
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx
    if x2 < x1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    gradient = float(dy) / float(dx)

    # handle first endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = rfpart(x1 + 0.5)
    xpxl1 = xend    # this will be used in the main loop
    ypxl1 = ipart(yend)
    plot(data, xpxl1, ypxl1, rfpart(yend) * xgap, col, steep,
         dash_interval)
    plot(data, xpxl1, ypxl1 + 1, fpart(yend) * xgap, col, steep,
         dash_interval)
    intery = yend + gradient  # first y-intersection for the main loop

    # handle second endpoint
    xend = round(x2)
    yend = y2 + gradient * (xend - x2)
    xgap = fpart(x2 + 0.5)
    xpxl2 = xend    # this will be used in the main loop
    ypxl2 = ipart(yend)
    plot(data, xpxl2, ypxl2, rfpart(yend) * xgap, col, steep,
         dash_interval)
    plot(data, xpxl2, ypxl2 + 1, fpart(yend) * xgap, col, steep,
         dash_interval)

    # main loop
    for x in range(int(xpxl1 + 1), int(xpxl2)):
        plot(data, x, ipart(intery), rfpart(intery), col, steep,
             dash_interval)
        plot(data, x, ipart(intery) + 1, fpart(intery), col, steep,
             dash_interval)
        intery = intery + gradient


def draw_line_distance(data, x1, y1, x2, y2, thickness=1.0):
    """Draw an antialiased line with arbitrary thickness using distance-based rasterization.

    Args:
        data: 2D numpy array (grayscale, modified in-place). Higher values = lighter.
        x1, y1, x2, y2: Line endpoints in array coordinates.
        thickness: Line width in pixels (default 1.0).

    Uses vectorized numpy: only touches pixels in the line's bounding box + margin.
    """
    w, h = data.shape
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    r = thickness / 2.0
    margin = r + 1.5  # antialiasing falloff ~1 px

    # Segment vector and squared length
    ax, ay = x2 - x1, y2 - y1
    seg_len_sq = ax * ax + ay * ay

    # Bounding box (clamped to image)
    x_min = max(0, int(min(x1, x2) - margin))
    x_max = min(w, int(max(x1, x2) + margin) + 1)
    y_min = max(0, int(min(y1, y2) - margin))
    y_max = min(h, int(max(y1, y2) + margin) + 1)
    if x_min >= x_max or y_min >= y_max:
        return

    # Pixel grid in the ROI: data[x, y] convention
    nx, ny = x_max - x_min, y_max - y_min
    px = np.arange(x_min, x_max, dtype=np.float32)[:, np.newaxis] + 0.5  # (nx, 1)
    py = np.arange(y_min, y_max, dtype=np.float32)[np.newaxis, :] + 0.5  # (1, ny)

    # Vector from (x1,y1) to pixel
    vx = px - x1
    vy = py - y1

    if seg_len_sq < 1e-10:
        # Degenerate segment: treat as point
        d = np.hypot(vx, vy)
    else:
        # t = projection of P onto segment, clamped to [0,1]
        t = (vx * ax + vy * ay) / seg_len_sq
        t = np.clip(t, 0, 1)
        # Closest point on segment
        cx = x1 + t * ax
        cy = y1 + t * ay
        d = np.hypot(px - cx, py - cy)

    # Coverage: 1 inside, 0 outside, linear falloff over ~1 px at edge
    coverage = np.clip(r + 0.5 - d, 0, 1).astype(np.float32)

    # Blend: data = data * (1 - coverage), same convention as Wu algorithm
    roi = data[x_min:x_max, y_min:y_max].astype(np.float32, copy=False)
    np.multiply(roi, 1 - coverage, out=roi)
    data[x_min:x_max, y_min:y_max] = np.rint(roi).clip(0, 255).astype(data.dtype)


def main():
    img = np.ones((60, 60), dtype=np.uint8) * 255
    draw_line_distance(img, 5, 5, 55, 30, thickness=1)
    Image.fromarray(img).save('out.png')


if __name__ == '__main__':
    main()