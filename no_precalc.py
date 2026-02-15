from asyncio import constants
import math
from PIL import Image, ImageDraw
import numpy as np
import time


physicalSize = 12
pixelSize = 720
image_path = 'dog.jpeg'
num_pegs = 180

peg_pos = np.array([[
        pixelSize / 2 + pixelSize / 2 * math.cos(2 * math.pi * (x / num_pegs)), 
        pixelSize / 2 + pixelSize / 2 * math.sin(2 * math.pi * (x / num_pegs))
    ]
    for x in range(num_pegs)])
            

def preprocess(path):
    img = Image.open(path)
    img = img.resize((pixelSize, pixelSize))
    img = img.convert('L')
    return img

# calculates which pixels each line goes through
def precalc_pixels():
    filepath = f'precacl_pixels_{pixelSize}_{num_pegs}.npy'
    try: 
        loaded = np.load(filepath)
        print('Loaded ' + filepath)
        return loaded
    except FileNotFoundError:
        pass
    vals = np.full((num_pegs, num_pegs, 2 * pixelSize), -1, dtype='int32')
    for i in range(num_pegs):
        for j in range(1, num_pegs):
            start = peg_pos[i]
            end = peg_pos[j]
            # print(start - end)
            dist = np.linalg.norm(start - end)
            points = np.rint(np.linspace(start, end, int(dist)))
            last = [-1, -1]
            k = 0
            for point in points:
                if (last[0] == point[0] and last[1] == point[1]):
                    continue
                if point[0] < 0 or point[0] >= pixelSize or point[1] < 0 or point[1] >= pixelSize:
                    continue
                vals[i][j][k] = point[0] + pixelSize * point[1]
                k+=1
                last = point
    np.save(filepath, vals)
    return vals


def try_line(pixels, pre_points):
    darkest = 2 * pixelSize * 255
    dark_pegs = [0,0]
    for i in range(num_pegs):
        for j in range(1, num_pegs):
            if (abs(i - j) < 4):
                continue
            start = peg_pos[i]
            end = peg_pos[j]
            dist = np.linalg.norm(start - end)
            points = pre_points[i][j]
            c = 0
            dark = -1
            for point in points:
                if point == -1:
                    break
                c = 1
                dark += pixels[point]
            dark /= dist
            if (c > 0 and dark < darkest):
                darkest = dark
                dark_pegs = [i, j]
    return dark_pegs

def draw_line(i, j, pre_points, pixels, newPixels):
    points = pre_points[i][j]
    print(i, j)
    print(peg_pos[i])
    print(peg_pos[j])
    print(points)
    for point in points:
        pixels[point] = 255
        newPixels[point] = 0



def test():
    img = preprocess(image_path)
    newPixels = np.full((pixelSize * pixelSize), 255, dtype='uint8')
    tempImg = Image.fromarray(newPixels, mode='L')
    pixels = np.array(img).flatten()

    precalc_start = time.time()
    points = precalc_pixels()
    precalc_end = time.time()
    # for i in range(180):
    #     draw_line(i, (i+1)%180, points, pixels, newPixels)
    draw_line(23, 27, points, pixels, newPixels)
    draw_line(22, 28, points, pixels, newPixels)
    draw_line(21, 30, points, pixels, newPixels)
    tempImg = Image.fromarray(newPixels.reshape((pixelSize, pixelSize)))
    tempImg.save(f'test.png')

def doIt():
    img = preprocess(image_path)
    newPixels = np.full((pixelSize * pixelSize), 255, dtype='uint8')
    tempImg = Image.fromarray(newPixels, mode='L')
    pixels = np.array(img).flatten()

    precalc_start = time.time()
    points = precalc_pixels()
    precalc_end = time.time()
    print(f'precalc: {precalc_end - precalc_start}\n')

    for i in range(500):
        precalc_start = time.time()
        dark_pegs = try_line(pixels, points)
        precalc_end = time.time()
        print(f'loop {i}: {precalc_end - precalc_start}')
        draw_line(dark_pegs[0], dark_pegs[1], points, pixels, newPixels)
        if ((i + 1) % 10) == 0:
            tempImg = Image.fromarray(newPixels.reshape((pixelSize, pixelSize)))
            tempImg.save(f'temp_{i + 1}.png')

test()