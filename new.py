from PIL import Image
import numpy as np
import math

def make_pegs(N, res):
    arr = np.array((N, *res))
    for i in range(N):
        for j in range(N):
            start = (math.cos(2 * math.pi * i / N), math.sin(2 * math.pi * i / N))
            end = (math.cos(2 * math.pi * j / N), math.sin(2 * math.pi * j / N))
    return arr

def start(res, pegs):
    img = Image.new('RGB', res)
    


def main():
    start((32, 32), 30)
    

if __name__ == '__main__':
    main()