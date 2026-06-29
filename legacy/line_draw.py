from PIL import Image
import numpy as np

class wuzAlgorithm:
    def __init__(self, img, point1, point2, clr):
        self.image = img
        self.colour = clr
        self.x1, self.y1 = point1
        self.x2, self.y2 = point2
        self.dx, self.dy = self.x2 - self.x1, self.y2 - self.y1
        self.steep = abs(self.dy) > abs(self.dx)
        self.calc_details()
        self.startX, self.endX = self.calc_endPoint(point1) + 1, self.calc_endPoint(point2)
        self.draw_lines()

        
    def calc_points(self, xCoOrd, yCoOrd):
        return ((xCoOrd, yCoOrd), (yCoOrd, xCoOrd))[self.steep]


    def fPart(self, x):
        return x - int(x)


    def rfPart(self, x):
        return 1 - self.fPart(x)


    def calc_endPoint(self, point):
        x, y = point
        return int(round(x))


    def fill_colour(self, point, alpha=1):
        colour = tuple(map(lambda background, foreground: int(round(alpha * foreground + (1-alpha) * background)), self.image.getpixel(point), self.colour))
        self.image.putpixel(point, colour)


    
    def calc_details(self):
        if self.steep:
            self.x1, self.x2, self.y1, self.y2, self.dx, self.dy = self.y1, self.y2, self.x1, self.x2, self.dy, self.dx
        if self.x2 < self.x1:
            self.x1, self.x2, self.y1, self.y2 = self.x2, self.x1, self.y2, self.y1
        self.gradient = self.dy / self.dx
        self.yIntersection = self.y1 + self.rfPart(self.x1) * self.gradient


    def draw_lines(self):
        for x in range(self.startX, self.endX):
            y = int(self.yIntersection)
            self.fill_colour(self.calc_points(x, y), self.rfPart(self.yIntersection))
            self.fill_colour(self.calc_points(x, y + 1), self.fPart(self.yIntersection))
            self.yIntersection += self.gradient






def main():
    img = Image.new('RGB', (20,20), 'black')
    # nimg = numpy.
    wa = wuzAlgorithm(img, (3,3), (18,7), (255,255,255))
    img.save('out.png')

if __name__ == '__main__':
    main()