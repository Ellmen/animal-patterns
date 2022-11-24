import numpy as np
import cv2 as cv
import pygmsh

animal = 'squid'

im = cv.imread('{}.png'.format(animal))
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# cv.drawContours(im, contours, 1, (0,255,0), 3)

# Reduce number of vertices in polygon
# TODO: more intelligently
every_n = 1
boundary = []
idx = 0
for p in contours[1]:
    if idx % every_n == 0:
        boundary.append(list(p[0]))
    idx += 1

with open('{}_pts.py'.format(animal), 'w') as f:
    f.write('pnts = {}'.format(boundary))
