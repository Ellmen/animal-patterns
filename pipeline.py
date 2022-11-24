import numpy as np
import cv2 as cv
import pygmsh

im = cv.imread('squid.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# cv.drawContours(im, contours, 1, (0,255,0), 3)

# Reduce number of vertices in polygon
# TODO: more intelligently
boundary = []
idx = 0
for p in contours[1]:
    if idx % 4 == 0:
        boundary.append(p[0])
    idx += 1

with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(
        # [
        #     [0.0, 0.0],
        #     [1.0, -0.2],
        #     [1.1, 1.2],
        #     [0.1, 0.7],
        # ],
        # [p[0] for p in contours[1]],
        boundary,
        # mesh_size=0.1,
        mesh_size=1,
    )
    # mesh = geom.generate_mesh(dim=2)
    # mesh = geom.generate_mesh(algorithm=3)
    mesh = geom.generate_mesh()

# print(mesh.points)
# mesh.points, mesh.cells, ...
mesh.write("squid.vtk")
# mesh.write("mesh.msh")
