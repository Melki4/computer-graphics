import math

import numpy as np
from PIL import Image
import random

from math import cos, sin

image = np.zeros((1000, 1000, 3), dtype=np.uint8)

boof_z = [[math.inf for j in range(1000)] for i in range(1000)]

def barichentr_coord(x,y,x0,y0, x1,y1,x2,y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 -lambda0 - lambda1
    return lambda0, lambda1, lambda2

def draw_triangle(image, x0,y0,x1,y1,x2,y2, color, z0, z1, z2):
    xmin = int(min(x0,x1,x2))
    if xmin<0 : xmin =0
    ymin = int(min(y0,y1,y2))
    if ymin < 0: ymin = 0
    xmax = int(max(x0, x1, x2))
    ymax = int(max(y0,y1,y2))

    for x in range (xmin, xmax+1):
        for y in range(ymin, ymax+1):
            array = barichentr_coord(x, y, x0, y0, x1, y1, x2, y2)
            boof = True
            for el in array :
                if el < 0 :
                    boof = False
                    continue
            if boof :

                newz = array[0]*z0 + array[1]*z1+ array[2]*z2

                if newz < boof_z[y][x] :
                    image[y, x] = [-255 * color, 0, 0]
                    boof_z[y][x] = newz

def rabbit1():

    fin = open('model_1.obj', 'r')

    array_of_v = list(list())
    array_of_f = list(list())

    import re

    while (True):

        boof = fin.readline()
        boof_z.append(math.inf)

        if not boof : break
        if boof[0] == 'v' and boof[1] == ' ' :
            array_of_v.append(re.split('[ , \n]', boof))
        elif boof[0] == 'f' and boof[1] == ' ':

            f = re.split('[ , \n]', boof)
            face_vertices = list()
            # print(f)
            for el in f[1:4]:
                vertex_index = el.split('/')[0]
                face_vertices.append(int(vertex_index))
            array_of_f.append(face_vertices)

    fin.close()

    array_of_normals = list()

    ind =0

    for el in array_of_f:

        a = np.array([float(array_of_v[el[0] - 1][1]) - float(array_of_v[el[1] - 1][1]),
                      float(array_of_v[el[0] - 1][2]) - float(array_of_v[el[1] - 1][2]),
                      float(array_of_v[el[0] - 1][3]) - float(array_of_v[el[1] - 1][3])])

        b = np.array([float(array_of_v[el[1] - 1][1]) - float(array_of_v[el[2] - 1][1]),
                      float(array_of_v[el[1] - 1][2]) - float(array_of_v[el[2] - 1][2]),
                      float(array_of_v[el[1] - 1][3]) - float(array_of_v[el[2] - 1][3])])

        f=np.cross(a, b)

        l = [0,0,1]

        coz= np.dot(f, l)/ np.linalg.norm(f)

        array_of_normals.append(f)

        if (coz < 0) :
            # print(float(array_of_v[el[0] - 1][3]),
            #               float(array_of_v[el[1] - 1][3]),
            #               float(array_of_v[el[2] - 1][3]), " ")

            draw_triangle(image, float(array_of_v[el[0] - 1][1]) * 9500 + 500,
                       float(array_of_v[el[0] - 1][2]) * 9500 + 50,
                       float(array_of_v[el[1] - 1][1]) * 9500 + 500,
                       float(array_of_v[el[1] - 1][2]) * 9500+ 50,
                       float(array_of_v[el[2] - 1][1]) * 9500 + 500,
                       float(array_of_v[el[2] - 1][2]) * 9500 + 50,
                          coz,
                          float(array_of_v[el[0] - 1][3]),
                          float(array_of_v[el[1] - 1][3]),
                          float(array_of_v[el[2] - 1][3]))
        ind+=1

    img = Image.fromarray(image, mode='RGB')
    img = img.rotate(180)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save('img.png')

rabbit1()
