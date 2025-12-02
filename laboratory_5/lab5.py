import math
import re

import numpy as np

from PIL import Image

image = np.zeros((1000, 1000, 3), dtype=np.uint8)

boof_z = [[math.inf for j in range(1000)] for i in range(1000)]

def make_a_rabit():

    array_of_dot_coordinates = list(list())
    array_of_f = list(list())
    array_of_textures = list(list())

    fill_arrays(array_of_dot_coordinates, array_of_f, array_of_textures)

    make_rotation(array_of_dot_coordinates)
    main_work(array_of_dot_coordinates, array_of_f, array_of_textures)
    save_image()

def fill_arrays(array_of_v, array_of_f, array_of_t):
    fin = open('../model_1.obj', 'r')

    while (True):

        boof = fin.readline()

        if not boof : break
        if boof[0] == 'v' and boof[1] == ' ' :
            array_of_v.append(re.split('[ , \n]', boof))
        elif boof[0] == 'v' and boof[1] == 't' :
            array_of_t.append(re.split('[ , \n]', boof))
            #!!!!!!
        elif boof[0] == 'f' and boof[1] == ' ':
            f = re.split('[ , \n]', boof)

            face_vertices = list()
            for el in f[1:4]:
                vertex_index = el.split('/')[0]
                face_vertices.append(int(vertex_index))
            for el in f[1:4]:
                vertex_index = el.split('/')[1]
                face_vertices.append(int(vertex_index))
            array_of_f.append(face_vertices)

    fin.close()

def make_rotation(arrays_of_v):
    for i in range(len(arrays_of_v)):

        el = arrays_of_v[i]

        vector = np.array([float(el[1]), float(el[2]), float(el[3])])
        angleX = math.radians(0)
        angleY = math.radians(180)
        angleZ = math.radians(0)

        Rx = np.array([[1, 0, 0],
                       [0, math.cos(angleX), math.sin(angleX)],
                       [0, -math.sin(angleX), math.cos(angleX)]])
        Ry = np.array([[math.cos(angleY),0, math.sin(angleY)],
                       [0, 1, 0],
                       [-math.sin(angleY), 0, math.cos(angleY)]])
        Rz = np.array([[math.cos(angleZ), math.sin(angleZ), 0],
                       [-math.sin(angleZ), math.cos(angleZ), 0],
                       [0, 0, 1]])

        R=np.dot(Rx, Ry)
        R=np.dot(R, Rz)

        vector = np.dot(R, vector)

        move_a_litle(vector)

        el[1] = vector[0]
        el[2] = vector[1]
        el[3] = vector[2]

def move_a_litle(vector):
    vector[0] += 0.00
    vector[1] -= 0.05
    vector[2] += 0.5

def find_normals(array_of_dot_coordinates, array_of_f):
    array_of_dot_normals = [[0, 0, 0]]
    for i in range(len(array_of_dot_coordinates) - 1):
        array_of_dot_normals.append([0, 0, 0])

    for el in array_of_f:
        a = np.array([float(array_of_dot_coordinates[el[0] - 1][1]) - float(array_of_dot_coordinates[el[1] - 1][1]),
                      float(array_of_dot_coordinates[el[0] - 1][2]) - float(array_of_dot_coordinates[el[1] - 1][2]),
                      float(array_of_dot_coordinates[el[0] - 1][3]) - float(array_of_dot_coordinates[el[1] - 1][3])])

        b = np.array([float(array_of_dot_coordinates[el[1] - 1][1]) - float(array_of_dot_coordinates[el[2] - 1][1]),
                      float(array_of_dot_coordinates[el[1] - 1][2]) - float(array_of_dot_coordinates[el[2] - 1][2]),
                      float(array_of_dot_coordinates[el[1] - 1][3]) - float(array_of_dot_coordinates[el[2] - 1][3])])

        f = np.cross(a, b)  # нормаль

        array_of_dot_normals[el[0] - 1] += f
        array_of_dot_normals[el[1] - 1] += f
        array_of_dot_normals[el[2] - 1] += f

    for el in array_of_dot_normals:
        el /= np.linalg.norm(el)
    return array_of_dot_normals

def main_work(array_of_v, array_of_f, array_of_t):
    ind = 0

    image_texture = Image.open("bunny-atlas.jpg")
    image_texture = image_texture.rotate(180)
    image_texture = image_texture.transpose(Image.FLIP_LEFT_RIGHT)

    as_array = np.array(image_texture)

    v_n = find_normals(array_of_v, array_of_f)

    for el in array_of_f:

        a = np.array([float(array_of_v[el[0] - 1][1]) - float(array_of_v[el[1] - 1][1]),
                      float(array_of_v[el[0] - 1][2]) - float(array_of_v[el[1] - 1][2]),
                      float(array_of_v[el[0] - 1][3]) - float(array_of_v[el[1] - 1][3])])

        b = np.array([float(array_of_v[el[1] - 1][1]) - float(array_of_v[el[2] - 1][1]),
                      float(array_of_v[el[1] - 1][2]) - float(array_of_v[el[2] - 1][2]),
                      float(array_of_v[el[1] - 1][3]) - float(array_of_v[el[2] - 1][3])])

        f = np.cross(a, b)

        l = [0,0,1]

        I0 = np.dot(v_n[el[0]-1], l)
        I1 = np.dot(v_n[el[1]-1], l)
        I2 = np.dot(v_n[el[2]-1], l)

        coz = np.dot(f, l)/ np.linalg.norm(f)

        if (coz < 0) :
            # draw_triangle(image, float(array_of_v[el[0] - 1][1]),
            #            float(array_of_v[el[0] - 1][2]),
            #            float(array_of_v[el[1] - 1][1]),
            #            float(array_of_v[el[1] - 1][2]),
            #            float(array_of_v[el[2] - 1][1]),
            #            float(array_of_v[el[2] - 1][2]),
            #            I0, I1, I2,
            #            array_of_t[el[3]-1],
            #            array_of_t[el[4]-1],
            #            array_of_t[el[5]-1],
            #            float(array_of_v[el[0] - 1][3]),
            #            float(array_of_v[el[1] - 1][3]),
            #            float(array_of_v[el[2] - 1][3]),
            #           image_texture, as_array)

            draw_triangle_no_texture(image, float(array_of_v[el[0] - 1][1]),
                          float(array_of_v[el[0] - 1][2]),
                          float(array_of_v[el[1] - 1][1]),
                          float(array_of_v[el[1] - 1][2]),
                          float(array_of_v[el[2] - 1][1]),
                          float(array_of_v[el[2] - 1][2]),
                          I0, I1, I2,
                          float(array_of_v[el[0] - 1][3]),
                          float(array_of_v[el[1] - 1][3]),
                          float(array_of_v[el[2] - 1][3]),
                          image_texture)
        ind+=1

def draw_triangle(image, x0, y0, x1, y1, x2, y2, i0, i1, i2, u1, u2, u3, z0, z1, z2, image_texture, as_array):

    Wt = image_texture.width
    Ht = image_texture.height

    Size = 3500
    height = 1000
    width = 1000

    x0_s =(x0*Size)/z0 + width/2
    x1_s =(x1*Size)/z1 + width/2
    x2_s =(x2*Size)/z2 + width/2
    y0_s =(y0*Size)/z0 + height/2
    y1_s =(y1*Size)/z1 + height/2
    y2_s =(y2*Size)/z2 + height/2

    xmin = int(min(x0_s,x1_s,x2_s))
    if xmin<0 : xmin =0
    ymin = int(min(y0_s,y1_s,y2_s))
    if ymin < 0: ymin = 0
    xmax = int(max(x0_s, x1_s, x2_s))
    ymax = int(max(y0_s,y1_s,y2_s))

    for x in range (xmin, xmax+1):
        for y in range(ymin, ymax+1):
            array = barichentr_coord(x, y, x0_s, y0_s, x1_s, y1_s, x2_s, y2_s)
            boof = True
            for el in array :
                if el < 0 :
                    boof = False
                    continue
            if boof:
                newz = array[0]*z0 + array[1]*z1+ array[2]*z2
                if newz < boof_z[y][x] :
                    if (array[0] * i0 + array[1] * i1 + array[2] * i2) < 0 :
                        image[y, x] = (-(array[0] * i0 + array[1] * i1 + array[2] * i2) *
                                       as_array[int(Wt*(array[0]*float(u1[2]) +
                                       array[1]*float(u2[2]) +
                                       array[2]*float(u3[2])))]
                                       [int(Ht*(array[0]*float(u1[1]) +
                                       array[1]*float(u2[1]) +
                                       array[2]*float(u3[1])))])
                    boof_z[y][x] = newz

def draw_triangle_no_texture(image, x0, y0, x1, y1, x2, y2, i0, i1, i2, z0, z1, z2, image_texture):

    Wt = image_texture.width
    Ht = image_texture.height

    Size = 3500
    height = 1000
    width = 1000

    x0_s =(x0*Size)/z0 + width/2
    x1_s =(x1*Size)/z1 + width/2
    x2_s =(x2*Size)/z2 + width/2
    y0_s =(y0*Size)/z0 + height/2
    y1_s =(y1*Size)/z1 + height/2
    y2_s =(y2*Size)/z2 + height/2

    xmin = int(min(x0_s,x1_s,x2_s))
    if xmin<0 : xmin =0
    ymin = int(min(y0_s,y1_s,y2_s))
    if ymin < 0: ymin = 0
    xmax = int(max(x0_s, x1_s, x2_s))
    ymax = int(max(y0_s,y1_s,y2_s))

    for x in range (xmin, xmax+1):
        for y in range(ymin, ymax+1):
            array = barichentr_coord(x, y, x0_s, y0_s, x1_s, y1_s, x2_s, y2_s)
            boof = True
            for el in array :
                if el < 0 :
                    boof = False
                    continue
            if boof:
                newz = array[0]*z0 + array[1]*z1+ array[2]*z2
                if newz < boof_z[y][x] :
                    if (array[0] * i0 + array[1] * i1 + array[2] * i2) < 0 :
                        image[y, x] = (-255*(array[0] * i0 + array[1] * i1 + array[2] * i2))
                    boof_z[y][x] = newz

def barichentr_coord(x,y,x0,y0, x1,y1,x2,y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 -lambda0 - lambda1
    return lambda0, lambda1, lambda2

def save_image():
    img = Image.fromarray(image, mode='RGB')
    img = img.rotate(180)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save('img4.png')

make_a_rabit()