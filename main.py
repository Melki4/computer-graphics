# import math
#
# import numpy as np
# from PIL import Image
#
# from math import cos, sin
#
# image = np.zeros((1000, 1000, 3), dtype=np.uint8)
#
# def draw_line(img_mat, x0, y0, x1, y1, color) :
#     step = 1.0/color
#     for t in np.arange(0, 1, step):
#         x = round((1.0-t)*x0 + t*x1)
#         y = round((1.0 - t) * y0 + t * y1)
#         image[y, x] = color
#
# def dotted_line(image, x0, y0, x1, y1, color):
#     count = math.sqrt((x0-x1)**2+(y0-y1)**2)
#     step = 1.0/count
#     for t in np.arange(0, 1, step):
#         x = round((1.0 - t) * x0 + t * x1)
#         y = round((1.0 - t) * y0 + t * y1)
#         image[y, x] = color
#
# def x_loop_line(image, x0, y0, x1,y1, color):
#
#     xchange = False
#
#     if abs(x0 - x1) < abs(y0 - y1):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
#
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#
#     for x in range(x0, x1):
#         t= (x-x0)/(x1-x0)
#         y = round((1.0 - t) * y0 + t * y1)
#         if xchange:
#             image[x, y] = color
#         else:
#             image[y, x] = color
#
# def one_more_func(image, x0, y0, x1, y1, color):
#
#     xchange = False
#
#     if abs(x0 - x1) < abs(y0 - y1):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
#
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#
#     y= y0
#     dy = 2.0*(x1-x0)*abs(y1-y0)/(x1-x0)
#     derror = 0.0
#     y_update = 1 if y1>y0 else -1
#
#     for x in range(x0, x1):
#
#         if xchange:
#             image[x, y] = color
#         else:
#             image[y, x] = color
#
#         derror+=dy
#         if(derror>2.0*(x1-x0)*0.5):
#             derror-=2.0*(x1-x0)*1.0
#             y+=y_update
#
# def bresenham_line(image, x0, y0, x1, y1, color):
#
#     xchange = False
#
#     if abs(x0 - x1) < abs(y0 - y1):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
#
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#
#     y = y0
#     dy = 2 * abs(y1 - y0)
#     derror = 0
#     y_update = 1 if y1 > y0 else -1
#
#     for x in range(x0, x1):
#
#         if xchange:
#             image[x, y] = color
#         else:
#             image[y, x] = color
#
#         derror += dy
#         if (derror > (x1 - x0)):
#             derror -= 2 * (x1 - x0)
#             y += y_update
#
# # for k in range(13):
# #     x0,y0=100,100
# #     x1 = int(100+95*cos(2*math.pi/13*k))
# #     y1 = int(100 + 95 * sin(2 * math.pi / 13 * k))
# #     bresenham_line(image, x0, y0, x1, y1, 255)
# #
# # img = Image.fromarray(image, mode='RGB')
# # img.save('img.png')
#
# def rabbit():
#
#     fin = open('model_1.obj', 'r')
#
#     array_of_v = list(list())
#     array_of_f = list(list())
#
#     import re
#
#     while (True):
#         boof = fin.readline()
#         if not boof : break
#         if boof[0] == 'v' and boof[1] == ' ' :
#             array_of_v.append(re.split('[ ,\n]', boof))
#         elif boof[0] == 'f' and boof[1] == ' ':
#             # print(boof)
#             array_of_f.append(re.split('[ ,\n]', boof))
#
#     fin.close()
#
#     for el in array_of_v:
#         image[int(9500 * float(el[1]) + 500), int(9500 * float(el[2])+50)] = 128
#
#     img = Image.fromarray(image, mode = 'RGB')
#     img = img.rotate(90)
#     img.save('img.png')
#
# def rabbit1():
#
#     fin = open('model_1.obj', 'r')
#
#     array_of_v = list(list())
#     array_of_f = list(list())
#
#     import re
#
#     while (True):
#         boof = fin.readline()
#         if not boof : break
#         if boof[0] == 'v' and boof[1] == ' ' :
#             array_of_v.append(re.split('[ , \n]', boof))
#         elif boof[0] == 'f' and boof[1] == ' ':
#
#             f = re.split('[ , \n]', boof)
#             face_vertices = list()
#             # print(f)
#             for el in f[1:4]:
#                 vertex_index = el.split('/')[0]
#                 face_vertices.append(int(vertex_index))
#             array_of_f.append(face_vertices)
#
#     fin.close()
#
#     for el in array_of_f:
#         bresenham_line(image, int(float(array_of_v[el[0] - 1][1]) * 9500 + 500),
#                        int(float(array_of_v[el[0] - 1][2]) * 9500 + 50),
#                        int(float(array_of_v[el[1] - 1][1]) * 9500 + 500),
#                        int(float(array_of_v[el[1] - 1][2]) * 9500+ 50), 255)
#         bresenham_line(image, int(float(array_of_v[el[1] - 1][1]) * 9500 + 500),
#                        int(float(array_of_v[el[1] - 1][2]) * 9500 + 50),
#                        int(float(array_of_v[el[2] - 1][1]) * 9500 + 500),
#                        int(float(array_of_v[el[2] - 1][2]) * 9500 + 50), 255)
#         bresenham_line(image, int(float(array_of_v[el[0] - 1][1]) * 9500 + 500),
#                        int(float(array_of_v[el[0] - 1][2]) * 9500 + 50),
#                        int(float(array_of_v[el[2] - 1][1]) * 9500 + 500),
#                        int(float(array_of_v[el[2] - 1][2]) * 9500 + 50), 255)
#
#     img = Image.fromarray(image, mode='RGB')
#     img = img.rotate(180)
#     img = img.transpose(Image.FLIP_LEFT_RIGHT)
#
#     # Покраска линий
#     img_array = np.array(img)
#     white_pixels = (img_array == [255, 255, 255]).all(axis=2)
#     img_array[white_pixels] = [128, 128, 128]
#     img = Image.fromarray(img_array)
#
#     img.save('img.png')
#
# rabbit1()